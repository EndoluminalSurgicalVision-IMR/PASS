# -*- coding:utf-8 -*-
"""
Test-time adaptable neural networks for robust medical image segmentation
Step1. Source joint training: Normalization module + Seg-Unet + DAE
"""
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import source_collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from models.unet import UNet_v2, Norm_Indentity_Net, U_Net_for_DAE
from torchvision.transforms import transforms
from utils.losses.seg_loss import DiceLoss, CELoss, Dice_CE_Loss, BCEDiceLoss
from utils.init import get_logger
import numpy as np
from PIL import Image
import copy

def save_pred2image(tensor, name, path):
    """
    Save a tensor image to the path.
    tensor : [C, H, W]
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    # file_path = path + '/' + name + '.png'
    # Tensor to PIL.Image
    tensor = (tensor > 0.5).to(torch.float32)
    # tensor = (tensor- tensor.min()) / (tensor.max() - tensor.min())
    PIL = transforms.ToPILImage()(tensor[0,:,:].unsqueeze(0).cpu())
    file_0_path = path + '/' + name +'c0'+ '.png'
    PIL.save(file_0_path)

    PIL = transforms.ToPILImage()(tensor[1, :, :].unsqueeze(0).cpu())
    file_1_path = path + '/' + name + 'c1' + '.png'
    PIL.save(file_1_path)

def save_pred2image_onelabel(tensor, name, path):
    """
    Save a tensor image to the path.
    tensor : [C, H, W]
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    # file_path = path + '/' + name + '.png'
    # Tensor to PIL.Image
    tensor = (tensor > 0.5).to(torch.float32)
    # tensor = (tensor- tensor.min()) / (tensor.max() - tensor.min())
    PIL = transforms.ToPILImage()(tensor.cpu())
    file_0_path = path + '/' + name + '.png'
    PIL.save(file_0_path)

def mask_random_replace(x, cnt=20):
    x_shape = x.shape
    if len(x_shape) == 3:
        ### x: [C, H, W]
        _, img_rows, img_cols = x_shape
        x_trans = x.clone()
        # cnt = 5
        while cnt > 0:
            block_noise_size_x = random.randint(img_rows // 32, img_rows // 16)
            block_noise_size_y = random.randint(img_cols // 32, img_cols // 16)

            noise_x1 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y1 = random.randint(3, img_cols - block_noise_size_y - 3)

            noise_x2 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y2 = random.randint(3, img_cols - block_noise_size_y - 3)

            # print('--------', noise_x1, noise_x1 + block_noise_size_x, noise_y1, noise_y1 + block_noise_size_y)

            x_trans[:,noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y] \
                = x[:, noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y]
           #  x_trans[
           # :,
           #  noise_y2:noise_y2 + block_noise_size_y, noise_z2] = x[
           #                                           :,
           #                                            noise_y1:noise_y1 + block_noise_size_y, noise_z1]
            cnt -= 1


    else:
        ### x: [C, H, W, D]
        _, img_rows, img_cols, img_deps = x_shape
        cnt = 30
        x_trans = x.clone()
        while cnt > 0:
            block_noise_size_x = random.randint(img_rows // 32, img_rows // 16)
            block_noise_size_y = random.randint(img_cols // 32, img_cols // 16)

            noise_x1 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y1 = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z1 = random.randint(0, x.shape[2])

            noise_x2 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y2 = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z2 = random.randint(0, x.shape[2])

            x_trans[:,noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y, noise_z1] \
                = x[:, noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y, noise_z1]
            x_trans[:, noise_x1:noise_x1 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y, noise_z2]\
                = x[:, noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y, noise_z2]
            cnt -= 1

    return x_trans


def train_RIGA(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = 20
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle

    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    # print('stage1', tr_label_list)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    # print('stage2', ts_label_list)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=source_collate_fn_tr)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads // 2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    Norm_model = Norm_Indentity_Net()
    Seg_model = UNet_v2()
    DAE = U_Net_for_DAE(img_ch=args.num_classes, output_ch=args.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Norm_model = nn.DataParallel(Norm_model, device_ids=gpu).to(device)
    Seg_model = nn.DataParallel(Seg_model, device_ids=gpu).to(device)
    DAE = nn.DataParallel(DAE, device_ids=gpu).to(device)

    optimizer = torch.optim.SGD(
            list(Norm_model.parameters()) + list(Seg_model.parameters()), lr=initial_lr,
        momentum=0.99, nesterov=True)

    dae_optimizer = torch.optim.SGD(
        DAE.parameters(),
        lr=initial_lr,  momentum=0.99, nesterov=True)


    start_epoch = 0
    if continue_training:
        # Try loading the latest model checkpoint, fallback to the final checkpoint if not found
        try:
            params = torch.load(join(model_folder, 'model_latest.model'))
        except FileNotFoundError:
            assert isfile(join(model_folder, 'model_final.model')), 'Missing model checkpoint!'
            params = torch.load(join(model_folder, 'model_final.model'))

        Norm_model.load_state_dict(params['norm_model_state_dict'])
        Seg_model.load_state_dict(params['seg_model_state_dict'])
        DAE.load_state_dict(params['dae_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        dae_optimizer.load_state_dict(params['dae_optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler1 = GradScaler()
    amp_grad_scaler2 = GradScaler()
    rec_criterion = nn.MSELoss(reduction='mean')
    seg_criterion = nn.BCEWithLogitsLoss()

    start = time()
    best_metric = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        Norm_model.train()
        Seg_model.train()
        DAE.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_seg_loss_list = list()
        train_dae_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            optimizer.zero_grad()
            with autocast():
                norm_data = Norm_model(data)
                output = Seg_model(norm_data)
                loss = seg_criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + seg_criterion(output[:, 1], (seg[:, 0] == 2)*1.0)
            amp_grad_scaler1.scale(loss).backward()
            amp_grad_scaler1.unscale_(optimizer)
            amp_grad_scaler1.step(optimizer)
            amp_grad_scaler1.update()
            train_seg_loss_list.append(loss.detach().cpu().numpy())

            dae_optimizer.zero_grad()
            with autocast():
                output = torch.sigmoid(output).detach()
                disrupted_output = torch.zeros_like(output)
                for i in range(output.shape[0]):
                    disrupted_output[i] = mask_random_replace(output[i])
                rec_output = DAE(disrupted_output)
                rec_output = torch.clamp(rec_output, 0., 1.)
                # print(disrupted_output.size(), rec_output.size(), output.size())
                # print(rec_output.min(), rec_output.max())
                # mse_loss = rec_criterion(rec_output, output)


                one_hot_seg = torch.zeros_like(output)
                one_hot_seg[:, 0] = (seg[:, 0] > 0) * 1.0
                one_hot_seg[:, 1] = (seg[:, 0] == 2) * 1.0

                disrupted_seg = torch.zeros_like(one_hot_seg)
                for i in range(one_hot_seg.shape[0]):
                    disrupted_seg[i] = mask_random_replace(one_hot_seg[i])
                rec_seg = DAE(disrupted_seg)

                mse_loss = (rec_criterion(rec_output, output) + rec_criterion(rec_seg, one_hot_seg)) / 2

            amp_grad_scaler2.scale(mse_loss).backward()
            amp_grad_scaler2.unscale_(dae_optimizer)
            amp_grad_scaler2.step(dae_optimizer)
            amp_grad_scaler2.update()
            train_dae_loss_list.append(mse_loss.detach().cpu().numpy())

        mean_tr_seg_loss = np.mean(train_seg_loss_list)
        mean_tr_dae_loss = np.mean(train_dae_loss_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Seg Loss", mean_tr_seg_loss, epoch)
        writer.add_scalar("Train Scalars/Train DAE Loss", mean_tr_dae_loss, epoch)
        print('  Tr Seg loss: {}\n'.format(mean_tr_seg_loss))
        print('  Tr DAE loss: {}\n'.format(mean_tr_dae_loss))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'seg_model_state_dict': Seg_model.state_dict(),
                'norm_model_state_dict': Norm_model.state_dict(),
                'dae_state_dict': DAE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dae_optimizer_state_dict': dae_optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
            save_dir = visualization_folder+'/epoch-'+str(epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for i in range(output.shape[0]):
                save_pred2image(output[i], name=str(i)+'_pred', path=save_dir)
                save_pred2image(rec_output[i], name=str(i) + '_rec_pred', path=save_dir)
                save_pred2image(disrupted_output[i], name=str(i) + '_disrupt_pred', path=save_dir)
                save_pred2image(one_hot_seg[i], name=str(i) + '_seg', path=save_dir)
                save_pred2image(rec_seg[i], name=str(i) + '_rec_seg', path=save_dir)
                save_pred2image(disrupted_seg[i], name=str(i) + '_disrupt_seg', path=save_dir)
                if i > 5:
                    break

        val_loss_list = list()
        val_disc_dice_list = list()
        val_cup_dice_list = list()
        with torch.no_grad():
            Norm_model.eval()
            Seg_model.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    normed_data = Norm_model(data)
                    output = Seg_model(normed_data)
                    loss = seg_criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + seg_criterion(output[:, 1],
                                                                                      (seg[:, 0] == 2) * 1.0)
                val_loss_list.append(loss.detach().cpu().numpy())
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
        mean_val_loss = np.mean(val_loss_list)
        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)
        writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
        writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
        writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
        writer.add_image('Val/Output Disc', make_grid(output_sigmoid[:10, 0][:, np.newaxis], 10, normalize=True), epoch)
        writer.add_image('Val/Output Cup', make_grid(output_sigmoid[:10, 1][:, np.newaxis], 10, normalize=True), epoch)
        writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)

        print('  Val loss: {}\n'
              '  Val disc dice: {}; Cup dice: {}'.format(mean_val_loss, mean_val_disc_dice, mean_val_cup_dice))

        avg_dice = (mean_val_disc_dice + mean_val_cup_dice) / 2
        if best_metric < avg_dice:
            best_metric = avg_dice
            saved_model = {
                'epoch': epoch + 1,
                'seg_model_state_dict': Seg_model.state_dict(),
                'norm_model_state_dict': Norm_model.state_dict(),
                'dae_state_dict': DAE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dae_optimizer_state_dict': dae_optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('best'))
            torch.save(saved_model, join(model_folder, 'model_best.model'))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)

    saved_model = {
        'epoch': epoch + 1,
        'seg_model_state_dict': Seg_model.state_dict(),
        'norm_model_state_dict': Norm_model.state_dict(),
        'dae_state_dict': DAE.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dae_optimizer_state_dict': dae_optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))


  # inference
    from inference.inference_nets.inference_unet import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference(args, 'model_best.model', root_folder, log_folder, [ts_csv_path], inference_tag)



def train_Prosatate(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = 20
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle

    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)
    logger = get_logger(log_folder)


    tr_img_list, tr_label_list = convert_labeled_list(args.tr_csv, r=-1)
    tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'val2d', tuple(args.patch_size), img_normalize=True)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=args.batch_size,
                                                num_workers= args.num_threads,
                                                shuffle= True,
                                                pin_memory=True)
        
    ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=-1)
    ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)

    Norm_model = Norm_Indentity_Net()
    Seg_model = UNet_v2(num_classes=args.num_classes)
    DAE = U_Net_for_DAE(img_ch=args.num_classes, output_ch=args.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Norm_model = nn.DataParallel(Norm_model, device_ids=gpu).to(device)
    Seg_model = nn.DataParallel(Seg_model, device_ids=gpu).to(device)
    DAE = nn.DataParallel(DAE, device_ids=gpu).to(device)

    optimizer = torch.optim.SGD(
            list(Norm_model.parameters()) + list(Seg_model.parameters()), lr=initial_lr,
        momentum=0.99, nesterov=True)

    dae_optimizer = torch.optim.SGD(
        DAE.parameters(),
        lr=initial_lr,  momentum=0.99, nesterov=True)


    start_epoch = 0
    if continue_training:
        try:
            params = torch.load(join(model_folder, 'model_latest.model'))
        except FileNotFoundError:
            assert isfile(join(model_folder, 'model_final.model')), 'Missing model checkpoint!'
            params = torch.load(join(model_folder, 'model_final.model'))

        Norm_model.load_state_dict(params['norm_model_state_dict'])
        Seg_model.load_state_dict(params['seg_model_state_dict'])
        DAE.load_state_dict(params['dae_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        dae_optimizer.load_state_dict(params['dae_optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler1 = GradScaler()
    amp_grad_scaler2 = GradScaler()
    rec_criterion = nn.MSELoss(reduction='mean')
    seg_criterion = BCEDiceLoss().to(device) #BCEDiceLoss().to(device) # nn.BCEWithLogitsLoss()#

    start = time()
    best_metric = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        Norm_model.train()
        Seg_model.train()
        DAE.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_seg_loss_list = list()
        train_dae_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            data, seg = batch
            data = data.to(device)
            seg = seg.to(device)
            optimizer.zero_grad()
            with autocast():
                norm_data = Norm_model(data)
                output = Seg_model(norm_data)
                loss = seg_criterion(output, seg) # seg_criterion(output, seg, sigmoid=True)
            amp_grad_scaler1.scale(loss).backward()
            amp_grad_scaler1.unscale_(optimizer)
            amp_grad_scaler1.step(optimizer)
            amp_grad_scaler1.update()
            train_seg_loss_list.append(loss.detach().cpu().numpy())

            dae_optimizer.zero_grad()
            with autocast():
                output = torch.sigmoid(output).detach()
                disrupted_output = torch.zeros_like(output)
                for i in range(output.shape[0]):
                    disrupted_output[i] = mask_random_replace(output[i],cnt=20)
                rec_output = DAE(disrupted_output)
                rec_output = torch.clamp(rec_output, 0., 1.)
                # print(disrupted_output.size(), rec_output.size(), output.size())
                # print(rec_output.min(), rec_output.max())
                # mse_loss = rec_criterion(rec_output, output)


                one_hot_seg = (seg == 1 ) * 1.0
                disrupted_seg = torch.zeros_like(one_hot_seg)
                for i in range(one_hot_seg.shape[0]):
                    disrupted_seg[i] = mask_random_replace(one_hot_seg[i], cnt=20)
                rec_seg = DAE(disrupted_seg)

                mse_loss = (rec_criterion(rec_output, output) + rec_criterion(rec_seg, one_hot_seg)) / 2

            amp_grad_scaler2.scale(mse_loss).backward()
            amp_grad_scaler2.unscale_(dae_optimizer)
            amp_grad_scaler2.step(dae_optimizer)
            amp_grad_scaler2.update()
            train_dae_loss_list.append(mse_loss.detach().cpu().numpy())

        mean_tr_seg_loss = np.mean(train_seg_loss_list)
        mean_tr_dae_loss = np.mean(train_dae_loss_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Seg Loss", mean_tr_seg_loss, epoch)
        writer.add_scalar("Train Scalars/Train DAE Loss", mean_tr_dae_loss, epoch)
        logger.info('  Tr Seg loss: {}\n'.format(mean_tr_seg_loss))
        logger.info('  Tr DAE loss: {}\n'.format(mean_tr_dae_loss))

    #     if epoch % save_interval == 0:
    #         saved_model = {
    #             'epoch': epoch + 1,
    #             'seg_model_state_dict': Seg_model.state_dict(),
    #             'norm_model_state_dict': Norm_model.state_dict(),
    #             'dae_state_dict': DAE.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'dae_optimizer_state_dict': dae_optimizer.state_dict()
    #         }
    #         print('  Saving model_{}.model...'.format('latest'))
    #         torch.save(saved_model, join(model_folder, 'model_latest.model'))
    #         save_dir = visualization_folder+'/epoch-'+str(epoch)
    #         if not os.path.exists(save_dir):
    #             os.mkdir(save_dir)
    #         for i in range(output.shape[0]):
    #             save_pred2image_onelabel(output[i], name=str(i)+'_pred', path=save_dir)
    #             save_pred2image_onelabel(rec_output[i], name=str(i) + '_rec_pred', path=save_dir)
    #             save_pred2image_onelabel(disrupted_output[i], name=str(i) + '_disrupt_pred', path=save_dir)
    #             save_pred2image_onelabel(one_hot_seg[i], name=str(i) + '_seg', path=save_dir)
    #             save_pred2image_onelabel(rec_seg[i], name=str(i) + '_rec_seg', path=save_dir)
    #             save_pred2image_onelabel(disrupted_seg[i], name=str(i) + '_disrupt_seg', path=save_dir)
    #             if i > 5:
    #                 break

    #     val_loss_list = list()
    #     val_2d_dice_list = list()

    #     with torch.no_grad():
    #         Norm_model.eval()
    #         Seg_model.eval()
    #         for iter, batch in enumerate(ts_dataloader):
    #             case_data, case_seg, case_name = batch
    #             case_data = case_data[0]
    #             case_seg = case_seg[0]
    #             for s in range(case_data.shape[0]//batch_size):
    #                 data = case_data[s*batch_size:(s+1)*batch_size]
    #                 seg = case_seg[s*batch_size:(s+1)*batch_size].unsqueeze(1)
    #                 data = data.to(device)
    #                 seg = seg.to(device)
    #                 data = data.unsqueeze(1).repeat(1, 3, 1, 1)
    #                 with autocast():
    #                     norm_data = Norm_model(data)
    #                     output = Seg_model(norm_data)
    #                     loss = seg_criterion(output, seg)#seg_criterion(output, seg, sigmoid=True)
    #                 val_loss_list.append(loss.detach().cpu().numpy())
    #                 output_sigmoid = torch.sigmoid(output)
    #                 val_2d_dice_list.append(get_hard_dice(output_sigmoid.cpu(), (seg == 1).cpu() * 1.0))

    #     mean_val_loss = np.mean(val_loss_list)
    #     mean_val_2d_dice = np.mean(val_2d_dice_list)
     
    #     writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
    #     writer.add_scalar("Val Scalars/2D Dice", mean_val_2d_dice, epoch)
    #     writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
    #     writer.add_image('Val/Output', make_grid(output_sigmoid[:10], 10, normalize=True), epoch)
    #     writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)

    #     logger.info('  Val loss: {}\n'
    #           '  Val 2D dice: {}'.format(mean_val_loss, mean_val_2d_dice))

    #     avg_dice = mean_val_2d_dice
    #     if best_metric < avg_dice:
    #         best_metric = avg_dice
    #         saved_model = {
    #             'epoch': epoch + 1,
    #             'seg_model_state_dict': Seg_model.state_dict(),
    #             'norm_model_state_dict': Norm_model.state_dict(),
    #             'dae_state_dict': DAE.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'dae_optimizer_state_dict': dae_optimizer.state_dict()
    #         }
    #         logger.info('  Saving model_{}.model...'.format('best'))
    #         torch.save(saved_model, join(model_folder, 'model_best.model'))

    #     time_per_epoch = time() - start_epoch
    #     logger.info('  Durations: {}'.format(time_per_epoch))
    #     writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
   
    # saved_model = {
    #     'epoch': epoch + 1,
    #     'seg_model_state_dict': Seg_model.state_dict(),
    #     'norm_model_state_dict': Norm_model.state_dict(),
    #     'dae_state_dict': DAE.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'dae_optimizer_state_dict': dae_optimizer.state_dict()
    # }
    # print('Saving model_{}.model...'.format('final'))
    # torch.save(saved_model, join(model_folder, 'model_final.model'))
    # if isfile(join(model_folder, 'model_latest.model')):
    #     os.remove(join(model_folder, 'model_latest.model'))
    # total_time = time() - start
    # print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))


  # inference
    from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
    inference_tag = split_path(ts_csv[0])[-1].replace('.csv', '')
    print("Running inference: {}".format(inference_tag))
    dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg('DAE', [Norm_model, Seg_model, DAE], ts_dataloader, logger, device, visualization_folder, metrics_folder, num_classes=1, test_batch=batch_size, save_pre=False)
    print('Dice: {}, HD: {}, ASSD: {}'.format(dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes))