# -*- coding:utf-8 -*-
"""
Test-time adaptable neural networks for robust medical image segmentation
Step1. Source joint training: Normalization module + Seg-Unet + DAE
According to the paperm, a 2D segmentation CNN is integrated with a 3D DAE.
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
from models.unet import UNet_v2, Norm_Indentity_Net, UNet3D_for_DAE
from torchvision.transforms import transforms
from utils.losses.seg_loss import DiceLoss, CELoss, Dice_CE_Loss, BCEDiceLoss
from utils.init import get_logger
import numpy as np
from PIL import Image

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

def mask_random_replace(x, cnt=5):
    x_shape = x.shape
    if len(x_shape) == 3:
        ### x: [C, H, W]
        _, img_rows, img_cols = x_shape
        x_trans = x.clone()
        cnt = 5
        while cnt > 0:
            block_noise_size_x = random.randint(img_rows // 32, img_rows // 16)
            block_noise_size_y = random.randint(img_cols // 32, img_cols // 16)

            noise_x1 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y1 = random.randint(3, img_cols - block_noise_size_y - 3)

            noise_x2 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y2 = random.randint(3, img_cols - block_noise_size_y - 3)
            
            x_trans[:,noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y] \
                = x[:, noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y]
           #  x_trans[
           # :,
           #  noise_y2:noise_y2 + block_noise_size_y, noise_z2] = x[
           #                                           :,
           #                                            noise_y1:noise_y1 + block_noise_size_y, noise_z1]
            cnt -= 1


    else:
        ### x: [C, D, H, W]
        _, img_deps, img_rows, img_cols = x_shape
        cnt = 10
        x_trans = x.clone()
        while cnt > 0:
            block_noise_size_x = random.randint(img_rows // 32, img_rows // 16)
            block_noise_size_y = random.randint(img_cols // 32, img_cols // 16)
            block_noise_size_z = random.randint(0, img_deps // 8)

            noise_x1 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y1 = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z1 = random.randint(0, img_deps-block_noise_size_z-3)

            noise_x2 = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y2 = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z2 = random.randint(0, img_deps-block_noise_size_z-3)

            x_trans[:, noise_z1:noise_z1+block_noise_size_z, noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y] = \
            x[:, noise_z2:noise_z2+block_noise_size_z, noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y]
        
            x_trans[:, noise_z2:noise_z2+block_noise_size_z, noise_x2:noise_x2+block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y] = \
                x[:, noise_z1:noise_z1+block_noise_size_z, noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y]
            
            cnt -= 1

    return x_trans

def disturb_mask(input_mask, n1=50, n2=20):
    D, H, W = input_mask.shape
    disturbed_mask = input_mask.clone()
    
    num_patches = np.random.randint(1, n1 + 1)
    
    for _ in range(num_patches):

        patch_size_d = np.random.randint(0, D//8)
        patch_size_h = np.random.randint(0, n2 + 1)
        patch_size_w = np.random.randint(0, n2 + 1)
        
    
        crop_d_1 = np.random.randint(0, D - patch_size_d + 1)
        crop_h_1 = np.random.randint(0, H - patch_size_h + 1)
        crop_w_1 = np.random.randint(0, W - patch_size_w + 1)

      
        patch_1 = disturbed_mask[crop_d_1:crop_d_1 + patch_size_d, crop_h_1:crop_h_1 + patch_size_h, crop_w_1:crop_w_1 + patch_size_w]
        
       
        crop_d_2 = np.random.randint(0, D - patch_size_d + 1)
        crop_h_2 = np.random.randint(0, H - patch_size_h + 1)
        crop_w_2 = np.random.randint(0, W - patch_size_w + 1)
        
        while crop_d_2 == crop_d_1 and crop_h_2 == crop_h_1 and crop_w_2 == crop_w_1:
            crop_d_2 = np.random.randint(0, D - patch_size_d + 1)
            crop_h_2 = np.random.randint(0, H - patch_size_h + 1)
            crop_w_2 = np.random.randint(0, W - patch_size_w + 1)

        patch_2 = disturbed_mask[crop_d_2:crop_d_2 + patch_size_d, crop_h_2:crop_h_2 + patch_size_h, crop_w_2:crop_w_2 + patch_size_w]

        # swap two patches
        disturbed_mask[crop_d_1:crop_d_1 + patch_size_d, crop_h_1:crop_h_1 + patch_size_h, crop_w_1:crop_w_1 + patch_size_w] = patch_2
        disturbed_mask[crop_d_2:crop_d_2 + patch_size_d, crop_h_2:crop_h_2 + patch_size_h, crop_w_2:crop_w_2 + patch_size_w] = patch_1
    
    return disturbed_mask



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


    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=-1)
    tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'test3d', args.patch_size, img_normalize=True)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=1,
                                                num_workers= args.num_threads,
                                                shuffle= True,
                                                pin_memory=True)
        
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=-1)
    ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)

    Norm_model = Norm_Indentity_Net()
    Seg_model = UNet_v2(num_classes=args.num_classes)
    DAE = UNet3D_for_DAE(in_channels=args.num_classes, n_class=args.num_classes, normalization='sigmoid')

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
    seg_criterion = BCEDiceLoss().to(device) # nn.BCEWithLogitsLoss()#

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
        for iter, case_data in enumerate(tr_dataloader):
            data, seg, case_name = case_data
            data = data[0].to(device)
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            seg = seg.to(device).squeeze()
            num_slice = data.shape[0]
            s = 0
            output = torch.zeros_like(seg)
            while s < num_slice:
                cur_batch = min(batch_size, num_slice-s)
                batch_data = data[s:s+cur_batch]
                batch_seg = seg[s:s+cur_batch].unsqueeze(1)
                optimizer.zero_grad()
                with autocast():
                    norm_data = Norm_model(batch_data)
                    batch_pred = Seg_model(norm_data)
                    loss = seg_criterion(batch_pred, batch_seg, sigmoid=True) #seg_criterion(batch_pred, batch_seg) # 
                    output[s:s+cur_batch] = batch_pred.squeeze().detach()
                amp_grad_scaler1.scale(loss).backward()
                amp_grad_scaler1.unscale_(optimizer)
                amp_grad_scaler1.step(optimizer)
                amp_grad_scaler1.update()
                train_seg_loss_list.append(loss.detach().cpu().numpy())
                s += batch_size

            dae_optimizer.zero_grad()
            with autocast():
                # ouput [d, h, w]
                output = torch.cat((output, torch.zeros((16 - output.shape[0] % 16, output.shape[-2], output.shape[-1])).to(device)), dim=0)
                seg = torch.cat((seg, torch.zeros((16 - seg.shape[0] % 16, seg.shape[-2], seg.shape[-1])).to(device)), dim=0)

                output = torch.sigmoid(output).unsqueeze(0)
                 # ouput [1, d, h, w]
                disrupted_output = mask_random_replace(output)

                rec_output = DAE(disrupted_output.unsqueeze(0))
                rec_output = torch.clamp(rec_output, 0., 1.).squeeze(0)
                # print(disrupted_output.size(), rec_output.size(), output.size())
                # print(rec_output.min(), rec_output.max())
                # mse_loss = rec_criterion(rec_output, output)

                one_hot_seg = (seg == 1) * 1.0
                one_hot_seg = one_hot_seg.unsqueeze(0)
                disrupted_seg = mask_random_replace(one_hot_seg)
                rec_seg = DAE(disrupted_seg.unsqueeze(0))
                rec_seg = torch.clamp(rec_seg, 0., 1.).squeeze(0)

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


  # inference
    from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
    inference_tag = split_path(ts_csv[0])[-1].replace('.csv', '')
    print("Running inference: {}".format(inference_tag))
    dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg('DAE', [Norm_model, Seg_model], ts_dataloader, logger, device, visualization_folder, metrics_folder, num_classes=1, test_batch=batch_size, save_pre=False)
    print('Dice: {}, HD: {}, ASSD: {}'.format(dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes))



def train_Prosatate_v2(args):
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


    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=-1)
    tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'test3d', args.patch_size, img_normalize=True)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=1,
                                                num_workers= args.num_threads,
                                                shuffle= True,
                                                pin_memory=True)
        
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=-1)
    ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)

    Norm_model = Norm_Indentity_Net()
    Seg_model = UNet_v2(num_classes=args.num_classes)
    DAE = UNet3D_for_DAE(in_channels=args.num_classes, n_class=args.num_classes, normalization='sigmoid')

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
    seg_criterion = BCEDiceLoss().to(device) # nn.BCEWithLogitsLoss()#

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
        for iter, case_data in enumerate(tr_dataloader):
            data, seg, case_name = case_data
            data = data[0].to(device)
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            seg = seg.to(device).squeeze()
            num_slice = data.shape[0]
            s = 0
            output = torch.zeros_like(seg)
            while s < num_slice:
                cur_batch = min(batch_size, num_slice-s)
                batch_data = data[s:s+cur_batch]
                batch_seg = seg[s:s+cur_batch].unsqueeze(1)
                optimizer.zero_grad()
                with autocast():
                    norm_data = Norm_model(batch_data)
                    batch_pred = Seg_model(norm_data)
                    loss = seg_criterion(batch_pred, batch_seg, sigmoid=True) #seg_criterion(batch_pred, batch_seg) # 
                    output[s:s+cur_batch] = batch_pred.squeeze().detach()
                amp_grad_scaler1.scale(loss).backward()
                amp_grad_scaler1.unscale_(optimizer)
                amp_grad_scaler1.step(optimizer)
                amp_grad_scaler1.update()
                train_seg_loss_list.append(loss.detach().cpu().numpy())
                s += batch_size

            dae_optimizer.zero_grad()
            with autocast():
                # ouput [d, h, w]
                #将output和seg不足16整数倍的部分补0
                output = torch.cat((output, torch.zeros((16 - output.shape[0] % 16, output.shape[-2], output.shape[-1])).to(device)), dim=0)
                seg = torch.cat((seg, torch.zeros((16 - seg.shape[0] % 16, seg.shape[-2], seg.shape[-1])).to(device)), dim=0)

                output = torch.sigmoid(output)
                 # ouput [1, d, h, w]
                disrupted_output = disturb_mask(output, n1=50, n2=20)

                rec_output = DAE(disrupted_output.unsqueeze(0).unsqueeze(0))
                rec_output = torch.clamp(rec_output, 0., 1.).squeeze()
                # print(disrupted_output.size(), rec_output.size(), output.size())
                # print(rec_output.min(), rec_output.max())
                # mse_loss = rec_criterion(rec_output, output)

                one_hot_seg = (seg == 1) * 1.0
                disrupted_seg = disturb_mask(one_hot_seg, n1=50, n2=20)
                rec_seg = DAE(disrupted_seg.unsqueeze(0).unsqueeze(0))
                rec_seg = torch.clamp(rec_seg, 0., 1.).squeeze()

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

  # inference
    from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
    inference_tag = split_path(ts_csv[0])[-1].replace('.csv', '')
    print("Running inference: {}".format(inference_tag))
    dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg('DAE', [Norm_model, Seg_model, DAE], ts_dataloader, logger, device, visualization_folder, metrics_folder, num_classes=1, test_batch=batch_size, save_pre=False)
    print('Dice: {}, HD: {}, ASSD: {}'.format(dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes))