# -*- coding:utf-8 -*-
"""
On-the-Fly Test-time Adaptation for Medical Image Segmentation:
Step2. Based on the pretrained DPG, training the source adaptive unet.
"""

from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from utils.init import init_random_and_cudnn, Recorder
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet, Adaptive_UNet
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import source_collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from models.dpg_tta.arch import priorunet
from utils.init import get_logger
from inference.inference_nets_3d.inference_prostate import  test3d_single_label_seg


def train_RIGA(args):
    model_name = args.model
    gpus = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)
    device = init_random_and_cudnn (gpus, manualseed=args.manualseed, bentchmark=True)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=source_collate_fn_tr)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = Adaptive_UNet()
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    start_epoch = 0


    priormodel = priorunet()
    priormodel.load_state_dict(torch.load(args.dpg_path)['model_state_dict'])
    priormodel.to(device)
    priormodel.eval()
    for param in priormodel.parameters():
        param.requires_grad = False


    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    start = time()
    best_metric = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        for iter, batch in enumerate(tr_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            optimizer.zero_grad()
            with autocast():
                prior = priormodel(data, True)
                output = model(data, prior)
                loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2)*1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        print('  Tr loss: {}\n'
              '  Tr disc dice: {}; Cup dice: {}'.format(mean_tr_loss, mean_tr_disc_dice, mean_tr_cup_dice))

        val_loss_list = list()
        val_disc_dice_list = list()
        val_cup_dice_list = list()
        with torch.no_grad():
            model.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    prior = priormodel(data, True)
                    output = model(data, prior)
                    loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('best'))
            torch.save(saved_model, join(model_folder, 'model_best.model'))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch+1) % 100 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
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
        inference(args, 'model_final.model', root_folder, log_folder, [ts_csv_path], inference_tag)


def train_prostate(args):
    model_name = args.model
    gpus = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle
    logger = get_logger(log_folder)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)
    device = init_random_and_cudnn(gpus, manualseed=args.manualseed, bentchmark=True)

    
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
    model = Adaptive_UNet(num_classes=args.num_classes)
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    start_epoch = 0


    priormodel = priorunet()
    priormodel.load_state_dict(torch.load(args.dpg_path)['model_state_dict'])
    priormodel.to(device)
    priormodel.eval()
    for param in priormodel.parameters():
        param.requires_grad = False


    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    start = time()
    best_metric = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_2d_dice_list = list()

        for iter, batch in enumerate(tr_dataloader):
            data, seg = batch
            data = data.to(device)
            seg = seg.to(device)
            optimizer.zero_grad()
            with autocast():
                prior = priormodel(data, True)
                output = model(data, prior)
                loss = criterion(output, seg)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output)
            train_2d_dice_list.append(get_hard_dice(output_sigmoid.cpu(), seg.cpu()))
           
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_2d_dice = np.mean(train_2d_dice_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Train 2D Dice", mean_tr_2d_dice, epoch)
        print('  Tr loss: {}\n'
              '  Tr 2d dice: {}'.format(mean_tr_loss, mean_tr_2d_dice))

        if epoch % 5 == 0:
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg('DPG', [priormodel, model], ts_dataloader, logger, device, visualization_folder, metrics_folder, num_classes=1, test_batch=batch_size, save_pre=False)
            print('Dice: {}, HD: {}, ASSD: {}'.format(dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes))
        

        avg_dice = dice_avg_all_classes
        if best_metric < avg_dice:
            best_metric = avg_dice
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('best'))
            torch.save(saved_model, join(model_folder, 'model_best.model'))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch+1) % 100 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch+1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch+1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)
    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    # inference
    dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg('DPG', [priormodel, model], ts_dataloader, logger, device, visualization_folder, metrics_folder, num_classes=1, test_batch=batch_size, save_pre=False)
    print('Dice: {}, HD: {}, ASSD: {}'.format(dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes))
