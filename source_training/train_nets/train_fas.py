# -*- coding:utf-8 -*-
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
from models.PromptTTA.pls_fas import UNet_FAS
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import source_collate_fn_tr_fda, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from models.PromptTTA.pls_fas import mix_data_prompt
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F


def train(args):
    model_name = args.model
    gpu = tuple(args.gpu)
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
    pretrained_model_path = args.pretrained_model
    prompt_model_path = args.prompt_model_path
    gamma = args.gamma

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=-1)
    tr_dataset = RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, patch_size, img_normalize=False)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=source_collate_fn_tr_fda)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_threads//2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)

    model = UNet_FAS()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt_model_params = torch.load(prompt_model_path)
    data_prompt = prompt_model_params['model_state_dict']['data_prompt'].cuda().to(dtype=torch.float32)

    assert isfile(pretrained_model_path), 'missing pretrained model!'
    pretrained_params = torch.load(pretrained_model_path)
    model.load_state_dict(pretrained_params['model_state_dict'])

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)

    start_epoch = 0
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
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_com_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        for iter, batch in enumerate(tr_dataloader):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(normalize_image(batch['fda_data'])).cuda().to(dtype=torch.float32)
            optimizer.zero_grad()
            with autocast():
                output_data, global_f_data = model(mix_data_prompt(data, data_prompt), gfeat=True)
                output_fda_data, global_f_fda_data = model(mix_data_prompt(fda_data, data_prompt), gfeat=True)
                compact_loss = F.l1_loss(global_f_fda_data, global_f_data.detach(), reduction='mean') + \
                               F.l1_loss(global_f_data, global_f_fda_data.detach(), reduction='mean')
                seg_loss = criterion(output_data[:, 0], (seg[:, 0] > 0) * 1.0) + \
                           criterion(output_data[:, 1], (seg[:, 0] == 2) * 1.0) + \
                           criterion(output_fda_data[:, 0], (seg[:, 0] > 0) * 1.0) + \
                           criterion(output_fda_data[:, 1], (seg[:, 0] == 2) * 1.0)
                loss = seg_loss + gamma * compact_loss
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_com_loss_list.append(compact_loss.detach().cpu().numpy())
            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output_data)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        mean_com_loss = np.mean(train_com_loss_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        writer.add_scalar("Train Scalars/Train Com Loss", mean_com_loss, epoch)
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
                    output = model(mix_data_prompt(data, data_prompt))
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

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch+1) % 200 == 0:
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
    from inference.inference_nets.inference_fas import inference
    disc_dice_lists = list()
    cup_dice_lists = list()
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        disc_dice_list, cup_dice_list = inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], prompt_model_path, inference_tag)
        disc_dice_lists += disc_dice_list
        cup_dice_lists += cup_dice_list
    return disc_dice_lists, cup_dice_lists