# -*- coding:utf-8 -*-
"""
On-the-Fly Test-time Adaptation for Medical Image Segmentation:
Step1. Training an auto-encoder as the Domain Prior Generator.
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
from datasets.dataloaders.RIGA_dataloader import RIGA_unlabeled_set
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_unlabeled_list, convert_labeled_list
from datasets.utils.transform import source_collate_fn_tr, collate_fn_tr, ae_collate_fn_tr, ae_collate_fn_tr_prostate
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from models.dpg_tta.arch import priorunet

from torchvision.transforms import transforms
import numpy as np
from PIL import Image
from utils.tools import save_tensor2image


def train(args):
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
    shuffle = not args.no_shuffle

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)

    # print('stage1', tr_label_list)
    if 'RIGA' in root_folder:
        tr_img_list, tr_label_list = convert_unlabeled_list(tr_csv, r=1)
        tr_dataset = RIGA_unlabeled_set(root_folder, tr_img_list, patch_size)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_threads,
                                                    shuffle=shuffle,
                                                    pin_memory=True,
                                                    collate_fn=ae_collate_fn_tr)
    elif 'prostate' in root_folder:
        tr_img_list, tr_label_list = convert_labeled_list(args.tr_csv, r=-1)
        tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'val2d', tuple(args.patch_size), img_normalize=True)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers= args.num_threads,
                                                    shuffle= True,
                                                    pin_memory=True,
                                                    collate_fn=ae_collate_fn_tr_prostate)
    else:
        raise NotImplementedError

    model = priorunet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    criterion = nn.MSELoss(reduction='mean')

    start = time()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        for iter, batch in enumerate(tr_dataloader):
            
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            gt = torch.from_numpy(batch['org_data']).cuda().to(dtype=torch.float32)
            
            optimizer.zero_grad()
            with autocast():
                output = model(data, get_prior=False)
                loss = criterion(output, gt)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            train_loss_list.append(loss.detach().cpu().numpy())

        mean_tr_loss = np.mean(train_loss_list)
        writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        print('  Tr loss: {}\n'.format(mean_tr_loss))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
            save_dir = visualization_folder+'/epoch-'+str(epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for i in range(output.shape[0]):
                save_tensor2image(output[i], name=str(i)+'_rec', path=save_dir)
                save_tensor2image(gt[i], name=str(i) + '_gt', path=save_dir)
                save_tensor2image(data[i], name=str(i) + '_org', path=save_dir)
                if i > 5:
                    break

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



