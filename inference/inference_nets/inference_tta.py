# -*- coding:utf-8 -*-
"""
Especially for test-time training.
"""
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils import data
from utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import collate_fn_ts
from utils.metrics.dice import get_hard_dice
from utils.visualization import visualization_as_nii
from test_time_training.pls_fas_tta import pseudo_label_refinement
import models.TENT.dua as dua

def inference(args, model, device, log_folder, ts_csv, inference_tag='all', logger=None):
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)
    print('******ts_csv*****', ts_csv)
    # device = torch.device('cuda:%d' % gpu[0]
    #                       if torch.cuda.is_available() and len(gpu) > 0 else 'cpu')
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    if ts_label_list is None:
        evaluate = False
        ts_dataset = RIGA_unlabeled_set(args.root, ts_img_list, args.patch_size)
    else:
        evaluate = True
        ts_dataset = RIGA_labeled_set(args.root, ts_img_list, ts_label_list, args.patch_size)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts,
                                                drop_last=False
                                                )

    seg_list = list()
    output_list = list()
    refined_output_list = list()
    data_list = list()
    name_list = list()

    if args.arch == 'adaptive_unet_2d':
        ### for unet 2d

        priormodel = model[0]
        segmodel = model[1]

        with torch.no_grad():
            priormodel.eval()
            segmodel.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).to(dtype=torch.float32)
                name = batch['name']
                data = data.to(device)
                prior = priormodel(data, True)
                output = segmodel(data, prior)
                output_sigmoid = torch.sigmoid(output).cpu().numpy()
                seg_list.append(batch['seg'])
                output_list.append(output_sigmoid)
                data_list.append(batch['data'])
                name_list.append(name)

    elif args.arch == 'norm_unet_2d':
        Norm_model = model[0]
        Seg_model = model[1]

        seg_list = list()
        output_list = list()
        data_list = list()
        name_list = list()
        with torch.no_grad():
            Norm_model.eval()
            Seg_model.eval()
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                name = batch['name']
                with autocast():
                    norm_data = Norm_model(data)
                    output = Seg_model(norm_data)
                output_sigmoid = torch.sigmoid(output).cpu().numpy()
                seg_list.append(batch['seg'])
                output_list.append(output_sigmoid)
                data_list.append(batch['data'])
                name_list.append(name)


    else:
        if args.model == 'TENT' or args.model == 'TIPI' or \
        args.model == 'PTBN' or args.model == 'SAR' or args.model == 'CoTTA':
            logger.info('**************Test-Time Inference with {}************'.format(args.model))
            for iter, batch in enumerate(ts_dataloader):
                data = torch.from_numpy(batch['data']).to(dtype=torch.float32)
                name = batch['name']
                data = data.to(device)
                output = model(data, reset=True)
                
                output_sigmoid = torch.sigmoid(output).detach().cpu().numpy()
                seg_list.append(batch['seg'])
                output_list.append(output_sigmoid)
                data_list.append(batch['data'])
                name_list.append(name)

        elif args.model == 'DUA':
            decay_factor = 0.94
            min_momentum_constant = 0.005
            mom_pre = 0.1
           
            for iter, batch in enumerate(ts_dataloader):
                model.eval()
                data = torch.from_numpy(batch['data']).to(dtype=torch.float32)
                name = batch['name']
                data = data.to(device)
                # print('min max', torch.min(data), torch.max(data))
                ################# DUA change momentum ###################
                mom_new = (mom_pre * decay_factor)
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train()
                        m.momentum = mom_new + min_momentum_constant
                mom_pre = mom_new
                ###################### AUG #################
                inputs = dua.get_adaption_inputs_seg(data, device, aug_view_num=64)
                _ = model(inputs)
                with torch.no_grad():
                    model.eval()
                    ###################### Inference #################
                    output = model(data)
                    refined_output = pseudo_label_refinement(data, output, idc=[0,1], uncertainty_threshold=0.3, alpha=0.6, beta=1.4)
                    output_sigmoid = torch.sigmoid(output).cpu().numpy()
                    seg_list.append(batch['seg'])
                    output_list.append(output_sigmoid)
                    refined_output_list.append(refined_output.cpu().numpy())
                    data_list.append(batch['data'])
                    name_list.append(name)

        else:
            with torch.no_grad():
                model.eval()
                for iter, batch in enumerate(ts_dataloader):
                    data = torch.from_numpy(batch['data']).to(dtype=torch.float32)
                    name = batch['name']
                    data = data.to(device)
                    output = model(data)
                    output_sigmoid = torch.sigmoid(output).cpu().numpy()
                    seg_list.append(batch['seg'])
                    output_list.append(output_sigmoid)
                    data_list.append(batch['data'])
                    name_list.append(name)

    all_data = list()
    all_seg = list()
    all_output = list()
    all_name = list()
 
    for i in range(len(data_list)):
        for j in range(data_list[i].shape[0]):
            all_data.append(data_list[i][j])
            all_seg.append(seg_list[i][j])
            all_output.append(output_list[i][j])
            all_name.append(name_list[i][j])
    all_data = np.stack(all_data)
    all_seg = np.stack(all_seg)
    all_output = np.stack(all_output)
   

    zero = np.zeros_like(all_output)
    one = np.ones_like(all_output)
    all_output_threshold = np.where(all_output>0.5, one, zero)
    visualization_as_nii(((all_data[:, 0]+all_data[:, 1]+all_data[:, 2])/3).astype(np.float32), join(visualization_folder, 'data_all.nii.gz'))
    visualization_as_nii((all_output_threshold[:, 0]+all_output_threshold[:, 1]).astype(np.float32), join(visualization_folder, 'output_all.nii.gz'))
  
    if evaluate:
        visualization_as_nii(all_seg[:, 0].astype(np.float32), join(visualization_folder, 'seg.nii.gz'))
        disc_dice, disc_std, disc_dice_list = get_hard_dice(torch.from_numpy(all_output[:, 0]), torch.from_numpy(((all_seg[:, 0] > 0) * 1.0)), std=True)
        cup_dice, cup_std, cup_dice_list = get_hard_dice(torch.from_numpy(all_output[:, 1]), torch.from_numpy(((all_seg[:, 0] > 1) * 1.0)), std=True)
        metrics_str = 'Tag: {}\n  Disc dice: {:.5f}+{:.5f}; Cup dice: {:.5f}+{:.5f}.'.format(inference_tag, disc_dice, disc_std, cup_dice, cup_std)
        logger.info(metrics_str)

        with open(join(metrics_folder, '{}.csv'.format(inference_tag)), 'w') as f:
            for dice_i in range(len(disc_dice_list)):
                f.write('{},{},{}\n'.format(all_name[dice_i], disc_dice_list[dice_i], cup_dice_list[dice_i]))
        return model
                       


