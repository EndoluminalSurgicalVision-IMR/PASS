# -*- coding:utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch.utils import data
from utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from models.PromptTTA.pls_fas import UNet_PLS
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import collate_fn_ts
from utils.metrics.dice import get_hard_dice
from utils.visualization import visualization_as_nii


def inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, pretrained_model_path, args, inference_tag='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)

    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    if ts_label_list is None:
        evaluate = False
        ts_dataset = RIGA_unlabeled_set(root_folder, ts_img_list, patch_size)
    else:
        evaluate = True
        ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=4,
                                                num_workers=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)
    print('Load pretrained model from {}'.format(pretrained_model_path))
   
    model = UNet_PLS(pretrained_path=pretrained_model_path)
  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    assert isfile(join(model_folder, chk_name)), 'missing model checkpoint {}!'.format(join(model_folder, chk_name))
    print('Load model for inference:{}'.format(join(model_folder, chk_name)))
    params = torch.load(join(model_folder, chk_name))
    model.load_state_dict(params['model_state_dict'])

    seg_list = list()
    output_list = list()
    data_list = list()
    name_list = list()
    with torch.no_grad():
        model.eval()
        for iter, batch in enumerate(ts_dataloader):
            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            name = batch['name']
            with autocast():
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
        print(metrics_str)
        with open(join(metrics_folder, '{}.txt'.format(inference_tag)), 'w') as f:
            f.write(metrics_str)
        with open(join(metrics_folder, '{}.csv'.format(inference_tag)), 'w') as f:
            for dice_i in range(len(disc_dice_list)):
                f.write('{},{},{}\n'.format(all_name[dice_i], disc_dice_list[dice_i], cup_dice_list[dice_i]))
        return disc_dice_list, cup_dice_list
