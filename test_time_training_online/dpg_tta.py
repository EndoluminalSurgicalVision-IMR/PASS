"""
On-the-Fly Test-time Adaptation for Medical Image Segmentation
"""
import sys
sys.path.append('/mnt/data/chuyan/Medical_TTA')
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import Adaptive_UNet
from models.dpg_tta.arch import priorunet

class DPG_Adapter(BaseAdapter):
    def __init__(self, args):
        super(DPG_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)

    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        assert isfile(self.args.dpg_path), 'missing DPG checkpoint!'

        self.model = Adaptive_UNet(num_classes=self.args.num_classes)
        self.model = nn.DataParallel(self.model, device_ids=self.gpus).to(self.device)
        self.model.load_state_dict(torch.load(self.args.pretrained_model)['model_state_dict'])

        self.priormodel = priorunet()
        self.priormodel.load_state_dict(torch.load(self.args.dpg_path)['model_state_dict'])
        self.priormodel.to(self.device)
        self.priormodel.eval()
        for param in self.priormodel.parameters():
            param.requires_grad = False

    def evaluate(self):
        # No need to train.
        if self.tag in ['Base1_test', 'Base2_test', 'Base3_test', 'MESSIDOR', 'BinRushed']:
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                print("Running inference: {}".format(inference_tag))
                inference(self.args, [self.priormodel, self.model], self.device, self.log_folder, [ts_csv_path],
                          inference_tag)
        elif 'Prostate' in self.tag:
            from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, [self.priormodel, self.model], self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)
            
        else:
            raise NotImplemented



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="DPG-TTA", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--arch', default="adaptive_unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=2, required=False,
                        help='Class number.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Base1_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base1_test.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--pretrained_model', default='log_dir/DPG-Ada-UNet_{Source}/checkpoints/model_final.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--dpg_path', required=False,default='log_dir/DPG_{Source}/checkpoints/model_final.model',
                        help='pretrained DPG model path.')

    args = parser.parse_args()
    
    # for prostate
    args.num_classes = 1
    args.batch_size = 16
    args.pacth_size = [384, 384]
    args.root = 'data/MRI_prostate/'
    args.tag ='Prostate_RUNMC2I2CVB'
    args.ts_csv = ['data/MRI_prostate/I2CVB_all.csv']
    args.pretrained_model = 'log_dir/DPG-Ada-UNet_Prostate_RUNMC/checkpoints/model_final.model'
    args.dpg_path = 'log_dir/DPG_Prostate_RUNMC/checkpoints/model_final.model'
    adaptor = DPG_Adapter(args)

    adaptor.evaluate()
