"""
Test-Time Adaptation with Shape Moments for Image Segmentation
https://link.springer.com/chapter/10.1007/978-3-031-16440-8_70
TENT-TTA, Moment-TTA, BNM-TTA(RN-CR)
"""
# /usr/bin/env python3.6
import sys
sys.path.append('/mnt/data/chuyan/Medical_TTA')
import argparse
from utils.init import init_random_and_cudnn, Recorder
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
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid

from models.TENT.tent import configure_model
import models.moment_tta.losses as moment_tta_losses
import models.moment_tta.utils as moment_tta_utils
from models.moment_tta.bounds import *
from typing import Any, Callable, List, Tuple
from test_time_training_offline.base_tta import BaseAdapter

#### Loss for adaptation

TENT = ['Weighted_self_entropy_loss',
         {'weights':[1, 10], 'idc':[0, 1], 'act':'sigmoid'}]

TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

RN_w_CR_Prostate = ['RN_w_CR_loss',
         {'idc':[0], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766], 'act':'sigmoid_onelabel'}]

LSIZECentroid = ['Constrain_prior_w_self_entropy_loss',
                 {'idc': [0, 1],
                  'weights_se':[0.8, 0.2],'lamb_se':1,
                  'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                  'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                  'mom_est':[[254.9747, 255.98499], [253.97656, 255.04263]],
                  'moment_fn':'soft_centroid', 'lamb_consprior':1,
                  'power': 1, 'act':'sigmoid'}]


LSIZEDistCentroid = ['Constrain_prior_w_self_entropy_loss',
                     {'idc':[0, 1],
                      'weights_se':[0.8, 0.2],'lamb_se':1,
                      'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                     'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                     'mom_est':[[39.157185, 37.18185 ],[17.430944,18.484102]],
                     'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,
                     'power': 1, 'act':'sigmoid'}]


class Moment_TTA(BaseAdapter):
    def __init__(self, args):
        super(Moment_TTA, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        if args.model == 'Moment-TTA':
            self.LOSS_Dict = LSIZE_Prostate
        elif args.model == 'Tent-TTA':
            self.LOSS_Dict = TENT_Prostate
        elif args.model == 'RN-CR-TTA':
            self.LOSS_Dict = RN_w_CR_Prostate
        else:
            self.LOSS_Dict = LSIZE_Prostate

    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':
          
            self.model = UNet(num_classes=self.args.num_classes)
            self.model.load_state_dict(params['model_state_dict'])
            self.model = self.model.to(self.device)
          
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info('normalization statistics updated')
            else:
                self.logger.info('WARNING all var updated')

        else:
            raise NotImplementedError

    def configure_model_2d(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return self.model

    def set_train_state(self):
        self.model.train()
      

    def evaluate(self):
        start_epoch = 0
        if args.continue_training:
            assert isfile(join(self.model_folder, 'model_final.model')), 'missing model checkpoint!'
            params = torch.load(join(self.model_folder, 'model_final.model'))
            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(start_epoch))

        loss_name, loss_params = self.LOSS_Dict
        self.logger.info('loss_name:{}'.format(loss_name))
        self.logger.info('Loss params:{}'.format(loss_params))

        loss_class = getattr(moment_tta_losses, loss_name)
        loss_fn = loss_class(**loss_params)

        start = time()
        for epoch in range(start_epoch, args.num_epochs):
            if self.args.dataset_name == 'RIGA':
                self.do_disc_cup_seg_one_epoch(epoch, loss_fn)
            elif self.args.dataset_name == 'Prostate':
                self.do_prostate_seg_one_epoch(epoch, loss_fn)
            else:
                raise NotImplemented

        saved_model = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print('Saving model_{}.model...'.format('final'))
        torch.save(saved_model, join(self.model_folder, 'model_final.model'))
        if isfile(join(self.model_folder, 'model_latest.model')):
            os.remove(join(self.model_folder, 'model_latest.model'))
        total_time = time() - start
        print("Running %d epochs took a total of %.2f seconds." % (args.num_epochs, total_time))

        # inference
        if self.args.dataset_name == 'RIGA':
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                self.recorder.logger.info("Running inference: {}".format(inference_tag))
                inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                          inference_tag)
                
        elif self.args.dataset_name == 'Prostate':
            from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)
        else:
            raise NotImplemented

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="RN-CR-TTA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base2_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='../../ProSFDA_master/prosfda/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['../../ProSFDA_master/prosfda/RIGAPlus/MESSIDOR_Base2_test.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=1, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='../log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')

    args = parser.parse_args()
    # evaluate(args)
    from scripts.tta_configs import Base1_config, Lung_Lobe_Convid_config, Lung_Lobe_Clean_config, Prostate_RUNMC2UCL_config, Prostate_MSD2BIDMC_config, Prostate_RUNMC2BMC_config
    args.__dict__.update(Prostate_RUNMC2UCL_config)
    adpater = Moment_TTA(args)
    adpater.evaluate()


