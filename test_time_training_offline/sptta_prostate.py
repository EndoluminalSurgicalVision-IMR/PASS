"""
Shape-Prompt-Learning: The proposed SP-TTA for MRI-Prostate dataset in offline setup.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
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
from models import *
import models.moment_tta.losses as moment_tta_losses
from models.moment_tta.bounds import *
from typing import Any, Callable, List, Tuple
from test_time_training_offline.base_tta import BaseAdapter
from loss_functions.bn_loss import bn_loss
from utils.losses.seg_loss import  BCEDiceLoss
from utils.tools import *
####
TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]


RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766], 'act':'sigmoid_onelabel'}]



class Offline_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Offline_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        if args.model == 'Moment-TTA':
            self.LOSS_Dict = LSIZE_Prostate
        elif args.model == 'Tent-TTA':
            self.LOSS_Dict = TENT_Prostate
        elif args.model == 'RN-CR-TTA':
            self.LOSS_Dict = RN_w_CR
        else:
            self.LOSS_Dict = LSIZE_Prostate

        self.criterion = BCEDiceLoss().to(self.device)

    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':  
            self.model = UNet(num_classes=self.args.num_classes)
            self.model.load_state_dict(self.pretrained_params['model_state_dict'])
            self.model = self.model.to(self.device)


        elif args.arch == 'unet_2d_sptta':
            self.model = UNet_SPTTA(pretrained_path=args.pretrained_model, num_classes=self.args.num_classes,  patch_size=self.args.patch_size)
            self.model = self.model.to(self.device)
            self.amp_grad_scaler = GradScaler()
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
                # if self.args.dataset_name == 'RIGA':
                # m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None


        for name, param in self.model.named_parameters():
            if name.find('prompt') != -1 or name.find('data') != -1:
                param.requires_grad = True
        return self.model
    
  
    def set_train_state(self):
        self.model.train()
     
        
    def optimize_parameters(self, input, loss_fn):
        self.optimizer.zero_grad()

        pred, _, _, _ = self.model(input, training=True) 
        loss = loss_fn(pred)

        loss.backward()
        self.optimizer.step()
        
        return pred, loss


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

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)

        start = time()
        for epoch in range(start_epoch, self.args.num_epochs):
            self.do_prostate_seg_one_epoch(epoch, loss_fn)
           
            saved_model = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
            print('Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(self.model_folder, 'model_latest.model'))
         
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

     
        from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
        dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)
      

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-TTA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta", required=False,
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
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='../log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    args = parser.parse_args()
    # evaluate(args)
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config,Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2BMC_config)
    args.tag = args.tag +'_sptta'
    adpater = Offline_Adapter(args)
    adpater.evaluate()


