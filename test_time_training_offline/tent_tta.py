"""
A set of TTA methods based on Tent ICLR 2021 Spotlight.
1. Norm-TTN
2. TNET 
Tent: Fully Test-Time Adaptation by Entropy Minimization
https://openreview.net/forum?id=uXl3bZLkr3c
3.TIPI
TIPI
https://github.com/atuannguyen/TIPI/blob/main/methods/tipi.py
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
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
from models.unet import UNet
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
import models.TENT.ttn as ttn
import argparse
from test_time_training_offline.base_tta import BaseAdapter
from models.TENT.tipi import TIPI
import models.TENT.tent as tent
import models.TENT.eata as eata
import models.TENT.cotta as cotta
import models.TENT.sar as sar
import math
from models import UNet_SODA_P2F_v8

def setup_source(model, logger):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model, args, logger):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = ttn.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = ttn.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_optimizer(params, args):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if args.optimizer == 'Adam':
        return torch.optim.Adam(params,
                    lr=args.initial_lr,
                    betas=(0.9, 0.999),
                    weight_decay= 0.)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(params,
                   lr=args.initial_lr,
                   momentum=0.9,
                   dampening=0.,
                   weight_decay=0.,
                   nesterov=True)
    else:
        raise NotImplementedError


def setup_tent(model, args, logger, act='softmax'):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params, args)
    tent_model = tent.Tent(model, optimizer,
                           steps=args.optim_steps,
                           episodic=args.model_episodic,
                           act=act)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_eata(model, args, logger, act='softmax'):
    """Set up EATA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = eata.configure_model(model)
    params, param_names = eata.collect_params(model)
    optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
    adapt_model = eata.EATA(model, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return adapt_model

def setup_cotta(model, cfg, logger, act='sigmoid'):
    """Set up CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params, args)
    if 'Base' in cfg.tag:
        # torchvision transforms
        aug_type = 'tv_trans'
    else:
        # batchgenerators transforms
        aug_type = 'bg_trans'
    cotta_model = cotta.CoTTA(model, optimizer, (cfg.patch_size[0], cfg.patch_size[1], 3),
                           steps=cfg.OPTIM_STEPS,
                           episodic=cfg.MODEL_EPISODIC, 
                           mt_alpha=cfg.OPTIM_MT, 
                           rst_m=cfg.OPTIM_RST, 
                           ap=cfg.OPTIM_AP,
                           act=act, 
                           task='seg',
                           aug_transform_type=aug_type)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def setup_sar(model, args, logger, act='sigmoid'):
    """Set up sar adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sar.SAM(params, base_optimizer, lr=args.initial_lr, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=args.sar_margin_e0, task='seg', act=act)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return adapt_model


class Tent_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Tent_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)

    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        params = torch.load(self.args.pretrained_model)
        if self.args.arch == 'unet_2d':
           
            base_model = UNet(num_classes=self.args.num_classes)
            base_model.load_state_dict(params['model_state_dict'])
            base_model = base_model.to(self.device)
          

        elif args.arch == 'unet_2d_sptta':
            base_model = UNet_SPTTA(pretrained_path=args.pretrained_model, num_classes=self.args.num_classes,  patch_size=self.args.patch_size)
            base_model = base_model.to(self.device)
        else:
            raise NotImplementedError

        if self.args.model == "Source":
            self.logger.info("test-time adaptation: NONE")
            self.model = setup_source(base_model, self.logger)

        elif self.args.model == "PTBN":
            self.logger.info("test-time adaptation: NORM")
            self.model = setup_norm(base_model, self.args, self.logger)
            # model = base_model

        elif self.args.model == "TENT":
            self.logger.info("test-time adaptation: TENT")
            self.model = setup_tent(base_model, self.args, self.logger, act='sigmoid')

        elif self.args.model == "TIPI":
             self.logger.info("test-time adaptation: TIPI")
             self.model = TIPI(base_model, lr_per_sample=0.001/200, optim='Adam', epsilon=0.01, random_init_adv=True, tent_coeff=0.0)

        elif self.args.model == "EATA":
             ### ERROR for segmenttaion
             self.logger.info("test-time adaptation:EATA")
             # fisher_size: number of samples to compute fisher information matrix.
             self.args.fisher_size = 2000
             # fisher_alpha: the trade-off between entropy and regularization loss, in Eqn. (8)
             self.args.fisher_alpha = 2000
             # e_margin: entropy margin E_0 in Eqn. (3) for filtering reliable samples
             self.args.e_margin = math.log(1000)*0.40
             # \epsilon in Eqn. (5) for filtering redundant samples
             self.args.d_margin = 0.05
             self.model = setup_eata(base_model, self.args, self.logger, act='sigmoid')

        elif self.args.model == "CoTTA":
            self.logger.info("test-time adaptation:CoTTA")
            self.args.OPTIM_STEPS = self.args.optim_steps
            self.args.MODEL_EPISODIC = self.args.model_episodic
            self.args.OPTIM_MT = 0.999
            self.args.OPTIM_RST = 0.01
            self.args.OPTIM_AP = 0.92
            self.model = setup_cotta(base_model, self.args, self.logger, act='sigmoid')

        elif self.args.model == "DUA":
            self.logger.info("test-time adaptation: DUA")
            self.model = setup_source(base_model, self.logger)

        elif self.args.model == "SAR":
            self.logger.info("test-time adaptation: SAR")
            # self.args.sar_margin_e0 = math.log(1000)*0.40
            # self.model = setup_sar(base_model, self.args, self.logger, act='sigmoid')

            model = sar.configure_model(base_model)
            params, param_names = sar.collect_params(model)
            base_optimizer = torch.optim.SGD
            optimizer = sar.SAM(params, base_optimizer, lr=args.initial_lr, momentum=0.9)
            self.model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(1000))

        else:
            raise NotImplementedError
        
    def evaluate(self):
        if args.dataset_name == 'RIGAPlus':
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                self.logger.info("Running inference: {}".format(inference_tag))
                inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                          inference_tag, self.logger)
      
        elif args.dataset_name == 'Prostate':
            from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)


        else:
            raise NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="TENT", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--arch', default="unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Prostate_BIDMC", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[384, 384], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/MRI_prostate', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/MRI_prostate/BIDMC_all.csv'],
                        help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=2, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Prostate/SiteA_RUNMC_batch_aug/checkpoints/model_final.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the model for each batch, choose True.')

    args = parser.parse_args()
    adapter = Tent_Adapter(args)
    adapter.evaluate()

