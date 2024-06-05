"""
Common training steps for Online Test-Time Adaptation
"""
from utils.init import init_random_and_cudnn, get_logger
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
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts,  target_collate_fn_tr_fda, collate_fn_ts,collate_fn_tr
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from tqdm import tqdm
from models import *
from utils.tools import AverageMeter, create_one_hot_from_2d_label, create_one_hot_from_3d_label
from models.TENT.setup import *
from models.PromptTTA.setup import *
from models.TENT.sam import SAM

class BaseAdapter(object):
    def __init__(
            self,
            args):
        """
       Steps:
           1、Init logger.
           2、Init device.
           3、Init seed.
           4、Init data_loader.
           5、Init model.
           6、Configure model for adaptaion with the optimizer and scheduler.

       After this call,
           All will be prepared for tta.
       """

        self.args = args
        self.gpus = tuple(args.gpu)
        self.tag = args.tag
        self.log_folder = os.path.join(args.log_folder, args.model + '_' + args.tag)
        self.patch_size = tuple(args.patch_size)
        self.ts_csv = tuple(args.ts_csv)

        self.tensorboard_folder, self.model_folder, self.visualization_folder, self.metrics_folder = check_folders(self.log_folder)
        self.writer = SummaryWriter(log_dir=self.tensorboard_folder)
        self.logger = get_logger(self.log_folder)
        print('RUNDIR: {}'.format(self.log_folder))
        self.logger.info('{}-TTA'.format(self.args.model))
        setting = {k: v for k, v in self.args._get_kwargs()}
        self.logger.info(setting)
        self.device = init_random_and_cudnn(self.gpus, manualseed=args.manualseed, bentchmark=True)
        if 'DAE' in self.args.model or 'DPG' in self.args.model:
            self.init_from_source_model()
        else:
            base_model = self.init_from_source_model()
            self.model = self.setup_model_w_optimizer(base_model)
        self.init_dataloader()
       
    def get_lr(self) -> int:
        return self.optimizer.param_groups[0]['lr']

    def init_dataloader(self):
        if self.args.tag in ['Base1_test', 'Base2_test', 'Base3_test', 'MESSIDOR', 'BinRushed']:
            ts_img_list, ts_label_list = convert_labeled_list(self.args.ts_csv, r=1)
            ts_dataset = RIGA_labeled_set(self.args.root, ts_img_list, ts_label_list, tuple(self.args.patch_size))
            test_batch = self.args.batch_size
            self.dataset_name = 'RIGAPlus'
            self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=test_batch,
                                                    num_workers=self.args.num_threads // 2,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn_ts,
                                                    drop_last=False)
            
        elif 'Prostate' in self.args.tag:
            ts_img_list, ts_label_list = convert_labeled_list(self.args.ts_csv, r=-1)
            ts_dataset = Prostate_labeled_set(self.args.root, ts_img_list, ts_label_list, 'test3d', self.args.patch_size, img_normalize=True)
            self.dataset_name = 'Prostate'
            # get volumetric data
            self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                        batch_size=1,
                                                        num_workers=self.args.num_threads//2,
                                                        shuffle=False,
                                                        pin_memory=True)

        return self.ts_dataloader


    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(self.args.pretrained_model)

        if self.args.arch == 'unet_2d':
            base_model = UNet(num_classes=self.args.num_classes)
            base_model.load_state_dict(self.pretrained_params['model_state_dict'])

    
        elif self.args.arch == 'unet_2d_sptta':
            base_model = UNet_SPTTA(pretrained_path=self.args.pretrained_model, num_classes=self.args.num_classes,  patch_size=self.args.patch_size)

        else:
            raise NotImplementedError

        base_model = base_model.to(self.device)
        
        return base_model

    def setup_model_w_optimizer(self, base_model):

        if self.args.model == "Source":
            self.logger.info("test-time adaptation: NONE")
            self.model = setup_source(base_model, self.logger)

        elif self.args.model == "PTBN":
            self.logger.info("test-time adaptation: PTBN")
            self.model = setup_norm(base_model, self.args, self.logger)

        elif self.args.model == "TENT":
            self.logger.info("test-time adaptation: TENT")
            self.model = setup_tent(base_model, self.args, self.logger, act='sigmoid')

        elif self.args.model == "TIPI":
             self.logger.info("test-time adaptation: TIPI")
             self.model = TIPI(base_model, lr_per_sample=0.001/200, optim='Adam', epsilon=0.01, random_init_adv=True, tent_coeff=0.5)

        elif self.args.model == "EATA":
             ### ERROR for segmenttaion
             self.logger.info("test-time adaptation: EATA")
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
            self.logger.info("test-time adaptation: CoTTA")
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
            # model = sar.configure_model(base_model)
            # params, param_names = sar.collect_params(model)
            # base_optimizer = torch.optim.SGD
            # optimizer = sar.SAM(params, base_optimizer, lr=self.args.initial_lr, momentum=0.9)
            # self.model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(1000))

            base_model = sar.configure_model(base_model)
            params, param_names = sar.collect_params(base_model )
            self.logger.info(param_names)
            self.args.sar_margin_e0 = math.log(1000)*0.40

            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(params, base_optimizer, lr=self.args.initial_lr, momentum=0.9)
            self.model = sar.SAR(base_model, self.optimizer, margin_e0=self.args.sar_margin_e0)

        else:
            self.logger.info("test-time adaptation: Ours")
            if 'EMA' in self.args.model:
                self.logger.info("CONFIGURE EMA")
                self.model, self.optimizer = setup_pt_tta_ema(base_model, self.args, self.logger)
                self.source_model = UNet(num_classes=self.args.num_classes)
                self.source_model.load_state_dict(self.pretrained_params['model_state_dict'])
                self.source_model.to(self.device)
            else:
                self.model, self.optimizer = setup_pt_tta(base_model, self.args, self.logger)

        return self.model
           
