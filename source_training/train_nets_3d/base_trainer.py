"""
Common training steps for 2D Test-Time Adaptation
"""
import sys
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
from models.unet import UNet, UNet_v2
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from utils.losses.seg_loss import DiceLoss, CELoss, Dice_CE_Loss


class BaseTrainer(object):
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
           6、Init optimizer and scheduler.

       After this call,
           All will be prepared for training.
       """

        self.args = args
        self.gpus = tuple(args.gpu)
        self.tag = args.tag
        self.log_folder = os.path.join(args.log_folder, args.model + '_' + args.tag)
        self.patch_size = tuple(args.patch_size)
        self.root = args.root
        self.recorder = Recorder(args)
        self.logger = self.recorder.logger
        self.writer = self.recorder.writer     
        self.tensorboard_folder, self.model_folder, self.visualization_folder, self.metrics_folder = self.recorder.tensorboard_folder, self.recorder.model_folder, self.recorder.visualization_folder,self.recorder.metrics_folder

        self.device = init_random_and_cudnn(self.gpus, manualseed=args.manualseed, bentchmark=True)

        self.init_dataloader()
        self.init_model()
        self.init_optimizer_and_scheduler()
        self.check_resume()
        self.init_criterion()

    def get_lr(self) -> int:
        return self.optimizer.param_groups[0]['lr']

    def init_dataloader(self):
        self.ts_dataset = None
        self.ts_dataloader = None
        self.tr_dataset = None
        self.tr_dataloader = None

    def init_model(self):
        self.model = UNet(num_classes=self.args.num_classes).to(self.device)
        if len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def init_criterion(self):
        """
        Define loss functions here
        """
        self.criterion = Dice_CE_Loss(n_classes=self.args.num_classes, weight=[1 for i in range(self.args.num_classes)]).to(self.device)

    def init_optimizer_and_scheduler(self):
        """
        Init optimizer and scheduler.
        """
        # init optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.initial_lr, momentum=0.99, nesterov=True)

        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.initial_lr, betas=(0.9, 0.999), weight_decay=0.0)

        else:
            raise NotImplementedError

        # init scheduler
        if hasattr(self.args, 'lr_scheduler'):
            if self.args.scheduler == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config.patience * 0.8), gamma=0.5)
        else:
            self.scheduler = None

    def check_resume(self):
        """
        Check resume for continue training.
        """
        self.start_epoch = 0
        if self.args.continue_training:
            model_path = self.args.pretrained_model
            assert isfile(model_path), 'missing model checkpoint!'
            params = torch.load(model_path)

            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            self.logger.info('Load model from: {}'.format(model_path))
            self.start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(self.start_epoch))

    def set_input(self, sample):
        input, target = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)

    def forward(self):
        """
        Define forward behavior here.
        Args:
            sample (tuple): an input-target pair
        """
        self.pred = self.model(self.input)

    def backward(self):
        """
        Compute the loss function.
        Args:
            sample (tuple): an input-target pair
        """

        self.loss = self.criteron(self.pred, self.target)
        self.loss.backward()


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def train(self):
        pass

    def test_one_case(self):
        pass

    def test(self):
        pass



