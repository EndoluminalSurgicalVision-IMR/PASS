"""
Common inference steps.
"""
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import os
import torch
import numpy as np
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet, UNet_v2
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts


class BaseTester(object):
    def __init__(
            self,
            args):
        """
       Steps:
           1、Init logger.
           2、Init device.
           3、Init data_loader.
           4、Load model.

       After this call,
           All will be prepared for inference.
       """

        self.args = args
        self.gpus = tuple(args.gpu)
        self.tag = args.tag
        self.log_folder = os.path.join(args.log_folder, args.model + '_' + args.tag)
        self.patch_size = tuple(args.patch_size)
        self.ts_csv = tuple(args.ts_csv)
        self.recorder = Recorder(args, flag='test')
        self.logger = self.recorder.logger

        self.visualization_folder, self.metrics_folder = self.recorder.visualization_folder, self.recorder.metrics_folder
        self.model_path = args.model_path
        self.device = init_random_and_cudnn(self.gpus, manualseed=args.manualseed, bentchmark=True)

        self.init_dataloader()
        self.load_model()

    def init_dataloader(self):
        self.ts_dataset = None
        self.ts_dataloader = None

    def load_model(self):
        self.model = UNet(num_classes=self.args.num_classes).to(self.device)
        if len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        assert isfile(self.model_path), 'missing model checkpoint {}!'.format(self.model_path)
        self.logger.info('Load model for inference:{}'.format(self.model_path))
        params = torch.load(self.model_path)
        self.model.load_state_dict(params['model_state_dict'])
        
    def test_one_case(self):
        pass

    def test(self):
        pass





