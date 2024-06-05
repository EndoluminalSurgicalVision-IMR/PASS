from torch.utils.tensorboard import SummaryWriter
import os.path
import os
import time

import matplotlib.pyplot as plt
import csv
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from monai.utils import set_determinism
import numpy as np
from collections import OrderedDict
import logging
import shutil
import subprocess
import sys


class Recorder:
    def __init__(self, config, flag='train'):
        self.config = config
        # result_dir: - model- tag
        log_folder = os.path.join(config.log_folder, config.model + '_' + config.tag)
        if flag == 'train':
            self.save_dir = log_folder + '/'+ config.method_tag +'/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
            self.logger = get_logger(self.save_dir)
            print('RUNDIR: {}'.format(self.save_dir))
            self.logger.info('{}-Train'.format(self.config.model))
            
            self.tensorboard_folder = os.path.join(self.save_dir, 'logs')
            self.model_folder =  os.path.join(self.save_dir, 'checkpoints')
            self.visualization_folder =  os.path.join(self.save_dir, 'visualization')
            self.metrics_folder =  os.path.join(self.save_dir, 'metrics')
            [os.makedirs(str(i), exist_ok=True) for i in [self.tensorboard_folder, self.model_folder, self.visualization_folder, self.metrics_folder]]
            # record config
            self.writer = SummaryWriter(log_dir=self.tensorboard_folder)
        else:
            self.save_dir = log_folder + '/'+ config.method_tag +'/infer_results'
            self.logger = get_logger(self.save_dir)
            print('RUNDIR: {}'.format(self.save_dir))
            self.visualization_folder =  os.path.join(self.save_dir, 'visualization')
            self.metrics_folder =  os.path.join(self.save_dir, 'metrics')
            [os.makedirs(str(i), exist_ok=True) for i in [self.visualization_folder, self.metrics_folder]]

        setting = {k: v for k, v in self.config._get_kwargs()}
        self.logger.info(setting)

    def logger_shutdown(self):
        import logging
        logging.shutdown()

    def plot_loss(self, start_epoch, epochs, val_freq, train_loss):
        # Draw the training loss.
        x1 = range(start_epoch, epochs, val_freq)
        y1 = train_loss
        plt.plot(x1, y1, '-')
        plt.title('Training loss vs.epochs')
        plt.xlabel('epoch')
        plt.ylabel('Training loss')
        # plt.show()
        plt.savefig(self.save_dir + '/training_loss.jpg')
        # get current figure
        fig = plt.gcf()
        plt.close(fig)

        return

    def plot_val_metrics(self, start_epoch, epochs, val_freq, metrics):
        # Draw the validation metrics.
        x1 = range(start_epoch, epochs, val_freq)
        y1 = metrics
        plt.plot(x1, y1, '-')
        plt.title('Validation results vs.epochs')
        plt.xlabel('epoch')
        plt.ylabel('Validation metric')
        # plt.show()
        plt.savefig(self.save_dir + '/validation_results.jpg')
        # get current figure
        fig = plt.gcf()
        plt.close(fig)

        return

def init_random_and_cudnn(gpu_ids, bentchmark, manualseed=100):
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device('cuda:%d' % gpu_ids[0]
                               if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu')

    # Set seed
    if manualseed is None:
        manualseed = random.randint(1, 10000)

    print('******* manual seed********', manualseed)
    np.random.seed(manualseed)
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    set_determinism(manualseed)

    if use_cuda:
        torch.cuda.manual_seed(manualseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = bentchmark

    return device


def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger
