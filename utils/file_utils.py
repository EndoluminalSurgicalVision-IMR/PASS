# -*- coding:utf-8 -*-
#from batchgenerators.utilities.file_and_folder_operations import *
import random
import hashlib
import datetime
import os


def maybe_mkdir_p(directory):
    os.makedirs(directory, exist_ok=True)

def check_folders(log_folder):
    tensorboard_folder = join(log_folder, 'logs')
    model_folder = join(log_folder, 'checkpoints')
    visualization_folder = join(log_folder, 'visualization')
    metrics_folder = join(log_folder, 'metrics')
    [maybe_mkdir_p(str(i)) for i in [tensorboard_folder, model_folder, visualization_folder, metrics_folder]]
    return tensorboard_folder, model_folder, visualization_folder, metrics_folder


def gen_random_str(length=12):
    return hashlib.md5((str(datetime.datetime.today())+str(random.randint(0, 999))).encode("utf-8")).hexdigest()[:length]


def split_path(path):
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)


join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split