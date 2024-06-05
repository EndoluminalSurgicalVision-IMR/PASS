from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import os
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn as nn

# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, save_all=False):
#         self.save_all = save_all
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#         if self.save_all:
#             self.all_data = []

#     def update(self, val, n=1):
#         self.val = val
#         if self.save_all:
#             if n == 1:
#                 self.all_data.append(val)
#             else:
#                 self.all_data.extend(val)
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, save_all=False):
        self.save_all = save_all
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.save_all:
            self.all_data = []

    def update(self, val, n=1):
        self.val = val
        if self.save_all:
            if n == 1:
                self.all_data.append(val)
            else:
                self.all_data.extend(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
 
def create_one_hot_from_2d_label(label, num_classes):
    """
    Input label: [B, H, W].
    Output label: [B, K, H, W]. The output label contains the background class in the 0th channel.
    """
    onehot_label = np.zeros(
        (label.shape[0], num_classes, label.shape[1], label.shape[2]))
    for i in range(num_classes):
        onehot_label[:, i, :, :] = (label == i).astype(np.int32)
    return onehot_label
       
        
def create_one_hot_from_3d_label(label, num_classes):
    """
    Input label: [D, H, W].
    Output label: [K, D, H, W]. The output label contains the background class in the 0th channel.
    """
    onehot_label = np.zeros(
        (num_classes, label.shape[0], label.shape[1], label.shape[2]))
    for i in range(num_classes):
        onehot_label[i, :, :, :] = (label == i).astype(np.int32)
    return onehot_label


def save_np2image(nimage, name, path):
    """
    Save a ndimage to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/' + name + '.png'
    if nimage.max() > 1 or nimage.min() < 0:
        nimage = (nimage-nimage.min()) / (nimage.max()-nimage.min())

    # Ndimage to PIL.Image
    nimage = np.array(nimage * 255, dtype=np.uint8)
    if len(nimage.shape) == 2:
        PIL = Image.fromarray(nimage).convert('L')
    else:
        PIL = Image.fromarray(nimage).convert('RGB')
    PIL.save(file_path)


def save_tensor2heatmap(tensor, name, path, color="YlGnBu_r"):
    """
    Save a tensor image to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path+ '/' + name +'_heatmap.png'

    # Tensor to PIL.Image
    PIL = transforms.ToPILImage()(tensor.cpu())

    # Heat map
    sns_plot = sns.heatmap(PIL,cmap=color)
    s1 = sns_plot.get_figure()
    s1.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()



def save_np2heatmap(nimage, name, path, color="YlGnBu_r"):
    """
    Save a ndimage to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path+ '/' + name +'_heatmap.png'

    # Ndimage to PIL.Image
    nimage = np.array(nimage)
    # PIL = Image.fromarray(nimage).convert('L')

    # Heat map
    sns_plot = sns.heatmap(nimage,cmap=color)
    s1 = sns_plot.get_figure()
    s1.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_tensor2nii(savedImg, saved_path, saved_name):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    savedImg = savedImg.cpu().numpy()
    newImg = sitk.GetImageFromArray(savedImg)
    sitk.WriteImage(newImg, os.path.join(saved_path, saved_name+'.nii'))


def save_np2nii(savedImg, saved_path, saved_name, direction=None):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    newImg = sitk.GetImageFromArray(savedImg)
    if direction is not None:
        newImg.SetDirection(direction)
    sitk.WriteImage(newImg,  os.path.join(saved_path, saved_name+'.nii'))


def save_tensor2image(tensor, name, path, normalize=False):
    """
    Save a tensor image to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/' + name + '.png'
    # Tensor to PIL.Image
    if normalize:
        tensor = (tensor- tensor.min()) / (tensor.max() - tensor.min())
    PIL = transforms.ToPILImage()(tensor.cpu())
    PIL.save(file_path)

def cal_trainable_params(model, include_bn=False):
    # model.requires_grad_(False)
    if include_bn:
        for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
    for k, v in model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params/1e6)
