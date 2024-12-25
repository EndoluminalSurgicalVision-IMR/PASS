# -*- coding:utf-8 -*-
from torch.utils import data
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import torch
import random
from scipy.special import comb
from scipy import ndimage
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform, Rot90Transform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.abstract_transforms import Compose

def load_all_nii(nii_list, case_normalize, patch_size, order=3):
    all_data = list()
    for nii in nii_list:
        nii_sitk = sitk.ReadImage(nii)
        nii_npy = sitk.GetArrayFromImage(nii_sitk)

        if len(nii_npy.shape) == 4:
            # print(('-------get T2--------'))
            # use the T2-weighted MRI
            nii_npy = nii_npy[0]

        if nii_npy.shape[-2] != patch_size[0] or nii_npy.shape[-1] != patch_size[1]:
            # down-sampling
            # print(('-------down-sampling--------'))
            nii_npy = ndimage.zoom(nii_npy, (1, patch_size[0] /nii_npy.shape[-2], patch_size[1] / nii_npy.shape[-1]), order=order)

        if case_normalize:
            # normalization
            # print(('-------normalize--------'))
            nii_npy = nii_npy.clip(np.percentile(nii_npy, 5), np.percentile(nii_npy, 95))
            nii_npy = (nii_npy - nii_npy.mean()) / nii_npy.std()

            
        for s in range(nii_npy.shape[0]):
            all_data.append(nii_npy[s])
      
        # if len(all_data) > 50:
        #     break
           
    return all_data


def load_all_nii_unlabeled(nii_list, case_normalize, patch_size, order=3):
    all_data = list()
    for nii in nii_list:
        nii_sitk = sitk.ReadImage(nii)
        nii_npy = sitk.GetArrayFromImage(nii_sitk)

        if len(nii_npy.shape) == 4:
            # use the T2-weighted MRI
            nii_npy = nii_npy[0]

        if nii_npy.shape[-2] != patch_size[0] or nii_npy.shape[-1] != patch_size[1]:
            # down-sampling
            nii_npy = ndimage.zoom(nii_npy, (1, patch_size[0] /nii_npy.shape[-2], patch_size[1] / nii_npy.shape[-1]), order=order)

        if case_normalize:
            # normalization
            nii_npy = nii_npy.clip(np.percentile(nii_npy, 5), np.percentile(nii_npy, 95))
            nii_npy = (nii_npy - nii_npy.mean()) / nii_npy.std()

        for s in range(nii_npy.shape[0]):
            if nii_npy[s].max() > 0:
                all_data.append(nii_npy[s])

    return all_data


class Prostate_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, split='train2d', target_size=(384, 384), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.split = split
        self.img_normalize = img_normalize
        self.target_size = target_size

        self.img_nii_list = [join(self.root, i) for i in self.img_list]
        self.label_nii_list = [join(self.root, i) for i in self.label_list]

        if self.split == 'train2d':
            self.img_data_list = load_all_nii(self.img_nii_list, case_normalize=img_normalize, patch_size=self.target_size, order=3)
            self.label_data_list = load_all_nii(self.label_nii_list, case_normalize=False, patch_size=self.target_size, order=0)
            self.len = len(self.img_data_list)
            # data augmentation
            self.train_transform = self.get_aug_transforms()
        
        elif self.split == 'val2d' or self.split == 'tta2d':
            self.img_data_list = load_all_nii(self.img_nii_list, case_normalize=img_normalize, patch_size=self.target_size, order=3)
            self.label_data_list = load_all_nii(self.label_nii_list, case_normalize=False, patch_size=self.target_size, order=0)
            self.len = len(self.img_data_list)

        else:
            self.len = len(self.img_nii_list)

       
        print('*******Len of dataset:{}, Split:{}********'.format(self.len, self.split))

    def __len__(self):
        return self.len
    

    def get_aug_transforms(self):
        # aug_v4
        train_transforms = [
        SpatialTransform_2(
            patch_size = self.target_size,
            do_elastic_deform=True, deformation_scale=(0, 0.2),
            do_rotation=True,
            angle_x=(-12, 12),
            angle_y=(-12, 12),
            do_scale=True, scale=(0.6, 1.4),
            # border_mode_data='constant',border_cval_data=0,
            # border_mode_seg='constant', border_cval_seg=0,
            border_mode_data='nearest',
            border_mode_seg='nearest',
            order_seg=0, order_data=3,random_crop=False,
            p_el_per_sample=0.15, p_rot_per_sample=0.15, p_scale_per_sample=0.15), 
        Rot90Transform(p_per_sample=0.2, num_rot=(1, 2, 3), axes=(0, 1)), 
        MirrorTransform(p_per_sample=0.2, axes=(0, 1)),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
        GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.15),
        # added at 10.7/14.39
        BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15),
        GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15),
        ContrastAugmentationTransform(per_channel=True, p_per_sample=0.2)]
        return Compose(train_transforms)
    
    def get_2d_slice(self, item):
        img_npy = self.img_data_list[item]
        label_npy = self.label_data_list[item]
        # print('bbbbbefore get 2d slice', label_npy.min(), label_npy.max())
        label_npy[label_npy >= 1] = 1
        # print('before get 2d slice', label_npy.min(), label_npy.max())
        # batch data augmentation
        sample_dict = {'data': np.expand_dims(np.expand_dims(img_npy, 0), 0), 'seg': np.expand_dims(np.expand_dims(label_npy, 0), 0)}
        # input dim for batch aug: [1, 1, H, W]
        sample_dict = self.train_transform(**sample_dict)
         # after batch aug: [1, 1, H, W]
        img_npy = sample_dict['data'][0]
        label_npy = sample_dict['seg'][0]
        # [1, H, W]
        # print('after get 2d slice', label_npy.min(), label_npy.max())
        return torch.from_numpy(img_npy.repeat(3, axis=0)).float(), torch.from_numpy(label_npy).float()
    
    def get_3d_volume(self, item):
        case_name = os.path.basename(self.img_nii_list[item]).split('.')[-3]
        img = sitk.ReadImage(self.img_nii_list[item])
        img_npy = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(self.label_nii_list[item])
        label_npy = sitk.GetArrayFromImage(label)

        # #将img_npy和label_npy沿着二三维度平面旋转180度
        # img_npy = np.flip(img_npy, axis=-1).copy()
        # label_npy = np.flip(label_npy, axis=-1).copy()
        if len(img_npy.shape) == 4:
            # use the T2-weighted MRI
            img_npy = img_npy[0]

        if img_npy.shape[-1] != self.target_size[0]:
            # down-sampling
            img_npy = ndimage.zoom(img_npy, (1, self.target_size[0] / img_npy.shape[-2], self.target_size[1] / img_npy.shape[-1]), order=3)
            label_npy = ndimage.zoom(label_npy, (1, self.target_size[0] / label_npy.shape[-2], self.target_size[-2] / label_npy.shape[-1]), order=0)
       
        if self.img_normalize:
            img_npy = img_npy.clip(np.percentile(img_npy, 5), np.percentile(img_npy, 95))
            img_npy = (img_npy - img_npy.mean()) / img_npy.std()
        return torch.from_numpy(img_npy).float(), torch.from_numpy(label_npy).float(), case_name

    def __getitem__(self, item):
        if self.split == 'train2d':
            img,label = self.get_2d_slice(item)
            return img, label
        
        elif self.split == 'val2d':
            img_npy = self.img_data_list[item]
            label_npy = self.label_data_list[item]
            mask_npy = np.zeros_like(label_npy)
            mask_npy[label_npy >= 1] = 1
            # if self.img_normalize:
            #     img_npy = img_npy.clip(np.percentile(img_npy, 5), np.percentile(img_npy, 95))
            #     img_npy = (img_npy - img_npy.mean()) / img_npy.std()
            return torch.from_numpy(np.expand_dims(img_npy, 0).repeat(3, axis=0)), torch.from_numpy(mask_npy[np.newaxis]).float()
            

        elif self.split == 'tta2d':
            img_npy = self.img_data_list[item]
            label_npy = self.label_data_list[item]
            mask_npy = np.zeros_like(label_npy)
            mask_npy[label_npy >= 1] = 1
            return np.expand_dims(img_npy, 0).repeat(3, axis=0), mask_npy[np.newaxis]#str(item)#, None

        else:
            assert self.split == 'test3d'
            img,label,name = self.get_3d_volume(item)
            label[label>=1] = 1
            return img,label,name


class Prostate_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(128, 128), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.target_size = target_size
        self.img_normalize = img_normalize

        self.img_nii_list = [join(self.root, i) for i in self.img_list]
        self.img_data_list = load_all_nii_unlabeled(self.img_nii_list, case_normalize=self.img_normalize, patch_size=self.target_size, order=3)

        self.len = len(self.img_data_list)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_npy = self.img_data_list[item]
        # if self.img_normalize:
        #     img_npy = img_npy.clip(np.percentile(img_npy, 5), np.percentile(img_npy, 95))
        #     img_npy = (img_npy - img_npy.mean()) / img_npy.std()
        return np.expand_dims(img_npy, 0).repeat(3, axis=0), None, None 
