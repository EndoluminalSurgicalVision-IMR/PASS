import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
import os
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=[1., 1., 1.]):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape does not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0

        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        return loss / self.n_classes



class CELoss(nn.Module):
    def __init__(self, n_classes, weight=[1., 1., 1.]):
        super(CELoss, self).__init__()
        self.n_classes = n_classes
        self.weight = torch.from_numpy(np.array(weight))

        self.ce_loss = CrossEntropyLoss(weight=self.weight.float())

    def forward(self, inputs, target):
        ce_loss = self.ce_loss(inputs, target)
        return ce_loss
    
    

class Dice_CE_Loss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(Dice_CE_Loss, self).__init__()
        self.n_classes = n_classes
        if weight == None:
            self.weight = torch.ones(n_classes)
        else:
             self.weight = torch.from_numpy(np.array(weight))
        self._ce_loss = CrossEntropyLoss(weight=self.weight.float())
        #self._ce_loss = CrossEntropyLoss()
        
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        ce_loss = self._ce_loss(inputs, target.long())
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape does not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * self.weight[i]
        return loss / self.n_classes + ce_loss
       

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

    def forward(self, input, target, sigmoid=True):
        '''
        :param input: (N,*), input must be the original probability image
        :param target: (N,*) * is any other dims but be the same with input,
        : shape is  N -1 or  N 1 H W
        :return:  sigmod + BCELoss +  sigmod + DiceLoss
        '''
        # N sample`s average
        bce = F.binary_cross_entropy_with_logits(input, target)
        if sigmoid:
            input = torch.sigmoid(input)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return self.bce_weight * bce + self.dice_weight * dice
