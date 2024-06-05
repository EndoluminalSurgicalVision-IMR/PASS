# -*- coding:utf-8 -*-
import numpy as np
import torch
import medpy.metric.binary as medpy_binay_metric


def get_dice_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_hard_dice(outputs, masks, std=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_list.append(get_dice_threshold(output, mask, threshold=0.5))
    if not std:
        return np.mean(dice_list)
    else:
        return np.mean(dice_list), np.std(dice_list), dice_list


class SegMetric_Numpy(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.SE = 0.0
        self.SP = 0.0
        self.PC = 0.0
        self.F1 = 0.0
        self.JS = 0.0
        self.DC = 0.0
        self.length = 0.0

        self.acc_i = 0.0
        self.SE_i = 0.0
        self.SP_i = 0.0
        self.PC_i = 0.0
        self.F1_i = 0.0
        self.JS_i = 0.0
        self.DC_i = 0.0

    def update(self, SR, GT, threshold=0.5):
        """
            SR:  Segmentation Result, a ndarray with the shape of [D, H, W]
            GT:  Ground Truth , a ndarray with the shape of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.shape == GT.shape

        SR = (SR > threshold).astype(int)
        GT = (GT == np.max(GT)).astype(int)
        corr = np.sum(SR == GT)
        if len(SR.shape) == 2:
            tensor_size = SR.shape[0] * SR.shape[1]
        else:
            tensor_size = SR.shape[0] * SR.shape[1] * SR.shape[2]
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))
        FN = ((SR == 0) & (GT == 1))
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))
        FP = ((SR == 1) & (GT == 0))

        self.SE_i = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)
        self.SP_i = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)
        self.PC_i = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = np.sum((SR + GT) == 2)
        Union = np.sum((SR + GT) >= 1)

        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1

    @property
    def get_current(self):
        return self.acc_i, self.SE_i, self.SP_i, self.PC_i, self.F1_i, self.JS_i, self.DC_i

    @property
    def get_avg(self):
        return self.acc / self.length, self.SE / self.length, \
               self.SP / self.length, self.PC / self.length, self.F1 / self.length, \
               self.JS / self.length, self.DC / self.length



def get_dice_assd_hd(target_array, pred_array, logger):
    if target_array.sum() == 0 and pred_array.sum() == 0:
        dc = 1
        assd = 0
        hd = 0

    elif target_array.sum() == 0 or pred_array.sum() == 0:
        logger.warning(
            'Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
        dc = 0
        assd = 1
        hd = 1
    else:
        target_array = np.asarray(target_array > 0.5).astype(np.bool)
        pred_array = np.asarray(pred_array > 0.5).astype(np.bool)

        dc = medpy_binay_metric.dc(target_array, pred_array)
        # dc = medpy_binay_metric.dc(target_array, pred_array)
        # bg_dc = medpy_binay_metric.dc(target_array == 0, pred_array == 0)
        # dc = (dc + bg_dc) / 2
        assd = medpy_binay_metric.assd(pred_array,
                                    target_array)
        hd = medpy_binay_metric.hd95(pred_array,
                                    target_array)
    return dc, assd, hd


def get_mean_dsc(pred_array, target_array):
    """
    :param pred_array: 3D array
    :param target_array: 3D array
    :return:
    """
    target_array = np.asarray(target_array > 0.5).astype(np.bool)
    pred_array = np.asarray(pred_array > 0.5).astype(np.bool)
    dc = medpy_binay_metric.dc(target_array, pred_array)
    bg_dc = medpy_binay_metric.dc(target_array == 0, pred_array == 0)
    print('dc', dc)
    print('bg_dc', bg_dc)
    return (dc + bg_dc) / 2

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))