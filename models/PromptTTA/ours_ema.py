import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.file_utils import *
import models.moment_tta.losses as moment_tta_losses
import copy
import math


####
TENT_RIGA = ['Weighted_self_entropy_loss',
         {'weights':[1, 1], 'idc':[0, 1], 'act':'sigmoid'}]

TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

RN_w_CR_Prostate = ['RN_w_CR_loss',
         {'idc':[0], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]


LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766], 'act':'sigmoid_onelabel'}]


def custom_scheduler(optimizer, epoch, peak=1e-2, bottom=1e-4, period=20):
    if epoch == 0:
        lr = peak
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch % period == 0 and epoch != 0:
        lr = peak*0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = max(bottom, 0.1 * peak * (1 + math.cos(math.pi * (epoch % period) / period)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

class OnlineTargetWrapper(nn.Module):
    def __init__(self, model, decay_factor=0.94, min_momentum_constant=0.005):
        super(OnlineTargetWrapper, self).__init__()
        self.online_network = model
        self.target_network = copy.deepcopy(self.online_network)
        self.pretrained_state_dict = model.state_dict()
        self.decay_factor = decay_factor
        self.min_momentum_constant = min_momentum_constant
        self.mom_pre = 0.1

    def forward(self, x, training=False):
        if training:
            output, _, _, _ = self.online_network(x, training=True)
            self.update_target_network()
            return output
        else:
            return self.target_network(x, training=False)
        
    #reset the params of the online_network from the target_network
    def reset_online_network(self, mode='from_target'):
        if mode == 'from_target':
            self.online_network.load_state_dict(self.target_network.state_dict())
        else:
            assert mode == 'from_source'
            self.online_network.load_state_dict(self.pretrained_state_dict)
    
    #momentum update trainable params the target_network
    def update_target_network(self):
        mom_new = (self.mom_pre * self.decay_factor)
        momentum = mom_new + self.min_momentum_constant
        self.mom_pre = mom_new
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            if online_param.requires_grad:
                target_param.data.mul_(1-momentum)
                target_param.data.add_((momentum) * online_param.data)





