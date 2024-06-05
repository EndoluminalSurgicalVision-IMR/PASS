# -*- coding:utf-8 -*-
"""
Models for different methods.
"""
from models.unet import UNet, Adaptive_UNet, U_Net_for_DAE, UNet3D_for_DAE
from models.dpg_tta.arch import priorunet
from models.unet_prompt import *
from models.PromptTTA.ours_ema import *


def get_model(args):
    if args.arch == 'unet_2d':
        network = UNet(args.num_classes)
        return network
    elif args.arch == 'adaptive_unet_2d':
        network =  Adaptive_UNet(args.num_classes)
        dpg = priorunet()
        return network, dpg