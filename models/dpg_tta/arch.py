import os
print(os.getcwd())
import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from torch.autograd import Function
from math import sqrt
# import models.unet.UnetBlock as UnetBlock
# from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152

__all__ = ['adaptiveunet', 'priorunet', 'UNet']



class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i]) + ft_l_list[
                    len(ft_h_list) - i - 1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale
        out = out + x
        return out


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        #         reflection_padding = kernel_size // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        #         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

########################################
### Architectures
########################################
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
    ) -> None:
        super().__init__()

        norm_layer = nn.BatchNorm2d  # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, 3)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class priorunet(nn.Module):
    def __init__(self, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(4, 4)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], input_channels, kernel_size=1)

    def forward(self, input, get_prior=False):
        # pdb.set_trace()
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        if get_prior:
            return x4_0
        else:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            output = self.final(x0_4)
            return output


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = torch.flatten(style, 1)
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class adaptiveunet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.adain1_e = AdaptiveInstanceNorm(nb_filter[0], 2048)
        self.adain2_e = AdaptiveInstanceNorm(nb_filter[1], 2048)
        self.adain3_e = AdaptiveInstanceNorm(nb_filter[2], 2048)
        self.adain4_e = AdaptiveInstanceNorm(nb_filter[3], 2048)

        self.adain1_d = AdaptiveInstanceNorm(nb_filter[3], 2048)
        self.adain2_d = AdaptiveInstanceNorm(nb_filter[2], 2048)
        self.adain3_d = AdaptiveInstanceNorm(nb_filter[1], 2048)
        self.adain4_d = AdaptiveInstanceNorm(nb_filter[0], 2048)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, prior):
        x0_0 = self.conv0_0(input)
        tmp = self.adain1_e(x0_0, prior)
        x1_0 = self.conv1_0(self.pool(tmp))
        tmp = self.adain2_e(x1_0, prior)
        x2_0 = self.conv2_0(self.pool(tmp))
        tmp = self.adain3_e(x2_0, prior)
        x3_0 = self.conv3_0(self.pool(tmp))
        tmp = self.adain4_e(x3_0, prior)
        x4_0 = self.conv4_0(self.pool(x3_0))

        # pdb.set_trace()

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.adain1_d(x3_1, prior)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.adain2_d(x2_2, prior)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3 = self.adain3_d(x1_3, prior)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.adain4_d(x0_4, prior)

        output = self.final(x0_4)
        return output

#
# class UNet_v2(nn.Module):
#     """
#     support nn.parallel
#     """
#     def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
#         super().__init__()
#         cut, lr_cut = [8, 6]
#
#         if resnet == 'resnet34':
#             base_model = resnet34
#         elif resnet == 'resnet18':
#             base_model = resnet18
#         elif resnet == 'resnet50':
#             base_model = resnet50
#         elif resnet == 'resnet101':
#             base_model = resnet101
#         elif resnet == 'resnet152':
#             base_model = resnet152
#         else:
#             raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
#                             'resnet101 and resnet152')
#
#         self.layers = list(base_model(pretrained=pretrained).children())[:cut]
#         base_layers = nn.Sequential(*self.layers)
#         self.rn = base_layers
#         nb_filter = [64, 64, 64, 64, 64, 128, 256, 512]
#
#         self.num_classes = num_classes
#         # # self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
#         # self.sfs3 = SaveFeatures(base_layers[6])
#         # self.sfs2 = SaveFeatures(base_layers[5])
#         # self.sfs1 = SaveFeatures(base_layers[4])
#         # self.sfs0 = SaveFeatures(base_layers[2])
#
#         self.adain1_e = AdaptiveInstanceNorm(nb_filter[0], 2048)
#         self.adain2_e = AdaptiveInstanceNorm(nb_filter[1], 2048)
#         self.adain3_e = AdaptiveInstanceNorm(nb_filter[2], 2048)
#         self.adain4_e = AdaptiveInstanceNorm(nb_filter[3], 2048)
#         self.adain5_e = AdaptiveInstanceNorm(nb_filter[4], 2048)
#         self.adain6_e = AdaptiveInstanceNorm(nb_filter[5], 2048)
#         self.adain7_e = AdaptiveInstanceNorm(nb_filter[6], 2048)
#         self.adain8_e = AdaptiveInstanceNorm(nb_filter[7], 2048)
#
#         self.adain1_d = AdaptiveInstanceNorm(256, 2048)
#         self.adain2_d = AdaptiveInstanceNorm(128, 2048)
#         self.adain3_d = AdaptiveInstanceNorm(64, 2048)
#         self.adain4_d = AdaptiveInstanceNorm(64, 2048)
#
#         self.up1 = UnetBlock(512, 256, 256)
#         self.up2 = UnetBlock(256, 128, 256)
#         self.up3 = UnetBlock(256, 64, 256)
#         self.up4 = UnetBlock(256, 64, 256)
#
#         self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
#
#
#     def forward(self, x, prior=None, rfeat=False, get_bottleneck_fea=False):
#         res_features = []
#         # for i in range(len(self.layers)):
#         #     x = self.rn[i](x)
#         #     x = self.adain1_e(x)
#         #     if i in [2, 4, 5, 6]:
#         #         res_features.append(x)
#
#         prior = torch.randn(x.size())
#
#         x0_0 = self.rn[0](x)
#         tmp = self.adain1_e(x0_0, prior)
#         x1_0 = self.rn[1](tmp)
#         tmp = self.adain2_e(x1_0, prior)
#         x2_0 = self.rn[2](tmp)
#         tmp = self.adain3_e(x2_0, prior)
#         x3_0 = self.rn[3](tmp)
#         tmp = self.adain4_e(x3_0, prior)
#         x4_0 = self.rn[4](tmp)
#         tmp = self.adain5_e(x4_0, prior)
#         x5_0 = self.rn[5](tmp)
#         tmp = self.adain5_e(x5_0, prior)
#         x6_0 = self.rn[6](tmp)
#         tmp = self.adain6_e(x6_0, prior)
#         x7_0 = self.rn[7](tmp)
#
#         res_features.append(x2_0)
#         res_features.append(x4_0)
#         res_features.append(x5_0)
#         res_features.append(x6_0)
#
#         x = F.relu(x7_0)
#         bo_fea = x
#         x3 = self.up1(x, res_features[3])
#         x3 = self.adain1_d(x3, prior)
#         x2 = self.up2(x3, res_features[2])
#         x2 = self.adain2_d(x2, prior)
#         x1 = self.up3(x2, res_features[1])
#         x1 = self.adain3_d(x1, prior)
#         x0 = self.up4(x1, res_features[0])
#         x0 = self.adain4_d(x0, prior)
#
#         fea = x0
#         output = self.up5(x0)
#         if get_bottleneck_fea:
#             return output, bo_fea
#         elif rfeat:
#             return output, fea
#         else:
#             return output
#
#     def close(self):
#         for sf in self.sfs: sf.remove()

if __name__ == '__main__':
    M = priorunet()
    b = torch.rand([2, 3, 512, 512])
    pp = M(b, True)
    print(pp.size())
    # M2 = adaptiveunet(num_classes=2)
    # cc = M2(b, pp)
    # print(cc.size())
    # b = torch.rand([2, 3, 512, 512])
    # M2 = UNet_v2(num_classes=2)
    # cc = M2(b)
    # print(cc.size())


    # # [2, 512, 2, 2]
    # style = EqualLinear(2048, 512*2)
    #
    # style.linear.bias.data[:512] = 1
    # style.linear.bias.data[512:] = 0
    # pp = torch.flatten(pp, 1)
    # style = style(pp).unsqueeze(2).unsqueeze(3)
    # gamma, beta = style.chunk(2, 1)
    # print(gamma.size(), beta.size())

