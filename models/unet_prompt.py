# -*- coding:utf-8 -*-
# For the prompt construction
from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from models.unet import SaveFeatures, UnetBlock, UNet
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, C, K):
        super(CrossAttention, self).__init__()
        # Q, K, V mapping

        self.query = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C)
        self.key = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
        self.value = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, groups=K)
    
    def forward(self, x_q, x_kv):
        query = self.query(x_q)
        key = self.key(x_kv)
        value = self.value(x_kv)

        batch_size, channels, height, width = query.size()
        batch_size, num_prompt, height, width = key.size()
        query = query.view(batch_size, channels, -1)
        key = key.view(batch_size,num_prompt, -1).permute(0, 2, 1)
        value = value.view(batch_size, num_prompt, -1)

        # calculate attention weights with scaled dot-product attention
        attn_weights = torch.matmul(query, key) / (num_prompt ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # sparsity atten
        num_non_zero = int(0.1 * attn_weights.size(1))
        sorted_values, sorted_indices = torch.sort(attn_weights, descending=True, dim=1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(1, sorted_indices[:, :num_non_zero], 1)
        attn_weights = attn_weights * mask

        # calculate attentive output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.view(batch_size, channels, height, width)

        return x_q + attn_output, attn_weights


##########################################################################
class UNet_SPTTA(nn.Module):
    def __init__(self, pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        data_prompt = torch.zeros(1, 512, patch_size[0]//32, patch_size[1]//32)
        self.data_prompt = nn.Parameter(data_prompt)
        self.prompt2feature = CrossAttention(C=512, K=512)
        self.data_adaptor = nn.Sequential(nn.Conv2d(3, 64, 1), 
                                 nn.InstanceNorm2d(64),
                                 nn.ReLU(inplace=True), 
                                 nn.Conv2d(64, 3, 1))
        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])
        self.bn_f = [SaveFeatures(self.unet.rn[0]),
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),  # 7
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2),
                     SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),  # 16
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),  # 29
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),  # 36
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),
                     ]
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def forward(self, x, training=False):
        perturbation = self.data_adaptor(x)
        adapted_x = x + perturbation
        bo_fea = self.unet.encoder(adapted_x)
        data_prompt = self.data_prompt.repeat(x.shape[0], 1, 1, 1)
        ## Q: Z, K/V: Pt
        # prompt matching
        adapted_bo_fea, attn_weights = self.prompt2feature(x_q=bo_fea, x_kv=data_prompt)
        # prompt matching
        output = self.unet.decoder(adapted_bo_fea)
        if training:
            return output, self.bn_f, perturbation, attn_weights
        else:
            return output


