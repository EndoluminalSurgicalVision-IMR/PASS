"""
Adapted from https://github.com/ShishuaiHu/ProSFDA
"""
from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from models.unet import SaveFeatures, UnetBlock


def mix_data_prompt(data, data_prompt):
    # print(data.size(), data_prompt.size())
    new_data = data + data_prompt
    return new_data


#==================================================================================
# For the first stage in ProSFDA: prompt learning stage
# =================================================================================

class UNet_PLS(nn.Module):
    def __init__(self, pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()

        data_prompt = torch.zeros((3, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)

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
          

    def forward(self, x, training=False, get_bottleneck_fea=False):
        output = self.unet(mix_data_prompt(x, self.data_prompt), get_bottleneck_fea=get_bottleneck_fea)
        if training:
            return output, self.bn_f
        else:
            return output



#==================================================================================
# For the second stage in ProSFDA: feature alignment stage
# =================================================================================

class UNet_FAS(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        self.global_features = SaveFeatures(self.rn[-1])

    def forward(self, x, gfeat=False):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)

        if not gfeat:
            return output
        else:
            return output, self.global_features.features

    def close(self):
        for sf in self.sfs: sf.remove()
