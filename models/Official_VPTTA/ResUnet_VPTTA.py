from torch import nn
import torch
import torch.nn.functional as F
import math
import os
import torch.utils.model_zoo as model_zoo
from models.unet import UNet
import numpy as np

hub_dir = torch.hub.get_dir()
model_dir = os.path.join(hub_dir, 'checkpoints')


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

in_affine = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, ini=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(ini, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        sfs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        sfs.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        sfs.append(x)

        x = self.layer2(x)
        sfs.append(x)

        x = self.layer3(x)
        sfs.append(x)

        x = self.layer4(x)
        sfs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, sfs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34-333f7ec4.pth')), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet50-19c8e357.pth')), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

#############convert################


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x


def convert_encoder_to_target(net, norm, start=0, end=5, verbose=True, bottleneck=False, input_size=512, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx, fea_size=input_size // 2)
            idx += 1
        else:
            down_sample = 2 ** (1 + i)

            for j, block in enumerate(layer):
                block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                if bottleneck:
                    block.bn3 = convert_norm(block.bn3, norm, block.bn3.num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
                if block.downsample is not None:
                    block.downsample[1] = convert_norm(block.downsample[1], norm, block.downsample[1].num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
    return net


def convert_decoder_to_target(net, norm, start=0, end=5, verbose=True, input_size=512, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    # layers = [net[0], net[1], net[2], net[3], net[4]]
    layers = [net[0], net[1], net[2], net[3]]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 4:
            net[4] = convert_norm(layer, norm, layer.num_features, idx, input_size)
            idx += 1
        else:
            down_sample = 2 ** (4 - i)
            layer.bn = convert_norm(layer.bn, norm, layer.bn.num_features, idx, input_size // down_sample)
            idx += 1
    return net


#########################
class SaveFeatures():
    def __init__(self, m, n):
        self.features = None
        self.name = n
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        out = self.bn(F.relu(cat_p))
        return out


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, convert=True, newBN=AdaBN, warm_n=5):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet50':
            base_model = resnet50
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise Exception('The Resnet Model only accept resnet34 and resnet50!')

        self.res = base_model(pretrained=pretrained)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        # self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        #self.bnout = nn.BatchNorm2d(32)

        #self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        # Convert BN layer
        self.newBN = newBN
        if convert:
            self.res = convert_encoder_to_target(self.res, newBN, start=0, end=4, verbose=False, bottleneck=bottleneck, warm_n=warm_n)
            self.up1, self.up2, self.up3, self.up4 = convert_decoder_to_target(
                [self.up1, self.up2, self.up3, self.up4], newBN, start=0, end=4, verbose=False, warm_n=warm_n)

        # Save the output feature of each BN layer.
        self.feature_hooks = []
        layers = [self.res.bn1, self.res.layer1, self.res.layer2, self.res.layer3, self.res.layer4]
        for i, layer in enumerate(layers):
            if i == 0:
                self.feature_hooks.append(SaveFeatures(layer, 'first_bn'))
            else:
                for j, block in enumerate(layer):
                    self.feature_hooks.append(SaveFeatures(block.bn1, str(i)+'-bn1'))      # BasicBlock
                    self.feature_hooks.append(SaveFeatures(block.bn2, str(i)+'-bn2'))      # BasicBlock
                    if resnet == 'resnet50':
                        self.feature_hooks.append(SaveFeatures(block.bn3, str(i)+'-bn3'))  # Bottleneck
                    if block.downsample is not None:
                        self.feature_hooks.append(SaveFeatures(block.downsample[1], str(i)+'-downsample_bn'))
        self.feature_hooks.append(SaveFeatures(self.up1.bn, '1-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up2.bn, '2-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up3.bn, '3-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up4.bn, '4-up_bn'))
        # self.feature_hooks.append(SaveFeatures(self.bnout, 'last_bn'))

    def change_BN_status(self, new_sample=True):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample

    def reset_sample_num(self):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = 0

    def forward(self, x):
        x, sfs = self.res(x)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        seg_output = self.up5(x)
        # head_input = F.relu(self.bnout(x))

        # seg_output = self.seg_head(head_input)

        return seg_output#, sfs, head_input

    def close(self):
        for sf in self.sfs:
            sf.remove()





