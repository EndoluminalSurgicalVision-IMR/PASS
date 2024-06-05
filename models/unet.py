from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from models.dpg_tta.arch import AdaptiveInstanceNorm


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


# =====================================================================================
# 2D UNet based on ResNet
# =======================================================================================

class UNet(nn.Module):
    """
       cannot use nn.parallel
       """
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

    def forward(self, x, rfeat=False, get_bottleneck_fea=False):
        x = F.relu(self.rn(x))
        bo_fea = x
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)

        if get_bottleneck_fea:
            return output, bo_fea
        elif rfeat:
            return output, fea
        else:
            return output
        
    def encoder(self, x):
        x = F.relu(self.rn(x))
        return x
    
    def decoder(self, x):
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)
        return output

    def close(self):
        for sf in self.sfs: sf.remove()


class UNet_v2(nn.Module):
    """
    supports nn.parallel
    """
    def __init__(self, resnet='resnet34', num_classes=2, patch_size=(512, 512), pretrained=False):
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

        self.layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*self.layers)
        self.rn = base_layers

        self.num_classes = num_classes
        # # self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        # self.sfs3 = SaveFeatures(base_layers[6])
        # self.sfs2 = SaveFeatures(base_layers[5])
        # self.sfs1 = SaveFeatures(base_layers[4])
        # self.sfs0 = SaveFeatures(base_layers[2])

        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x, rfeat=False, get_bottleneck_fea=False):
        res_features = []
        # print('Len', len(self.layers))
        for i in range(len(self.layers)):
            x = self.rn[i](x)
            # print(x.size())
            if i in [2, 4, 5, 6]:
                res_features.append(x)
        x = F.relu(x)
        bo_fea = x
        x = self.up1(x, res_features[3])
        x = self.up2(x, res_features[2])
        x = self.up3(x, res_features[1])
        x = self.up4(x, res_features[0])
        fea = x
        output = self.up5(x)
        if get_bottleneck_fea:
            return output, bo_fea
        elif rfeat:
            return output, fea
        else:
            return output
        

    def encoder(self, x):
        self.res_features = []
        # print('Len', len(self.layers))
        for i in range(len(self.layers)):
            x = self.rn[i](x)
            if i in [2, 4, 5, 6]:
                self.res_features.append(x)
        x = F.relu(x)
        return x
    

    def decoder(self, x):
        x = self.up1(x, self.res_features[3])
        x = self.up2(x, self.res_features[2])
        x = self.up3(x, self.res_features[1])
        x = self.up4(x, self.res_features[0])
        fea = x
        output = self.up5(x)
        return output
      

    def close(self):
        for sf in self.sfs: sf.remove()


#=====================================================================================
# Adaptive_UNet for <<On-the-Fly Test-time Adaptation for Medical Image Segmentation>>.
# =======================================================================================
class Adaptive_UNet(nn.Module):
    """
    support nn.parallel
    For <<On-the-Fly Test-time Adaptation for Medical Image Segmentation>>.
    """
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

        self.layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*self.layers)
        self.rn = base_layers
        nb_filter = [64, 64, 64, 64, 64, 128, 256, 512]

        self.num_classes = num_classes
        # # self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        # self.sfs3 = SaveFeatures(base_layers[6])
        # self.sfs2 = SaveFeatures(base_layers[5])
        # self.sfs1 = SaveFeatures(base_layers[4])
        # self.sfs0 = SaveFeatures(base_layers[2])

        self.adain1_e = AdaptiveInstanceNorm(nb_filter[0], 2048)
        self.adain2_e = AdaptiveInstanceNorm(nb_filter[1], 2048)
        self.adain3_e = AdaptiveInstanceNorm(nb_filter[2], 2048)
        self.adain4_e = AdaptiveInstanceNorm(nb_filter[3], 2048)
        self.adain5_e = AdaptiveInstanceNorm(nb_filter[4], 2048)
        self.adain6_e = AdaptiveInstanceNorm(nb_filter[5], 2048)
        self.adain7_e = AdaptiveInstanceNorm(nb_filter[6], 2048)
        self.adain8_e = AdaptiveInstanceNorm(nb_filter[7], 2048)

        self.adain1_d = AdaptiveInstanceNorm(256, 2048)
        self.adain2_d = AdaptiveInstanceNorm(256, 2048)
        self.adain3_d = AdaptiveInstanceNorm(256, 2048)
        self.adain4_d = AdaptiveInstanceNorm(256, 2048)

        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x, prior=None, rfeat=False, get_bottleneck_fea=False):
        res_features = []
        # for i in range(len(self.layers)):
        #     x = self.rn[i](x)
        #     x = self.adain1_e(x)
        #     if i in [2, 4, 5, 6]:
        #         res_features.append(x)

        # prior = torch.randn([x.shape[0], 512, 2, 2])

        x0_0 = self.rn[0](x)
        tmp = self.adain1_e(x0_0, prior)
        x1_0 = self.rn[1](tmp)
        tmp = self.adain2_e(x1_0, prior)
        x2_0 = self.rn[2](tmp)
        tmp = self.adain3_e(x2_0, prior)
        x3_0 = self.rn[3](tmp)
        tmp = self.adain4_e(x3_0, prior)
        x4_0 = self.rn[4](tmp)
        tmp = self.adain5_e(x4_0, prior)
        x5_0 = self.rn[5](tmp)
        tmp = self.adain6_e(x5_0, prior)
        x6_0 = self.rn[6](tmp)
        tmp = self.adain7_e(x6_0, prior)
        x7_0 = self.rn[7](tmp)

        res_features.append(x2_0)
        res_features.append(x4_0)
        res_features.append(x5_0)
        res_features.append(x6_0)

        x = F.relu(x7_0)
        bo_fea = x
        x3 = self.up1(x, res_features[3])
        x3 = self.adain1_d(x3, prior)
        x2 = self.up2(x3, res_features[2])
        x2 = self.adain2_d(x2, prior)
        x1 = self.up3(x2, res_features[1])
        x1 = self.adain3_d(x1, prior)
        x0 = self.up4(x1, res_features[0])
        x0 = self.adain4_d(x0, prior)

        fea = x0
        output = self.up5(x0)
        if get_bottleneck_fea:
            return output, bo_fea
        elif rfeat:
            return output, fea
        else:
            return output

    def close(self):
        for sf in self.sfs: sf.remove()

# ======================================================================================================
# Normalization network for <<Test-time adaptable neural networks for robust medical image segmentation>>.
# =======================================================================================================

# class ConvBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(ConvBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
#                      padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.stride = stride
#
#     def forward(self, x):
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.relu(out)
#
#         return out

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Norm_Indentity_Net(nn.Module):
    """
    support nn.parallel
    """
    def __init__(self, in_channel=3):
        super().__init__()

        self.layer1 = ConvBlock(in_channel, 16)
        self.layer2 = ConvBlock(16, 16)
        self.layer3 = ConvBlock(16, in_channel)

    def forward(self, x):
        delta = self.layer3(self.layer2(self.layer1(x)))
        x = x + delta
        return x


# ======================================================================================================
# DAE network for <<Test-time adaptable neural networks for robust medical image segmentation>>.
# =======================================================================================================


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block_IN(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_IN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_IN(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_IN, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net_for_DAE(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_for_DAE, self).__init__()
        nb_filter = [32, 64, 128, 256]
        # nb_filter = [32,64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1_exp = conv_block(ch_in=img_ch, ch_out=nb_filter[0])
        self.Conv2 = conv_block(ch_in=nb_filter[0], ch_out=nb_filter[1])
        self.Conv3 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[2])
        self.Conv4 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[3])


        self.Up4 = up_conv(ch_in=nb_filter[3], ch_out=nb_filter[2])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])

        self.Up3 = up_conv(ch_in=nb_filter[2], ch_out=nb_filter[1])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])

        self.Up2 = up_conv(ch_in=nb_filter[1], ch_out=nb_filter[0])
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.sigmoid = nn.Sigmoid()
        # self.use_sigmoid = use_sigmoid

    def forward(self, x):
        # encoding path
        x1 = self.Conv1_exp(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)


        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



#=================================================================================================
# 3D DAE network for <<Test-time adaptable neural networks for robust medical image segmentation>>.
#=================================================================================================


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(out_chan))

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.conv1(x))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        #print('scale', in_channel, 32 * (2 ** (depth + 1)), 32 * (2 ** (depth+1)))
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        #print('scale', in_channel, 32 * (2 ** (depth + 1)), 32*(2**depth)*2)
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels, normalization):

        super(OutputTransition, self).__init__()

        # if up_sample:
        #     self.final_conv = nn.Sequential(
        #         nn.Conv3d(inChans, n_labels, kernel_size=1),
        #         nn.Upsample(scale_factor=(1, 2, 2), mode = 'trilinear'),
        #         nn.Sigmoid()
        #     )
        # else:
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)

        if normalization == 'sigmoid':
            assert n_labels == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert n_labels > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        out = self.normalization(self.final_conv(x))
        return out


class UNet3D_for_DAE(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels=1, n_class=1, normalization=None, act='relu'):
        super(UNet3D_for_DAE, self).__init__()

        self.down_tr64 = DownTransition(in_channels,0,act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class, normalization)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['down_tr64', 'down_tr128', 'down_tr256', 'down_tr512']
        decoder_layers = ['up_tr256', 'up_tr128', 'up_tr64']
        out_layers = ['out_tr']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}
        return module_dict

