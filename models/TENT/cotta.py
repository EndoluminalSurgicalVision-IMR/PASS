# Adapted from https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
# import my_transforms as my_transforms
from time import time
import logging

# KATANA: Simple Post-Training Robustness Using Test Time Augmentations
# https://arxiv.org/pdf/2109.08191v1.pdf
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random
from tqdm import tqdm
from datasets.dataloaders.Prostate_dataloader import GaussianNoiseTransform, GaussianBlurTransform, BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string


def get_my_tta_transforms():
        aug_transforms = [
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.3),
        GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.3),
        BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.3),
        GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.3),
        ContrastAugmentationTransform(per_channel=True, p_per_sample=0.3)]
        return BGCompose(aug_transforms)

def get_tta_transforms(soft=True, clip_inputs=False, img_shape = (32, 32, 3)):
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    # p_hflip = 0.5
    ## for RIGA+
    tta_transforms = transforms.Compose([
        # Clip(0.0, 1.0), 
        # ColorJitterPro(
        #  brightness=[0.8, 1.2] if soft else [0.6, 1.4],
        #     contrast=[0.85, 1.15] if soft else [0.7, 1.3],
        #     saturation=[0.75, 1.25] if soft else [0.5, 1.5],
        #     hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
        #     gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        # ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        # transforms.RandomAffine(
        #     degrees=[-8, 8] if soft else [-15, 15],
        #     translate=(1/16, 1/16),
        #     scale=(0.95, 1.05) if soft else (0.9, 1.1),
        #     shear=None,
        #     interpolation=PIL.Image.BILINEAR,
        #     fill=None
        # ),
        # transforms.RandomAffine(
        #     degrees=[-8, 8] if soft else [-15, 15],
        #     translate=(1/16, 1/16),
        #     scale=(0.95, 1.05) if soft else (0.9, 1.1),
        #     # shear=None,
        #     # interpolation=PIL.Image.BILINEAR,
        #     # fill=None
        # ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        # transforms.CenterCrop(size=n_pixels),
        # transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, 0.005),
        # Clip(clip_min, clip_max)
    ])


    # tta_transforms = transforms.Compose([
    #     # ColorJitterPro(
    #     #  brightness=[0.8, 1.2],
    #     #     contrast=[0.85, 1.15],
    #     #     # saturation=[0.75, 1.25],
    #     #     # hue=[-0.03, 0.03],
    #     #     gamma=[0.85, 1.15]
    #     # ),
    #     transforms.GaussianBlur(kernel_size=5, sigma=[0.5, 1.5]),
    #     GaussianNoise(0, 0.05),
    #     # Clip(clip_min, clip_max)
    # ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, img_shape, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9, act='softmax', task= 'cls', aug_transform_type = 'tv_trans'):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.task = task
        self.act = act
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform_type = aug_transform_type
        if self.transform_type == 'tv_trans':
            self.transform = get_tta_transforms(img_shape = img_shape)
        else:
            self.transform = get_my_tta_transforms()# get_tta_transforms(img_shape = img_shape)    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in tqdm(range(self.steps)):
            if self.task == 'cls':
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
            else:
                assert self.task == 'seg'
                outputs = self.forward_and_adapt_seg(x, self.model, self.optimizer, self.act)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema
    

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_seg(self, x, model, optimizer, act='softmax'):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_pred = self.model_anchor(x)
        if act == 'softmax':
            anchor_prob = torch.nn.functional.softmax(anchor_pred, dim=1).max(1)[0]
        else:
            assert act == 'sigmoid'
            anchor_prob = torch.sigmoid(anchor_pred).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            if self.transform_type == 'tv_trans':
                aug_x = self.transform(x)
            else:
                sample_dict = {'data': x.cpu().numpy()}
                # input dim for batch aug: [1, 1, H, W]
                sample_dict = self.transform(**sample_dict)
                # after batch aug: [1, 1, H, W]
                aug_x_npy = sample_dict['data']
                aug_x = torch.from_numpy(aug_x_npy).to(x.device).to(dtype=torch.float32)
            outputs_  = self.model_ema(aug_x).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean()<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
       
        # Student update
        if act == 'softmax':
            loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        else:
            assert act == 'sigmoid'
            loss = (sigmoid_entropy(outputs, outputs_ema)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

# @torch.jit.script
# def sigmoid_entropy(x, x_ema):# -> torch.Tensor:
#     """Entropy of sigmoid distribution from logits."""
#     return -(x_ema.sigmoid().max(1)[0] * torch.log(x.sigmoid().max(1)[0]+1e-6)).sum(1)


@torch.jit.script
def sigmoid_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of sigmoid distribution from logits."""
    return -(x_ema.sigmoid() * torch.log(x.sigmoid()+1e-6)).mean(1)

# @torch.jit.script
# def sigmoid_entropy(x, x_ema):# -> torch.Tensor:
#     """Entropy of sigmoid distribution from logits."""
#     return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"