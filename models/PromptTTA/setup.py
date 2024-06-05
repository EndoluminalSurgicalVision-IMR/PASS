import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from models.PromptTTA.ours_ema import OnlineTargetWrapper

def setup_optimizer(params, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.initial_lr, momentum=0.99, nesterov=True)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.initial_lr, betas=(0.9, 0.999), weight_decay=0.0)

    return optimizer


def configure_model_2d(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    for name, param in model.named_parameters():
        if name.find('prompt') != -1 or name.find('data') != -1 or name.find('eff') != -1:
            param.requires_grad = True
    return model

def setup_pt_tta(model, args, logger):
    """Set up the adaptation for prompt-tta methods.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    if args.only_bn_updated:
        model = configure_model_2d(model)
        logger.info('normalization statistics updated')
    else:
        logger.info('WARNING all var updated')
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = setup_optimizer(params, args)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", params)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return model, optimizer

def setup_pt_tta_ema(model, args, logger):
    """Set up the adaptation for prompt-tta methods.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model_2d(model)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = setup_optimizer(params, args)

    tta_model = OnlineTargetWrapper(model, args.ema_decay, args.min_momentum_constant)
    
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", params)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tta_model, optimizer

