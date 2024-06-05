import torch
from models.TENT.tipi import TIPI
import models.TENT.ttn as ttn
import models.TENT.tent as tent
import models.TENT.eata as eata
import models.TENT.cotta as cotta
import models.TENT.sar as sar

def setup_source(model, logger):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model, args, logger):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = ttn.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = ttn.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_optimizer(params, args):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if args.optimizer == 'Adam':
        return torch.optim.Adam(params,
                    lr=args.initial_lr,
                    betas=(0.9, 0.999),
                    weight_decay= 0.)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(params,
                   lr=args.initial_lr,
                   momentum=0.9,
                   dampening=0.,
                   weight_decay=0.,
                   nesterov=True)
    else:
        raise NotImplementedError


def setup_tent(model, args, logger, act='softmax'):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params, args)
    tent_model = tent.Tent(model, optimizer,
                           steps=args.optim_steps,
                           episodic=args.model_episodic,
                           act=act)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_eata(model, args, logger, act='softmax'):
    """Set up EATA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = eata.configure_model(model)
    params, param_names = eata.collect_params(model)
    optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
    adapt_model = eata.EATA(model, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return adapt_model

def setup_cotta(model, cfg, logger, act='sigmoid'):
    """Set up CoTTA adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    if 'Base' in cfg.tag:
        # torchvision transforms
        aug_type = 'tv_trans'
    else:
        # batchgenerators transforms
        aug_type = 'bg_trans'
    cotta_model = cotta.CoTTA(model, optimizer, (cfg.patch_size[0], cfg.patch_size[1], 3),
                           steps=cfg.OPTIM_STEPS,
                           episodic=cfg.MODEL_EPISODIC, 
                           mt_alpha=cfg.OPTIM_MT, 
                           rst_m=cfg.OPTIM_RST, 
                           ap=cfg.OPTIM_AP,
                           act=act, 
                           task='seg',
                           aug_transform_type=aug_type)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def setup_sar(model, args, logger, act='sigmoid'):
    """Set up sar adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = sar.SAM(params, base_optimizer, lr=args.initial_lr, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=args.sar_margin_e0, task='seg', act=act)
    # logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return adapt_model