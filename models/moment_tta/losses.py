#!/usr/env/bin python3.6

from typing import List, Tuple, cast
# from functools import reduce
from operator import add
from functools import reduce
import numpy as np
import torch
from torch import einsum
from torch import Tensor
from torch import nn
import pandas as pd
import torch.nn.functional as F
from operator import mul
import models.moment_tta.utils as utils
from models.moment_tta.utils import soft_compactness, soft_length, soft_size, soft_inertia, soft_eccentricity, soft_moment
from einops import rearrange
from typing import Optional, Sequence
import pdb

from models.moment_tta.utils import simplex, sset, probs2one_hot
import torch.nn.modules.padding



class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        # self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        # self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.__fn__ = getattr(utils, kwargs['fn'])

        # print(f"> Initialized {self.__class__.__name__} with kwargs:")
        # pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape
        predicted_mask = probs2one_hot(probs).detach()
        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape
        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        # assert len(upper_z) == len(lower_b) == len(filenames)

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class NaivePenalty(AbstractConstraints):
    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        return F.relu(z_) ** 2


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum(f"bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum(f"bkwh->bk", pc) + einsum(f"bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)
        loss = divided.mean()

        return loss


class EntKLPropWMoment():
    """
    Entropy minimization with KL proportion regularisation
    """

    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        # self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        # self.__momfn__ = getattr(__import__('utils'), kwargs['moment_fn'])
        self.__fn__ = getattr(utils, kwargs['fn'])
        self.__momfn__ = getattr(utils, kwargs['moment_fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.mom_est: List[float] = kwargs["mom_est"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_moment: float = kwargs["lamb_moment"]
        self.margin: float = kwargs["margin"]
        self.temp: float = kwargs["temp"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, c, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop: Tensor = self.__fn__(probs, self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:, :, 0]
                bounds = bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop) * bounds / (w * h)
            gt_prop = gt_prop[:, :, 0]
        else:
            gt_prop: Tensor = self.__fn__(target, self.power)
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()

        size_pred = soft_size(predicted_mask[:, self.idc, ...].type(torch.float32))
        size_gt = soft_size(target[:, self.idc, ...].type(torch.float32))
        bool_size = (size_pred > 10).type(torch.float32)
        bool_gt_size = (size_gt > 1).type(torch.float32)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->",
                                                                                            [est_prop, log_est_prop])
        loss_cons_prior /= b
        loss_cons_prior *= self.temp

        if self.__momfn__.__name__ == "class_dist_centroid":
            loss_moment = self.__momfn__(probs.type(torch.float32))
            loss_moment = einsum("bou->", loss_moment)
        else:
            probs_moment = self.__momfn__(probs[:, self.idc, ...].type(torch.float32))
            # for numerical stability, only keep probs_moment if there is a predicted structure
            probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_size])
            # add the tag
            probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_gt_size])

            if probs_moment.shape[2] == 2:  # centroid and dist2centroid

                binary_est_gt_moment_w = torch.FloatTensor(self.mom_est[0]).expand(b, c).unsqueeze(2)
                binary_est_gt_moment_h = torch.FloatTensor(self.mom_est[1]).expand(b, c).unsqueeze(2)
                binary_est_gt_moment = torch.cat((binary_est_gt_moment_w, binary_est_gt_moment_h), 2)
                binary_est_gt_moment = binary_est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
                est_gt_moment = binary_est_gt_moment

            else:
                est_gt_moment = torch.FloatTensor(self.mom_est).unsqueeze(0).unsqueeze(2)
                est_gt_moment = est_gt_moment[:, self.idc, ...].to(loss_cons_prior.device)
            est_gt_moment = torch.einsum("bct,bco->bct", [est_gt_moment, bool_gt_size])

            upper_z = (est_gt_moment * (1 + self.margin) - probs_moment).flatten()
            lower_z = (probs_moment - est_gt_moment * (1 - self.margin)).flatten()
            upper_z = F.relu(upper_z) ** 2
            lower_z = F.relu(lower_z) ** 2
            loss_moment = upper_z + lower_z
            loss_moment = einsum("f->", loss_moment)

        # Adding division by batch_size to normalise
        loss_moment /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se * loss_se + self.lamb_consprior * loss_cons_prior, self.lamb_moment * loss_moment, est_prop


class EntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """

    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(utils, kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds, epc) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop: Tensor = self.__fn__(probs, self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:, :, 0]
                bounds = bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop) * bounds / (w * h)
            gt_prop = gt_prop[:, :, 0]
        else:
            gt_prop: Tensor = self.__fn__(target,
                                          self.power)  # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->",
                                                                                            [est_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10
        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se * loss_se, self.lamb_consprior * loss_cons_prior, est_prop


class ProposalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)
        predicted_mask = probs2one_hot(probs).detach()
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = predicted_mask[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss


class SelfEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss


class BCELoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.dtype = kwargs["dtype"]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, d_out: Tensor, label: float):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(d_out, Tensor(d_out.data.size()).fill_(label).to(d_out.device))
        return loss


class BCEGDice():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.lamb: List[int] = kwargs["lamb"]
        self.weights: List[float] = kwargs["weights"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss_gde = divided.mean()

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask_weighted = torch.einsum("bcwh,c->bcwh", [tc, Tensor(self.weights).to(tc.device)])
        loss_ce = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_ce /= tc.sum() + 1e-10
        loss = loss_ce + self.lamb * loss_gde

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))
        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)
        loss = divided.mean()

        return loss


class Weighted_self_entropy_loss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # weights for per-class losses
        self.weights: List[float] = kwargs["weights"]
        self.act = kwargs["act"]

    def self_entropy_loss(self, probs):
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss

    def max_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).sum(1)

    def tent_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x * torch.log(x)).sum(1).mean()

    def __call__(self, pred: Tensor) -> Tensor:
        b, k, h, w = pred.shape
        #### self-entropy loss
        if self.act == 'softmax':
            pred = pred.softmax(1)
            loss_entropy = self.tent_entropy(pred)
        else:
            pred = pred.sigmoid()
            ############ splitting type1
            # print('******splitting type1********')
            # union = pred.max(1)[0]
            # label_2 = (pred[:, 1] > 0.5).int()
            # pred_2 = label_2 * union
            # pred_1 = union - pred_2
            # pred = torch.stack([pred_1, pred_2], dim=1)
            ############# splitting type2
            # print('******splitting type2********')
            # pred_1 = pred[:, 0] - pred[:, 1]
            # pred_1 = torch.where(pred_1 >= 0.5, pred_1, torch.zeros_like(pred_1).to(pred_1.device))
            # pred_2 = pred[:, 1]
            # pred = torch.stack([pred_1, pred_2], dim=1)
            # label_1 = (pred[:, 0] > 0.5).int() - (pred[:, 1] > 0.5).int()
            # pred_1 = label_1 * pred[:, 0]
            # pred_2 = pred[:, 1]
            # # # pred_1 = pred[:, 0] - pred[:, 1]
            # # # pred_2 = pred[:, 1]
            # # # pred_3 = torch.ones_like(pred[:, 0]) - torch.max(pred, dim=1)[0]

            loss_entropy = self.self_entropy_loss(pred)
            # loss_entropy = self.max_entropy(pred).mean()
        return loss_entropy


class KL_class_ratio_entropy_loss():
    def __init__(self, **kwargs):
        class_ratio_prior: List[float] = kwargs["class_ratio_prior"]
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # weights for per-class losses
        self.weights: List[float] = kwargs["weights"]
        self.act = kwargs["act"]
        self.K = len(class_ratio_prior)
        self.class_ratio_prior = torch.from_numpy(np.array(class_ratio_prior)).to(torch.float32)

    def self_entropy_loss(self, probs):
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).sum(1)

    def prop_prior_loss(self, est_prop, gt_prop):
        gt_prop = gt_prop.unsqueeze(0).repeat_interleave(est_prop.shape[0], dim=0)
        # print('dismatch?', est_prop.size(), gt_prop.size())
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->",
                                                                                            [est_prop, log_est_prop])
        return loss_cons_prior

    def __call__(self, pred: Tensor) -> Tensor:
        b, k, h, w = pred.shape
        #### self-entropy loss
        if self.act == 'softmax':
            pred = pred.softmax(1)
            loss_entropy = self.self_entropy_loss(pred)

        elif self.act == 'sigmoid_onelabel':
            pred = pred.sigmoid()
            loss_entropy = self.self_entropy_loss(pred)

        else:
            # pred = pred.sigmoid()
            # loss_entropy = self.entropy(pred).mean()
            pred = pred.sigmoid()
            union = pred.max(1)[0]
            label_2 = (pred[:, 1] > 0.5).int()
            pred_2 = label_2 * union
            pred_1 = union - pred_2
            pred = torch.stack([pred_1, pred_2], dim=1)

            # pred_1 = pred[:, 0] - pred[:, 1]
            # pred_2 = pred[:, 1]
            # # pred_1 = pred[:, 0] - pred[:, 1]
            # # pred_2 = pred[:, 1]
            # # pred_3 = torch.ones_like(pred[:, 0]) - torch.max(pred, dim=1)[0]
            # pred_softmax = torch.stack([pred_1, pred_2], dim=1)
            # pred_softmax = pred
            loss_entropy = self.self_entropy_loss(pred)

            print('****splitting***')

        assert k == self.K
        # pred = pred.sigmoid()
        batch_ratio = torch.zeros([b, len(self.idc)]).to(pred.device).to(torch.float32)
        for i, c in enumerate(self.idc):
            class_sum = pred[:, c].sum([-2, -1])
            batch_ratio[:, i] = class_sum / (h*w)
            # print(i, class_sum/(h*w))

        batch_ratio = batch_ratio.mean(0)
        # print(batch_ratio)
        loss_shape_order_0 = torch.kl_div(batch_ratio, self.class_ratio_prior.to(pred.device)).mean()

        # batch_ratio = batch_ratio
        # loss_shape_order_0 = self.prop_prior_loss(est_prop=batch_ratio, gt_prop=self.class_ratio_prior.to(pred.device))
        return loss_entropy + loss_shape_order_0


class Constrain_prior_w_self_entropy_loss():
    def __init__(self, **kwargs):
        self.act = kwargs["act"]
        # load the prior moments from source data
        self.__momfn__ = getattr(utils, kwargs['moment_fn'])
        class_ratio_prior: List[float] = kwargs["class_ratio_prior"]
        # centroid_prior: List[float] = kwargs["centroid_prior"]
        # centroid_dist_prior: List[float] = kwargs["centroid_dist_prior"]
        self.mom_est: List[float] = kwargs["mom_est"]
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # weights for per-class losses
        self.weights_se: List[float] = kwargs["weights_se"]
        self.K = len(class_ratio_prior)
        self.class_ratio_prior = torch.from_numpy(np.array(class_ratio_prior)).to(torch.float32)
        # self.centroid_prior = torch.from_numpy(np.array(centroid_prior)).to(torch.float32)
        # self.centroid_dist_prior = torch.from_numpy(np.array(centroid_dist_prior)).to(torch.float32)
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.lamb_moment: float = kwargs["lamb_moment"]
        self.margin: float = kwargs["margin"]
        self.temp: float = kwargs["temp"]
        self.contain_fg = True

    def self_entropy_loss(self, probs):
        # self.weights = [5, 1]
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights_se).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).sum(1)

    def centroid_loss(self, probs):
        b, c, h, w = probs.shape
        # size_pred = soft_size(pred_mask[:, self.idc, ...].type(torch.float32))
        # bool_size = (size_pred > 10).type(torch.float32)
        if self.__momfn__.__name__ == "class_dist_centroid":
            loss_moment = self.__momfn__(probs.type(torch.float32))
            loss_moment = einsum("bou->", loss_moment)
        else:
            if self.contain_fg:
                print('**********moment_fn-contain fg*********')
                
                probs_fg = probs.sum([1, 2, 3])
                contain_fg_probs = probs[probs_fg > 0]
                probs_moment = self.__momfn__(contain_fg_probs[:, self.idc, ...].type(torch.float32))
            else:
                probs_moment = self.__momfn__(probs[:, self.idc, ...].type(torch.float32))
            # # for numerical stability, only keep probs_moment if there is a predicted structure
            # probs_moment = torch.einsum("bct,bco->bct", [probs_moment, bool_size])

            if probs_moment.shape[-1] == 2:  # centroid and dist2centroid
                if len(self.idc) ==1:
                    # only for one class
                    binary_est_gt_moment_w = torch.FloatTensor([self.mom_est[0]]).unsqueeze(0).expand(b, c).unsqueeze(-1)
                    binary_est_gt_moment_h = torch.FloatTensor([self.mom_est[1]]).unsqueeze(0).expand(b, c).unsqueeze(-1)
                    binary_est_gt_moment = torch.cat((binary_est_gt_moment_w, binary_est_gt_moment_h), 2)
                    binary_est_gt_moment = binary_est_gt_moment[:, self.idc, ...].to(probs.device)
                    est_gt_moment = binary_est_gt_moment
                else:
                    binary_est_gt_moment_w = torch.FloatTensor(self.mom_est[0]).expand(b, c).unsqueeze(-1)
                    binary_est_gt_moment_h = torch.FloatTensor(self.mom_est[1]).expand(b, c).unsqueeze(-1)
                    binary_est_gt_moment = torch.cat((binary_est_gt_moment_w, binary_est_gt_moment_h), 2)
                    binary_est_gt_moment = binary_est_gt_moment[:, self.idc, ...].to(probs.device)
                    est_gt_moment = binary_est_gt_moment
            else:
                est_gt_moment = torch.FloatTensor(self.mom_est).unsqueeze(0).unsqueeze(2)
                est_gt_moment = est_gt_moment[:, self.idc, ...].to(probs.device)
            # est_gt_moment = torch.einsum("bct,bco->bct", [est_gt_moment, bool_gt_size])

            upper_z = (est_gt_moment * (1 + self.margin) - probs_moment).flatten()
            lower_z = (probs_moment - est_gt_moment * (1 - self.margin)).flatten()
            upper_z = F.relu(upper_z) ** 2
            lower_z = F.relu(lower_z) ** 2
            loss_moment = upper_z + lower_z
            loss_moment = einsum("f->", loss_moment)
        # Adding division by batch_size to normalise
        loss_moment /= b
        return loss_moment

    def __call__(self, pred: Tensor) -> Tensor:
        b, c, h, w = pred.shape
        # #### self-entropy loss
        if self.act == 'softmax':
            pred_softmax = pred.softmax(1)
            loss_entropy = self.self_entropy_loss(pred_softmax)

        elif self.act == 'sigmoid_onelabel':
            pred = pred.sigmoid()
            loss_entropy = self.self_entropy_loss(pred)
        else:
            pred = pred.sigmoid()
            # loss_entropy = self.self_entropy_loss(pred)
            # pred = pred.sigmoid()
            # union = pred.max(1)[0]
            # label_2 = (pred[:, 1] > 0.5).int()
            # pred_2 = label_2 * union
            # pred_1 = union - pred_2
            # pred = torch.stack([pred_1, pred_2], dim=1)
            loss_entropy = self.self_entropy_loss(pred)
            # print('***max***')
            # loss_entropy = self.entropy(pred).mean()
        assert c == self.K
        batch_ratio = torch.zeros([b, len(self.idc)]).to(pred.device).to(torch.float32)
        for i, k in enumerate(self.idc):
            class_sum = pred[:, k].sum([-2, -1])
            batch_ratio[:, i] = class_sum/(h*w)
            # print(i, class_sum/(h*w))

        batch_ratio = batch_ratio.mean(0)
        # print(batch_ratio)
        loss_prop = torch.kl_div(batch_ratio, self.class_ratio_prior.to(pred.device)).mean()

        loss_shape_moment = self.centroid_loss(pred)
        # print('sub-loss', loss_entropy, loss_prop,loss_shape_moment  )
        return self.lamb_se*loss_entropy + self.lamb_consprior*loss_prop + self.lamb_moment*loss_shape_moment


class RN_w_CR_loss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.act = kwargs["act"]
        self.k = kwargs["k"]
        self.d = kwargs["d"]
        self.tag = kwargs["tag"]
        self.alpha = kwargs["alpha"]
        if self.tag == '2d':
            self.M_k = nn.MaxPool2d(self.k, stride=self.k)
            self.M_d = nn.MaxPool2d(2*self.d+1, stride=2*self.d+1)
        else:
            self.M_k = nn.MaxPool3d(self.k, stride=self.k)
            self.M_d = nn.MaxPool3d(2 * self.d + 1, stride=2 * self.d + 1)

    def RN_loss(self, pred):
        """
        Regional Nuclear-Norm Loss: https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24
        https://github.com/cuishuhao/BNM
        The input should be [N, C]
        """
        pred = self.M_k(pred)
        C = pred.shape[0]
        pred = pred.transpose(1, 0).reshape(C, -1).T
        L_BNM = -torch.norm(pred, 'nuc')
        return L_BNM

    def CR_loss(self, pred):
        """
        Contour Regularization Loss.: https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24
        """
        max = self.M_d(pred)
        min = self.M_d(-pred)
        C_d = max + min
        C_d = torch.norm(C_d, dim=1, p=2)
        L_CR = C_d.mean()
        return L_CR

    def __call__(self, pred: Tensor) -> Tensor:
        if self.act == 'softmax':
            pred = pred.softmax(1)
        else:
            pred = pred.sigmoid()

        rn_l = self.RN_loss(pred)
        cr_l = self.CR_loss(pred)
      
        return 0.001 * rn_l + cr_l
    

class CR_loss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.act = kwargs["act"]
        self.k = kwargs["k"]
        self.d = kwargs["d"]
        self.tag = kwargs["tag"]
        if self.tag == '2d':
            # self.M_k = nn.MaxPool2d(self.k, stride=self.k)
            self.M_d = nn.MaxPool2d(2*self.d+1, stride=2*self.d+1)
        else:
            # self.M_k = nn.MaxPool3d(self.k, stride=self.k)
            self.M_d = nn.MaxPool3d(2 * self.d + 1, stride=2 * self.d + 1)

    def CR_loss(self, pred):
        """
        Contour Regularization Loss.: https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24
        """
        max = self.M_d(pred)
        min = self.M_d(-pred)
        C_d = max + min
        # print('c_d', C_d.shape)
        C_d = C_d.reshape(C_d.shape[0], -1)
        C_d = torch.norm(C_d, dim=1, p=2)
        L_CR = C_d.mean()
        return L_CR

    def __call__(self, pred: Tensor) -> Tensor:
        if self.act == 'softmax':
            pred = pred.softmax(1)
        else:
            pred = pred.sigmoid()
        cr_l = self.CR_loss(pred)
       
        return cr_l


class Entropy_KLProp_Loss(nn.Module):

    """
    Simplify Implementaion of Entropy and KLProp Loss (MICCAI 2020)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Entropy_KLProp_Loss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.eps = 1e-6
         
    def forward(self, probs, gt_prop) -> torch.Tensor:
        # Update proportions
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        est_prop = probs.mean(dim=0, keepdim=True)
        log_gt_prop = (gt_prop + 1e-6).log()
        log_est_prop = (est_prop + 1e-6).log()
        
        
        # entropy_loss = torch.sum(-weights * probs * torch.log(probs + 1e-6), dim=0).mean()
        entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
        klprop_loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
        loss = self.s_par*entropy_loss + (1-self.s_par)*klprop_loss
        
        return loss
    

class EntropyLoss(nn.Module):
    def __init__(self, num_classes, weights=None, act='softmax'):
        super(EntropyLoss, self).__init__()
        self.act = act
        if weights is not None:
            self.weights = torch.tensor(weights).cuda()
        else:
            self.weights = torch.ones(num_classes).cuda()

    
    def forward(self, probs) -> torch.Tensor:
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        if self.act == 'softmax':
            probs = F.softmax(probs,dim=1)
            entropy = torch.sum(-probs * self.weights.view(1, -1) * torch.log(probs + 1e-6), dim=1).mean()
        else:
            probs = probs.sigmoid().max(dim=1)[0]
            entropy = torch.mean(-probs * torch.log(probs + 1e-6))
        return entropy
    
    
class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):
        avg_p = probs.mean(dim=[2, 3]) # avg along the pixels dim h x w -> size is [batch, n_classes]
        entropy_cm = torch.sum(avg_p * torch.log(avg_p + 1e-6), dim=1).mean()
        return entropy_cm


class Weighted_self_entropy_loss_v2():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # weights for per-class losses
        self.weights: List[float] = kwargs["weights"]
        self.act = kwargs["act"]

    def self_entropy_loss(self, probs):
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->bchw", [mask_weighted, log_p])
        loss = loss.mean()
        return loss

    def max_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).sum(1)

    def tent_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x * torch.log(x+1e-6)).sum(1).mean()
    
    def dltta_entropy_loss(self, p, c=2):
        # p N*C*W*H
        assert p.dim() == 4
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
        ent = torch.mean(y1)
        return ent

    def __call__(self, pred: Tensor) -> Tensor:
        b, k, h, w = pred.shape
        #### self-entropy loss
        if self.act == 'softmax':
            pred = pred.softmax(1)
            loss_entropy = self.tent_entropy(pred)
        else:
            pred = pred.sigmoid()
            if pred.shape[1] > 1:
                disc_prob = torch.cat([pred[:, 0].unsqueeze(1), 1 - pred[:, 0].unsqueeze(1)], dim=1)
                cup_prob = torch.cat([pred[:, 1].unsqueeze(1), 1 - pred[:, 1].unsqueeze(1)], dim=1)
                loss_entropy =  self.weights[0]*self.tent_entropy(disc_prob) +  self.weights[1]*self.tent_entropy(cup_prob)

            else: 
                loss_entropy = self.self_entropy_loss(pred)

        return loss_entropy


class KL_class_ratio_entropy_loss_v2():
    def __init__(self, **kwargs):
        class_ratio_prior: List[float] = kwargs["class_ratio_prior"]
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # weights for per-class losses
        self.weights: List[float] = kwargs["weights"]
        self.act = kwargs["act"]
        self.K = len(class_ratio_prior)
        self.class_ratio_prior = torch.from_numpy(np.array(class_ratio_prior)).to(torch.float32)

    def self_entropy_loss(self, probs):
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.max(1)[0] * torch.log(x.max(1)[0])).mean()
    

    def entropy_loss_sigmoid(self, p, c=2):
        # p N*C*H*W
        # p = p.max(dim=1)[0]
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
        ent = torch.mean(y1)
        return ent

    def prop_prior_loss(self, est_prop, gt_prop):
        gt_prop = gt_prop.unsqueeze(0).repeat_interleave(est_prop.shape[0], dim=0)
        # print('dismatch?', est_prop.size(), gt_prop.size())
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->",
                                                                                            [est_prop, log_est_prop])
        return loss_cons_prior

    def __call__(self, pred: Tensor) -> Tensor:
        b, k, h, w = pred.shape
        #### self-entropy loss
        if self.act == 'softmax':
            pred = pred.softmax(1)
            loss_entropy = self.self_entropy_loss(pred)

        elif self.act == 'sigmoid_onelabel':
            pred = pred.sigmoid()
            loss_entropy = self.self_entropy_loss(pred)

        else:
            pred = pred.sigmoid()
            # if pred.shape[1] > 1:
            #     disc_prob = torch.cat([pred[:, 0].unsqueeze(1), 1 - pred[:, 0].unsqueeze(1)], dim=1)
            #     cup_prob = torch.cat([pred[:, 1].unsqueeze(1), 1 - pred[:, 1].unsqueeze(1)], dim=1)
            #     loss_entropy = 1*self.self_entropy_loss(disc_prob) + 1*self.self_entropy_loss(cup_prob)
            # else:
            loss_entropy = self.self_entropy_loss(pred)

        assert k == self.K
        # pred = pred.sigmoid()
        batch_ratio = torch.zeros([b, len(self.idc)]).to(pred.device).to(torch.float32)
        for i, c in enumerate(self.idc):
            class_sum = pred[:, c].sum([-2, -1])
            batch_ratio[:, i] = class_sum / (h*w)
            # print(i, class_sum/(h*w))

        batch_ratio = batch_ratio.mean(0)
        loss_shape_order_0 = torch.kl_div(batch_ratio, self.class_ratio_prior.to(pred.device)).mean()


        return loss_entropy + loss_shape_order_0


# def tent_entropy(x: torch.Tensor) -> torch.Tensor:
#     return -(x * torch.log(x+1e-6)).sum(1).mean()

# def Weighted_sigmoid_entropy_loss_v2(pred, weights=[5, 1]):
#     pred = pred.sigmoid()
#     if pred.shape[1] > 1:
#         disc_prob = torch.cat([pred[:, 0].unsqueeze(1), 1 - pred[:, 0].unsqueeze(1)], dim=1)
#         cup_prob = torch.cat([pred[:, 1].unsqueeze(1), 1 - pred[:, 1].unsqueeze(1)], dim=1)
#         loss_entropy =  weights[0]*tent_entropy(disc_prob) +  weights[1]*tent_entropy(cup_prob)
#     else: 
#         loss_entropy = tent_entropy(pred)
#     return loss_entropy
