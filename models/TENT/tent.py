"""
Tent ICLR 2021
Adapted from https://github.com/DequanWang/tent/blob/master/tent.py
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import numpy as np


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False, act='softmax'):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.act = act

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, reset=False):
        if self.episodic or reset:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, act=self.act)

            self.model.eval()
            outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of sigmoid distribution from logits."""
    return -(x.max(-1)[0] * torch.log(x.max(-1)[0])).sum(1)

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of sigmoid distribution from logits."""
    return -(x * torch.log(x+1e-6))

def entropy_loss_sigmoid(p):
    # p N*C*W*H*D
     p = torch.sigmoid(p).max(dim=1)[0]
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) # / torch.tensor(np.log(c)).cuda()
     ent = torch.mean(y1)
     return ent


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, act):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    model.train()
    outputs = model(x)
    if act == 'softmax':
        loss = softmax_entropy(outputs).mean()

    elif act == 'sigmoid':
        # pred = outputs.sigmoid()
        # loss = sigmoid_entropy(pred)
        # loss = loss.mean()
        loss = entropy_loss_sigmoid(outputs)

    else:
       # for RIGA
        # print('------for RIGA-------------')
        pred = outputs.sigmoid()
        pred_1 = pred[:, 0] - pred[:, 1]
        pred_2 = pred[:, 1]
        pred_3 = torch.ones_like(pred[:, 0]) - torch.max(pred, dim=1)[0]
        pred_softmax = torch.stack([pred_3, pred_1, pred_2], dim=1)
        #print(pred_softmax.sum(1))
        loss = entropy(pred_softmax).mean()
        # loss = softmax_entropy(pred_softmax).mean()
        # pred1 = torch.stack([1 - pred[:, 0],  pred[:, 0]], dim=1)
        # en1 = -(pred1.max(1)[0] * torch.log(pred1.max(1)[0])).sum(1)
        # pred2 = torch.stack([1 - pred[:, 1], pred[:, 1]], dim=1)
        # en2 = -(pred2.max(1)[0] * torch.log(pred2.max(1)[0])).sum(1)
        # loss = en1.mean() + en2.mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    
    for name, param in model.named_parameters():
        if name.find('prompt') != -1 or name.find('data') != -1 or name.find('eff') != -1:
            params.append(param)
            names.append(f"{name}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)



def configure_model(model):
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


### more BN variants

def configure_model_TEMA(model, eps, momentum):
    """Configure model for adaptation TEMA."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            m.reset_running_stats()
            # use ema-test statistics in forward
            m.eval()
    return model

def configure_model_TBR(model, eps, momentum):
    """Configure model for adaptation Test-time batch Renormalization (TBR)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            m.reset_running_stats()
            # use ema-test statistics in forward
            m.eval()
    return model