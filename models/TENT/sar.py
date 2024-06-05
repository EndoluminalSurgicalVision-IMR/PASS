# """
# Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
# built upon on Tent code.
# """

# from copy import deepcopy

# import torch
# import torch.nn as nn
# import torch.jit
# import math
# import numpy as np


# class SAM(torch.optim.Optimizer):
#     """
#     from https://github.com/davda54/sam
#     """
#     def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups
#         self.defaults.update(self.base_optimizer.defaults)

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group["rho"] / (grad_norm + 1e-12)

#             for p in group["params"]:
#                 if p.grad is None: continue
#                 self.state[p]["old_p"] = p.data.clone()
#                 e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
#                 p.add_(e_w)  # climb to the local maximum "w + e(w)"

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
#                 p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

#         self.base_optimizer.step()  # do the actual "sharpness-aware" update

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def step(self, closure=None):
#         assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
#         closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

#         self.first_step(zero_grad=True)
#         closure()
#         self.second_step()

#     def _grad_norm(self):
#         shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
#         norm = torch.norm(
#                     torch.stack([
#                         ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
#                         for group in self.param_groups for p in group["params"]
#                         if p.grad is not None
#                     ]),
#                     p=2
#                )
#         return norm

#     def load_state_dict(self, state_dict):
#         super().load_state_dict(state_dict)
#         self.base_optimizer.param_groups = self.param_groups


# def update_ema(ema, new_data):
#     if ema is None:
#         return new_data
#     else:
#         with torch.no_grad():
#             return 0.9 * ema + (1 - 0.9) * new_data


# class SAR(nn.Module):
#     """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
#     Once SARed, a model adapts itself by updating on every forward.
#     """
#     def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2, act='sigmoid', task='seg'):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.steps = steps
#         assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
#         self.episodic = episodic

#         self.margin_e0 =  0.004  # margin E_0 for reliable entropy minimization, Eqn. (2)
#         self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
#         self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

#         # note: if the model is never reset, like for continual adaptation,
#         # then skipping the state copy would save memory
#         self.model_state, self.optimizer_state = \
#             copy_model_and_optimizer(self.model, self.optimizer)

#     def forward(self, x, reset=False):
#         if self.episodic or reset:
#             self.reset()

#         for _ in range(self.steps):
#             self.model.train()
#             outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
#             if reset_flag:
#                 self.reset()
#             self.ema = ema  # update moving average value of loss
     
#         self.model.eval()
#         outputs = self.model(x)

#         return outputs

#     def reset(self):
#         if self.model_state is None or self.optimizer_state is None:
#             raise Exception("cannot reset without saved model/optimizer state")
#         load_model_and_optimizer(self.model, self.optimizer,
#                                  self.model_state, self.optimizer_state)
#         self.ema = None


# @torch.jit.script
# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# # @torch.jit.script
# # def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
# #     """Entropy of sigmoid distribution from logits."""
# #     pred = torch.sigmoid(x).max(dim=1)[0]
# #     pl = (pred > 0.5).detach().float()
# #     entropy =  -1 * torch.mean(pl * pred * torch.log(pred + 1e-6), dim=[-1, -2])
# #     #masked_entropy = (0.5*entropy.sum(1) * pl)
# #     # print('masked_entropy', masked_entropy.shape)
# #     # 只对p1进行entropy计算
# #     return entropy#(emasked_entropy.sum([-1, -2])) / (pl.sum([-1, -2]) + 1e-5)
# #     # return -(x.sigmoid() * torch.log(x.sigmoid()+1e-5)).sum(1).mean([-1, -2])

# @torch.jit.script
# def sigmoid_entropy(p: torch.Tensor) -> torch.Tensor:
#     # p N*C*W*H*D
#      p = torch.sigmoid(p).max(dim=1)[0]
#      ent = -1 * torch.mean(p * torch.log(p + 1e-6), dim=[-1, -2]) #/ torch.tensor(np.log(2)).cuda()
#      # ent = torch.mean(y1)
#      return ent/0.6931

# @torch.enable_grad()  # ensure grads in possible no grad context for testing
# def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
#     """Forward and adapt model input data.
#     Measure entropy of the model prediction, take gradients, and update params.
#     """
#     optimizer.zero_grad()
#     # forward
#     outputs = model(x)
#     # adapt
#     # filtering reliable samples/gradients for further adaptation; first time forward
#     entropys = sigmoid_entropy(outputs)

#     filter_ids_1 = torch.where(entropys < margin)
#     print('ids', entropys, filter_ids_1)
#     entropys = entropys[filter_ids_1]
#     loss = entropys.mean(0)
#     loss.backward()

#     optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
#     entropys2 = sigmoid_entropy(model(x))
#     entropys2 = entropys2[filter_ids_1]  # second time forward  
#     loss_second_value = entropys2.clone().detach().mean(0)
#     filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
#     loss_second = entropys2[filter_ids_2].mean(0)
#     if not np.isnan(loss_second.item()):
#         ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

#     # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
#     loss_second.backward()
#     optimizer.second_step(zero_grad=True)

#     # perform model recovery
#     reset_flag = False
#     if ema is not None:
#         if ema < 0.2:
#             print("ema < 0.2, now reset the model")
#             reset_flag = True

#     return outputs, ema, reset_flag


# def collect_params(model):
#     """Collect the affine scale + shift parameters from norm layers.
#     Walk the model's modules and collect all normalization parameters.
#     Return the parameters and their names.
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         # # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
#         # if 'layer4' in nm:
#         #     continue
#         # if 'blocks.9' in nm:
#         #     continue
#         # if 'blocks.10' in nm:
#         #     continue
#         # if 'blocks.11' in nm:
#         #     continue
#         # if 'norm.' in nm:
#         #     continue
#         # if nm in ['norm']:
#         #     continue

#         if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")

#     return params, names


# def copy_model_and_optimizer(model, optimizer):
#     """Copy the model and optimizer states for resetting after adaptation."""
#     model_state = deepcopy(model.state_dict())
#     optimizer_state = deepcopy(optimizer.state_dict())
#     return model_state, optimizer_state


# def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
#     """Restore the model and optimizer states from copies."""
#     model.load_state_dict(model_state, strict=True)
#     optimizer.load_state_dict(optimizer_state)


# def configure_model(model):
#     """Configure model for use with SAR."""
#     # train mode, because SAR optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what SAR updates
#     model.requires_grad_(False)
#     # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             # m.track_running_stats = False
#             # m.running_mean = None
#             # m.running_var = None
#         # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
#         if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
#             m.requires_grad_(True)
#     return model


# def check_model(model):
#     """Check model for compatability with SAR."""
#     is_training = model.training
#     assert is_training, "SAR needs train mode: call model.train()"
#     param_grads = [p.requires_grad for p in model.parameters()]
#     has_any_params = any(param_grads)
#     has_all_params = all(param_grads)
#     assert has_any_params, "SAR needs params to update: " \
#                            "check which require grad"
#     assert not has_all_params, "SAR should not update all params: " \
#                                "check which require grad"
#     has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
#     assert has_norm, "SAR needs normalization layer parameters for its optimization"

# #######注释部分用于RIGA的SAR
# """
# Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
# built upon on Tent code.
# """

# from copy import deepcopy

# import torch
# import torch.nn as nn
# import torch.jit
# import math
# import numpy as np


# class SAM(torch.optim.Optimizer):
#     """
#     from https://github.com/davda54/sam
#     """
#     def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups
#         self.defaults.update(self.base_optimizer.defaults)

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group["rho"] / (grad_norm + 1e-12)

#             for p in group["params"]:
#                 if p.grad is None: continue
#                 self.state[p]["old_p"] = p.data.clone()
#                 e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
#                 p.add_(e_w)  # climb to the local maximum "w + e(w)"

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
#                 p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

#         self.base_optimizer.step()  # do the actual "sharpness-aware" update

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def step(self, closure=None):
#         assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
#         closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

#         self.first_step(zero_grad=True)
#         closure()
#         self.second_step()

#     def _grad_norm(self):
#         shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
#         norm = torch.norm(
#                     torch.stack([
#                         ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
#                         for group in self.param_groups for p in group["params"]
#                         if p.grad is not None
#                     ]),
#                     p=2
#                )
#         return norm

#     def load_state_dict(self, state_dict):
#         super().load_state_dict(state_dict)
#         self.base_optimizer.param_groups = self.param_groups


# def update_ema(ema, new_data):
#     if ema is None:
#         return new_data
#     else:
#         with torch.no_grad():
#             return 0.9 * ema + (1 - 0.9) * new_data


# class SAR(nn.Module):
#     """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
#     Once SARed, a model adapts itself by updating on every forward.
#     """
#     def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2, act='softmax', task='seg'):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.steps = 10 #steps
#         assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
#         self.episodic = episodic

#         self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
#         self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
#         self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

#         # note: if the model is never reset, like for continual adaptation,
#         # then skipping the state copy would save memory
#         self.model_state, self.optimizer_state = \
#             copy_model_and_optimizer(self.model, self.optimizer)

#     def forward(self, x):
#         # if self.episodic:
#         #     self.reset()

#         for _ in range(self.steps):
#             outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
#             # if reset_flag:
#             #     self.reset()
#             # self.ema = ema  # update moving average value of loss

#         return outputs

#     def reset(self):
#         if self.model_state is None or self.optimizer_state is None:
#             raise Exception("cannot reset without saved model/optimizer state")
#         load_model_and_optimizer(self.model, self.optimizer,
#                                  self.model_state, self.optimizer_state)
#         self.ema = None


# @torch.jit.script
# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# @torch.jit.script
# def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     pl = (x.sigmoid() > 0.5).detach().float()
#     entropy = -(x.sigmoid() * torch.log(x.sigmoid()+1e-5))
#     # 只对p1进行entropy计算
#     return ((entropy * pl).mean(1).sum([-2, -1])) / (pl.sum([-3, -2, -1]) + 1e-5)
#     # return -(x.sigmoid() * torch.log(x.sigmoid()+1e-5)).sum(1).mean([-1, -2])


# @torch.enable_grad()  # ensure grads in possible no grad context for testing
# def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
#     """Forward and adapt model input data.
#     Measure entropy of the model prediction, take gradients, and update params.
#     """
#     optimizer.zero_grad()
#     # forward
#     outputs = model(x)
#     # adapt
#     # filtering reliable samples/gradients for further adaptation; first time forward
#     entropys = sigmoid_entropy(outputs)
#     filter_ids_1 = torch.where(entropys < margin)
#     # print('ids', entropys, filter_ids_1)
#     entropys = entropys[filter_ids_1]
#     loss = entropys.mean(0)
#     loss.backward()

#     optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
#     entropys2 = sigmoid_entropy(model(x))
#     entropys2 = entropys2[filter_ids_1]  # second time forward  
#     loss_second_value = entropys2.clone().detach().mean(0)
#     filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
#     loss_second = entropys2[filter_ids_2].mean(0)
#     if not np.isnan(loss_second.item()):
#         ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

#     # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
#     loss_second.backward()
#     optimizer.second_step(zero_grad=True)

#     # perform model recovery
#     reset_flag = False
#     # if ema is not None:
#     #     if ema < 0.2:
#     #         print("ema < 0.2, now reset the model")
#     #         reset_flag = True

#     return outputs, ema, reset_flag


# def collect_params(model):
#     """Collect the affine scale + shift parameters from norm layers.
#     Walk the model's modules and collect all normalization parameters.
#     Return the parameters and their names.
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
#         if 'layer4' in nm:
#             continue
#         if 'blocks.9' in nm:
#             continue
#         if 'blocks.10' in nm:
#             continue
#         if 'blocks.11' in nm:
#             continue
#         if 'norm.' in nm:
#             continue
#         if nm in ['norm']:
#             continue

#         if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")

#     return params, names


# def copy_model_and_optimizer(model, optimizer):
#     """Copy the model and optimizer states for resetting after adaptation."""
#     model_state = deepcopy(model.state_dict())
#     optimizer_state = deepcopy(optimizer.state_dict())
#     return model_state, optimizer_state


# def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
#     """Restore the model and optimizer states from copies."""
#     model.load_state_dict(model_state, strict=True)
#     optimizer.load_state_dict(optimizer_state)


# def configure_model(model):
#     """Configure model for use with SAR."""
#     # train mode, because SAR optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what SAR updates
#     model.requires_grad_(False)
#     # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
#         if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
#             m.requires_grad_(True)
#     return model


# def check_model(model):
#     """Check model for compatability with SAR."""
#     is_training = model.training
#     assert is_training, "SAR needs train mode: call model.train()"
#     param_grads = [p.requires_grad for p in model.parameters()]
#     has_any_params = any(param_grads)
#     has_all_params = all(param_grads)
#     assert has_any_params, "SAR needs params to update: " \
#                            "check which require grad"
#     assert not has_all_params, "SAR should not update all params: " \
#                                "check which require grad"
#     has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
#     assert has_norm, "SAR needs normalization layer parameters for its optimization"


"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0 # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, reset=False):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.model.train()
            outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        #self.model.eval()
        #outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    bz = x.shape[0]
    k = x.shape[1]
    x = x.sigmoid().max(dim=1)[0].view(bz, -1)
    return -(x *torch.log(x + 1e-6)).sum(1) / 200#.mean([-1, -2])

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = sigmoid_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    # print(entropys, filter_ids_1)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    entropys2 = sigmoid_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    print('ema', ema)
    reset_flag = False
    if ema is not None:
        if ema < 1:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        # if 'layer4' in nm:
        #     continue
        # if 'blocks.9' in nm:
        #     continue
        # if 'blocks.10' in nm:
        #     continue
        # if 'blocks.11' in nm:
        #     continue
        # if 'norm.' in nm:
        #     continue
        # if nm in ['norm']:
        #     continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

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
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"