"""
Adapted from https://github.com/dazhangyu123/OCL
"""

import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath('tools'))
import torch.optim as optim
from copy import deepcopy
from test_time_training_offline.base_tta import BaseAdapter
from models import *
from utils.file_utils import *
from utils.metrics.dice import get_hard_dice
from tqdm import tqdm
from inference.inference_nets_3d.inference_prostate import *

#######################################################
###### Functions for OCL-TTA ##########################
######################################################

def modified_bn_forward(self, input, momentum=0.001):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    # self.running_mean = (1 - momentum) * self.running_mean + momentum * est_mean
    # self.running_var = (1 - momentum) * self.running_var + momentum * est_var
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    # print('inds', inds)
    return x[inds]


class Evaluater(BaseAdapter):
    def __init__(self, args):
        super(Evaluater, self).__init__(args)
        self.model.eval()
        if args.prior > 0.0:
            assert isinstance(args.prior, float) and args.prior <= 1 and args.prior >= 0, 'False prior exists.'
            nn.BatchNorm2d.prior = None
            nn.BatchNorm2d.forward = modified_bn_forward
            nn.BatchNorm2d.prior = args.prior

    
    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':
            self.model = UNet()
           
            self.model = UNet(num_classes=self.args.num_classes)
            self.model.load_state_dict(self.pretrained_params['model_state_dict'])
            self.model = self.model.to(self.device)
         
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info('normalization statistics updated')
            else:
                self.logger.info('WARNING all var updated')
        else:
            raise NotImplementedError
        

    def configure_model_2d(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # if self.args.dataset_name == 'RIGA':
                # m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None
        return self.model
    

    def main(self):
        if self.args.model == 'OCL-TTT':
            # validate
            self.TTT()
        elif self.args.model == 'OCL-TTT-OnTheFly':
            # validate
            self.TTT_OnTheFly()
        elif self.args.model == 'Baseline':
            self.validate()
        else:
            raise AssertionError("do not implement ttt method")

        self.writer.close()

 
    def TTT(self):
        self.logger.info('Test time training...')
        start_epoch = 0
        anchor = deepcopy(self.model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(),
                              lr=self.args.learning_rate, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)


        header = 'Adapt:'
        i = 0
        self.model.eval()
        for epoch in range(start_epoch, self.args.num_epochs):
            train_loss_list = list()
            train_dice_list = list()
            lr = self.optimizer.param_groups[0]['lr']
            for iter, batch in tqdm(enumerate(self.tr_dataloader)):
                x, seg = batch
                x = x.to(self.device)
                seg = seg.to(self.device)
                x_flip = flip(x, -1)
                # x: [B, 3, 384, 384], x_flip: [B, 3, 384, 384]
                input = torch.cat([x, x_flip], dim=0)
                # input: [2B, 3, 384, 384]
                output = self.model(input)
                # output: [2B, 1, 384, 384]
                output_s = torch.stack([output[0:x.shape[0]], flip(output[x.shape[0]:], -1)], dim=0).permute(1, 0, 2, 3, 4)
                # output_s: [2, B, 1, 384, 384] - > [B, 2, 1, 384, 384]
                b, naug, c, h, w = output_s.shape
                if c > 1:
                    output_s_norm = F.normalize(F.softmax(output_s, dim=-3), p=2, dim=-3)
                else:
                    output_s_norm = torch.sigmoid(output_s)
        
                output_s_ = output_s_norm.view(b, naug, c, -1)
                # random sampling
                output_s_sampled = output_s_[:, :, :, torch.randperm(output_s_.shape[-1])[:10000]]
                #print('output_s_sample', output_s_sampled.shape)
                pos_loss = -(torch.mul(output_s_norm[:, 0], output_s_norm[:, 1])).sum(0).mean()
                # neg_loss = (torch.einsum('cn,cm->nm', output_s_[0], output_s_[0]).mean() + torch.einsum('cn,cm->nm', output_s_[1], output_s_[1]).mean())/ naug
                # neg_loss = ((output_s_[0].T @ output_s_[0]).mean() + (output_s_[1].T @ output_s_[1]).mean()) / naug

                neg_loss = ((output_s_sampled[:, 0].permute(0, 2, 1) @ output_s_sampled[:, 0]).mean() + (output_s_sampled[:, 1].permute(0, 2, 1) @ output_s_sampled[:, 1]).mean()) / (b*naug)

                loss = pos_loss + neg_loss


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #print('pred s shape', output_s_norm.shape)
                # b, aug, c, h, w
                pred = output_s_norm.mean(1)
                # pred: [b, c, h, w]

                train_loss_list.append(loss.detach().cpu().numpy())
                # cal 2d dice
                train_dice_list.append(get_hard_dice(pred.cpu().squeeze(),seg.cpu().squeeze()))
                # del seg
            mean_tr_loss = np.mean(train_loss_list)
            self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
            self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
            self.logger.info(' Tr loss: {}'.format(mean_tr_loss))
            
            mean_tr_dice = np.mean(train_dice_list)
            self.writer.add_scalar("Train Scalars/Dice", mean_tr_dice, epoch)
            self.logger.info('  Tr-dice: {}'.format(mean_tr_dice))


            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.args.mask_ratio).float().to(self.device)
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)


        # inference
        if self.args.dataset_name == 'RIGA':
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                self.recorder.logger.info("Running inference: {}".format(inference_tag))
                inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                          inference_tag)
                
        elif self.args.dataset_name == 'Prostate':
            self.validate_prostate()

        else:
            raise NotImplemented


        return 


    @torch.no_grad()
    def validate_prostate(self):
    # Calculate the dice of different class, respectively.
        Dice_Metrics = AverageMeter(save_all=True)
        ASSD_Metrics = AverageMeter(save_all=True)
        HD_Metrics = AverageMeter(save_all=True)
        all_case_name = []

        self.model.eval()
        all_case_name = []
        test_batch = 1
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(self.ts_dataloader)):
                case_data, label, case_name = sample
                self.logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(self.device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([self.args.num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(test_batch, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) == 3:
                        # test_batch  = 1
                        slice = slice.unsqueeze(1)
                    # repeat 3 channels for ResNet input
                    x = slice.repeat(1, 3, 1, 1)
                    x_flip = flip(x, -1)
                    # print('x-shape', x.shape, x_flip.shape)
                    input = torch.cat([x, x_flip], dim=0)
                    output = self.model(input)
                    output_s = torch.stack([output[0:x.shape[0]], flip(output[x.shape[0]:], -1)], dim=0)
                    # print('output_s', output_s.shape)
                    output_s = output_s.mean(0)
                   
                    # pred_s: [B, K, H, W]
                    pred_aggregated[:, s:s_end, :, :] = torch.sigmoid(output_s).permute(1, 0, 2, 3)
                    s = s_end
                print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())

                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()

                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                self.logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated
             
                sys.stdout.flush()

        avg_dc = Dice_Metrics.avg
        avg_hd = HD_Metrics.avg
        avg_assd = ASSD_Metrics.avg

        std_dc = np.std(Dice_Metrics.all_data)
        std_hd = np.std(HD_Metrics.all_data)
        std_assd = np.std(ASSD_Metrics.all_data)

        self.logger.info(
                " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
        
        self.logger.info("Test finished!")


    def TTT_OnTheFly(self):
        self.logger.info('Test time training...')
        start_epoch = 0
        
        anchor = deepcopy(self.model.state_dict())
        self.optimizer = optim.SGD(self.model.parameters(),
                              lr=self.args.learning_rate, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)


        header = 'Adapt:'
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)
        i = 0
        Dice_Metrics = AverageMeter(save_all=True)
        ASSD_Metrics = AverageMeter(save_all=True)
        HD_Metrics = AverageMeter(save_all=True)
        all_case_name = []
        self.model.eval()

        for epoch in range(start_epoch, self.args.num_epochs):
            train_loss_list = list()
            lr = self.optimizer.param_groups[0]['lr']
            all_case_name = []
            for iter, sample in tqdm(enumerate(self.ts_dataloader)):
                case_data, label, case_name = sample
                self.logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(self.device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([self.args.num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(self.args.batch_size, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) == 3:
                        slice = slice.unsqueeze(1)
                    # repeat 3 channels for ResNet input
                    x = slice.repeat(1, 3, 1, 1)
                    x_flip = flip(x, -1)
                    input = torch.cat([x, x_flip], dim=0)
                    output = self.model(input)
                    output_s = torch.cat([output[0:x.shape[0]], flip(output[x.shape[0]:], -1)], dim=0)

                    naug, c, h, w = output_s.shape
                    if c > 1:
                        output_s_norm = F.normalize(F.softmax(output_s, dim=1), p=2, dim=1)
                    else:
                        output_s_norm = torch.sigmoid(output_s)
            
                    output_s_ = output_s_norm.view(naug, c, -1)

                    # random sampling
                    output_s_sampled = output_s_[:, :, torch.randperm(output_s_.shape[2])[:20000]]
                    pos_loss = -(torch.mul(output_s_norm[0], output_s_norm[1])).sum(0).mean()
                    neg_loss = ((output_s_sampled[0].T @ output_s_sampled[0]).mean() + (output_s_sampled[1].T @ output_s_sampled[1]).mean()) / naug
                    loss = self.args.pos_coeff * pos_loss + neg_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    output_s = output_s_norm.mean(0, keepdim=True).detach()
                    # pred_s: [B, K, H, W]
                    pred_aggregated[:, s:s_end, :, :] = (output_s).permute(1, 0, 2, 3)
                    s = s_end
                    train_loss_list.append(loss.detach().cpu().numpy())

                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()

                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                self.logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated
            
            mean_tr_loss = np.mean(train_loss_list)
            self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
            self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
            self.logger.info(' Tr loss: {}'.format(mean_tr_loss))


            avg_dc = Dice_Metrics.avg
            avg_hd = HD_Metrics.avg
            avg_assd = ASSD_Metrics.avg

            std_dc = np.std(Dice_Metrics.all_data)
            std_hd = np.std(HD_Metrics.all_data)
            std_assd = np.std(ASSD_Metrics.all_data)

            self.logger.info(
                    " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
            
            self.logger.info("Test finished!")
            
            
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.args.mask_ratio).float().to(self.device)
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        return 
     

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="OCL-TTT", required=False,choices=['OCL-TTT', 'OCL-TTT-OnTheFly'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base2_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                        help='batch size.')
    parser.add_argument('--optimizer', type=str, default='sgd', required=False,
                        help='optimizer method.')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['RIGAPlus/MESSIDOR_Base2_test.csv'],
                        required=False, help='test csv file.')
    # parser.add_argument('--optim_steps', type=int, default=1, required=False,
    #                     help='optimization steps.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='../log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--source_dataset', default='None', type=str,
                            help='source dataset choice')
    parser.add_argument('--city_name', default='None', type=str,
                            help='source dataset choice')
    parser.add_argument('--flip', action='store_true',help="flip")

    # evaluation methods setting
    parser.add_argument("--pos-coeff", type=float, default=3.0,
                        help='Positive loss coefficient')
    parser.add_argument("--mask-ratio", type=float, default=0.01,
                        help='masking ratio in the stochastic restoration')
    parser.add_argument("--prior", type=float, default=0.0, help=
                        "the hyperparameter determine the weight of training statistic")

    # optimizer
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="initial learning rate for the segmentation network.")
    args = parser.parse_args()
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config,Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2BIDMC_config)
    args.tag = args.tag +'_OCL_TTT'

    agent = Evaluater(args)
    agent.main()
