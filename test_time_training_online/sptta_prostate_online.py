"""
Shape-Prompt-Learning: The proposed SP-TTA for MRI-Prostate dataset in online steup.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from batchgenerators.utilities.file_and_folder_operations import *
from models import *
from utils.metrics.dice import  get_dice_assd_hd, get_dice_threshold
from utils.tools import AverageMeter, save_np2nii, save_tensor2nii
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F
import models.moment_tta.losses as moment_tta_losses
from models.moment_tta.bounds import *
from loss_functions.bn_loss import bn_loss
from models.PromptTTA.ours_ema import *


def entropy_loss_sigmoid(p):
    # p N*C*W*H*D
     p = torch.sigmoid(p).max(dim=1)[0]
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0)  #/ torch.tensor(np.log(c)).cuda()
     ent = torch.mean(y1)
     return ent


class Outer_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Outer_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0

    def do_prostate_seg_tta_per_case(self, data):
        """
        tta for one case:BATCH 2d slices
         """
        num_samples = data.shape[0]
        num_batches = num_samples //self.args.batch_size
        if not self.args.model_episodic:
            self.model.reset_online_network()
        for step in range(0, self.args.optim_steps):
            self.model.online_network.train()
            train_loss_list = list()
            preds = []
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]

                self.model.online_network.train() 
                output = self.model.online_network(batch_data)           
                loss_entropy_before = entropy_loss_sigmoid(output)
                all_loss = loss_entropy_before
                train_loss_list.append(all_loss.item())
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step() 
                self.model.update_target_network()

                self.source_model.eval()
                output_wo_ada = self.source_model(batch_data)

                self.model.online_network.eval()
                output = self.model.online_network(batch_data)
                if entropy_loss_sigmoid(output) <= entropy_loss_sigmoid(output_wo_ada):
                    preds.append(output)
                else:
                    preds.append(output_wo_ada)
    
                train_loss_list.append(all_loss.detach().cpu().numpy())

            self.all_steps += 1
            self.logger.info('Step {}:'.format(step))
            self.logger.info('  loss:{}'.format(np.mean(train_loss_list)))
            self.writer.add_scalar("Train Scalars/Train Loss", np.mean(train_loss_list), self.all_steps)
            

        self.model.online_network.eval()
        preds_after_one_case = [] 
        with torch.no_grad(): 
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]
                preds_after_one_case.append(self.model.online_network(batch_data))
                
        full_pred = torch.cat(preds, dim=0)
        full_pred = torch.sigmoid(full_pred)
        full_pred = (full_pred > 0.5).float()

        full_pred_final = torch.cat(preds_after_one_case, dim=0)
        full_pred_final = torch.sigmoid(full_pred_final)
        full_pred_final = (full_pred_final > 0.5).float()
        return full_pred, full_pred_final


    def evaluate_prostate_seg(self):
        for epoch in range(1):
            case_name_all = []
            Dice_Metrics = AverageMeter(save_all=True)
            ASSD_Metrics = AverageMeter(save_all=True)
            HD_Metrics = AverageMeter(save_all=True)

            Dice_Metrics_final = AverageMeter(save_all=True)
            ASSD_Metrics_final = AverageMeter(save_all=True)
            HD_Metrics_final = AverageMeter(save_all=True)

            for iter, batch in enumerate(self.ts_dataloader):
                data, seg, case_name = batch
                data = data[0].to(self.device)
                seg = seg[0].to(self.device)
             
                ##################### perform tta per case ################
                start = time()
                pred_aggregated, pred_aggregated_final = self.do_prostate_seg_tta_per_case(data.unsqueeze(1).repeat(1, 3, 1, 1))
                end = time()
                case_name_all.append(case_name[0])
                self.logger.info('  Case {}  TTA time:{}'.format(case_name[0], end-start))
                ########### compute the tta-pred metrics ##################
                pred_array = pred_aggregated.detach().squeeze().cpu().numpy()
                pred_array_final = pred_aggregated_final.detach().squeeze().cpu().numpy()

                target_array = seg.squeeze().cpu().numpy()
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
                dc_final,assd_final, hd_final = get_dice_assd_hd(target_array, pred_array_final, self.logger)

                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)

                Dice_Metrics_final.update(dc_final)
                ASSD_Metrics_final.update(assd_final)
                HD_Metrics_final.update(hd_final)

              
                self.logger.info(
                "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name[0], dc, assd, hd))
                self.logger.info(
                "Cur-patient {}  dice_final:{:.3f} assd_final:{:.3f} hd_final:{:.3f}".format(case_name[0], dc_final, assd_final, hd_final))
                        
            avg_dc = Dice_Metrics.avg
            avg_hd = HD_Metrics.avg
            avg_assd = ASSD_Metrics.avg

            std_dc = np.std(Dice_Metrics.all_data)
            std_hd = np.std(HD_Metrics.all_data)
            std_assd = np.std(ASSD_Metrics.all_data)

            avg_dc_final = Dice_Metrics_final.avg
            avg_hd_final = HD_Metrics_final.avg
            avg_assd_final = ASSD_Metrics_final.avg

            std_dc_final = np.std(Dice_Metrics_final.all_data)
            std_hd_final = np.std(HD_Metrics_final.all_data)
            std_assd_final = np.std(ASSD_Metrics_final.all_data)

            self.logger.info(
                    " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
            self.logger.info(
                " Test Avg- dice_final:{:.4f}+{:.4f} hd_final:{:.4f}+{:.4f} assd_final:{:.4f}+{:.4f}".format(avg_dc_final,std_dc_final, avg_hd_final,std_hd_final, avg_assd_final, std_assd_final))
            
        
    def evaluate(self):
        if self.dataset_name == 'Prostate':
            self.evaluate_prostate_seg()
        else:
            raise NotImplementedError
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-EMA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA', 'Tent_EMA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta", required=False,
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
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='../medical_TTA/data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['../medical_TTA/data/RIGAPlus/MESSIDOR_Base2_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=1, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    parser.add_argument('--save_pre', required=False,default=False,
                        help='Whether to save the predicted mask.')

    ### ema model
    parser.add_argument('--ema_decay', type=float, default=0.6, required=False,
                        help='ema decay.')
    parser.add_argument('--min_momentum_constant', type=float, default=0.005, required=False,help='min momentum constant.')


    args = parser.parse_args()
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config,Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2UCL_config)
    args.tag = args.tag +'_sptta'
    adpater = Outer_Adapter(args)
    adpater.evaluate()
