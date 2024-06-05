"""
Our implementation for VP-TTA on MRI-Prostate dataset.
<<Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation>>
"""
import sys
import os
from utils.init import init_random_and_cudnn, Recorder
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *
from utils.metrics.dice import  get_dice_assd_hd, get_dice_threshold
from utils.tools import AverageMeter, save_np2nii, save_tensor2nii
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F
import models.moment_tta.losses as moment_tta_losses
from models.moment_tta.bounds import *
import time
from models.PromptTTA.vptta import *
from loss_functions.warm_up_bn_loss import warm_up_bn_loss
import pandas as pd


class VPTT_Adapter(BaseAdapter):
    def __init__(self, args):
        super(VPTT_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0
        self.memory = Memory(S=64, K=16, image_shape=(3, self.args.patch_size[0], self.args.patch_size[1]))

    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(self.args.pretrained_model)
        base_model = UNet_VPTTA(device=self.device, pretrained_path=self.args.pretrained_model, patch_size=self.args.patch_size, num_classes=self.args.num_classes)
        return base_model
    
    def setup_model_w_optimizer(self, base_model):
        self.model = base_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.initial_lr)
        return self.model

    def do_prostate_seg_tta_per_case(self, data):
        """
        tta for one case:BATCH 2d slices
         """
        num_samples = data.shape[0]
        num_batches = num_samples //self.args.batch_size
        if self.args.model_episodic:
            base_model = self.init_from_source_model()
            self.model = self.setup_model_w_optimizer(base_model)
      
        train_loss_list = list()
        preds = []
        for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
            batch_start = i * self.args.batch_size
            batch_end = min((i + 1) * self.args.batch_size, num_samples)
            batch_data = data[batch_start:batch_end, :, :, :]

            amp_i, pha_i, low_freq_amp_i, low_freq_pha_i  = fft_tensor(batch_data)
            memory_size = self.memory.get_size()
            if memory_size >= self.memory.K:
                with torch.no_grad():
                    prompt_i = self.memory.get_neighbours(low_freq_amp_i.cpu().numpy())
                    prompt_i = torch.tensor(prompt_i).to(self.device).squeeze()
            else:
                prompt_i = None
            # Init the prompt
            self.model.init_freq_prompt(prompt_i)
            self.model.train()
          
            output, bn_f, img_ada_i = self.model(amp_i, pha_i, training=True)
            all_loss = warm_up_bn_loss(self.model, self.pretrained_params, bn_f, index=i)
            print('iter:{}, loss:{}'.format(i, all_loss.item()))
    
            train_loss_list.append(all_loss.item())
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step() 
        
            self.memory.push(low_freq_amp_i.cpu().numpy(), self.model.freq_prompt.data.unsqueeze(0).cpu().numpy())

            self.model.eval()
            output, img_ada_i, freq_prompt = self.model(amp_i, pha_i)
            preds.append(output) 

        self.all_steps += 1
        self.logger.info('Step {}:'.format(self.all_steps))
        self.logger.info('  loss:{}'.format(np.mean(train_loss_list)))
        self.writer.add_scalar("Train Scalars/Train Loss", np.mean(train_loss_list), self.all_steps)
        

        self.model.eval()
        preds_after_one_case = [] 
        with torch.no_grad(): 
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]
                amp_i, pha_i, low_freq_amp_i, low_freq_pha_i  = fft_tensor(batch_data)
                output, img_ada_i, freq_prompt = self.model(amp_i, pha_i)
                preds_after_one_case.append(output)
                
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
                start = time.time()
                pred_aggregated, pred_aggregated_final = self.do_prostate_seg_tta_per_case(data.unsqueeze(1).repeat(1, 3, 1, 1))
                end =  time.time()
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


            
            self.logger.info(
                    " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
            self.logger.info(
                " Test Avg- dice_final:{:.4f}+{:.4f} hd_final:{:.4f}+{:.4f} assd_final:{:.4f}+{:.4f}".format(avg_dc_final,std_dc_final, avg_hd_final,std_hd_final, avg_assd_final, std_assd_final))
            
            # save the results
            time_str = time.strftime("%Y%m%d%H%M", time.localtime())
            data_frame = pd.DataFrame(
            data={'Case': case_name_all,
                            'Dice': Dice_Metrics.all_data,
                            'HD': HD_Metrics.all_data,
                            'ASSD': ASSD_Metrics.all_data},
                    index=range(len(case_name_all)))
            data_frame.to_csv(self.metrics_folder + '/'+ time_str + '_results.csv',
                                    index_label='Index')
            
            data_frame_final = pd.DataFrame(
            data={'Case': case_name_all,
                            'Dice': Dice_Metrics_final.all_data,
                            'HD': HD_Metrics_final.all_data,
                            'ASSD': ASSD_Metrics_final.all_data}, index=range(len(case_name_all)))
            data_frame_final.to_csv(self.metrics_folder + '/'+ time_str + '_results_final.csv',
                                    index_label='Index')
            
        
    def evaluate(self):
        if self.dataset_name == 'Prostate':
            self.evaluate_prostate_seg()
        else:
            raise NotImplementedError
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-VP", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA', 'Tent_EMA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_vptta", required=False,
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
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base2_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=1, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=True, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    parser.add_argument('--save_pre', required=False,default=False,
                        help='Whether to save the predicted mask.')


    args = parser.parse_args()
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2HK_config,Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2HK_config)
    args.tag = args.tag +'_vptta'
    args.batch_size = 1
    adpater = VPTT_Adapter(args)
    adpater.evaluate()
  


  
