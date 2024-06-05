"""
Official implementation for VPTTA on the MRI-Prostate dataset.
"""
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
import time
from loss_functions.warm_up_bn_loss import warm_up_bn_loss
import pandas as pd
from models.Official_VPTTA.ResUnet_VPTTA import ResUnet
from models.PromptTTA.vptta import UNet_BN
from models.Official_VPTTA.vptta import *


class VPTTA(BaseAdapter):
    def __init__(self, args):
        super(VPTTA, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        
        # Model
        self.backbone = args.backbone
        self.in_ch = args.in_ch
        self.out_ch = args.out_ch

        # Warm-up
        self.warm_n = args.warm_n

        # Prompt
        self.prompt_alpha = args.prompt_alpha
        self.iters = args.iters

        # Memory Bank
        self.neighbor = args.neighbor
        self.memory_bank = Memory(size=args.memory_size, dimension=self.prompt.data_prompt.numel())

        # Print Information
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)


    def init_from_source_model(self):
        self.prompt = Prompt(prompt_alpha=self.args.prompt_alpha, image_size=self.args.patch_size[0]).to(self.device)
        self.pretrained_params = torch.load(self.args.pretrained_model)
        self.model = UNet_BN(pretrained_path=self.args.pretrained_model, patch_size=self.args.patch_size, resnet='resnet34', num_classes=self.args.out_ch).to(self.device)
        return self.model


    def setup_model_w_optimizer(self, base_model):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.args.lr,
                betas= (self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay
            )

        return base_model

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))


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

            self.model.eval()
            self.prompt.train()
            # self.model.change_BN_status(new_sample=True)

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(batch_data)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data)

            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_iter in range(self.iters):
                prompt_x, _ = self.prompt(batch_data)
                output, bn_f, _ = self.model(prompt_x, training=True)
                loss = warm_up_bn_loss(self.model, self.pretrained_params, bn_f=bn_f, index=i)
                train_loss_list.append(loss.item())

                # times, bn_loss = 0, 0
                # for nm, m in self.model.named_modules():
                #     if isinstance(m, AdaBN):
                #         bn_loss += m.bn_loss
                #         times += 1
                # loss = bn_loss / times
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.model.change_BN_status(new_sample=False)

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(batch_data)
                pred_logit = self.model(prompt_x)
            preds.append(pred_logit) 

           # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())


        self.logger.info('  loss:{}'.format(np.mean(train_loss_list)))
    
        
        self.model.eval()
        self.prompt.eval()
        preds_after_one_case = [] 
        with torch.no_grad(): 
            for i in range(num_batches + 1 if num_samples % self.args.batch_size != 0 else num_batches):
                batch_start = i * self.args.batch_size
                batch_end = min((i + 1) * self.args.batch_size, num_samples)
                batch_data = data[batch_start:batch_end, :, :, :]
                prompt_x, low_freq = self.prompt(batch_data)
                output = self.model(prompt_x)
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
 
    parser.add_argument('--tag', default="Base2_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
 
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base2_test.csv'], required=False, help='test csv file.')
 
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

     # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)
  
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

 
    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')



    args = parser.parse_args()
    from scripts.tta_configs import Base1_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config,Prostate_RUNMC2I2CVB_config 
    args.__dict__.update(Prostate_RUNMC2HK_config)
    args.tag = args.tag +'_vptta'
    args.batch_size = 1
    adpater = VPTTA(args)
    adpater.evaluate()
  


  
