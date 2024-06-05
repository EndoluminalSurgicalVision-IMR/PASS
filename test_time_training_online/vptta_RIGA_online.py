"""
Our implementation for VPTTA on the RIGA+ dataset.
<<Each Test Image Deserves A Specific Prompt: Continual Test-Time Adaptation for 2D Medical Image Segmentation>>
"""
import sys
import os
import time
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
import torch.nn.functional as F
from models import *
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image
from models.PromptTTA.vptta import *
from loss_functions.warm_up_bn_loss import warm_up_bn_loss
import pandas as pd

class VPTT_Adapter(BaseAdapter):
    def __init__(self, args):
        super(VPTT_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0
        self.memory = Memory(S=30, K=8)


    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(self.args.pretrained_model)
        base_model = UNet_VPTTA(device=self.device, pretrained_path=self.args.pretrained_model, patch_size=self.args.patch_size, num_classes=self.args.num_classes)
        return base_model
    
    def setup_model_w_optimizer(self, base_model):
        self.model = base_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.initial_lr)
        return self.model

    def evaluate_dis_cup_seg(self):
        test_disc_dice_list = []
        test_cup_dice_list = []
        case_name_list = []
        for iter, batch in enumerate(self.ts_dataloader):
            if self.args.model_episodic:
                base_model = self.init_from_source_model()
                self.model = self.setup_model_w_optimizer(base_model)
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            name = batch['name'][0]
            amp_i, pha_i, low_freq_amp_i, low_freq_pha_i  = fft_tensor(data)
    
            memory_size = self.memory.get_size()
            if memory_size >= self.memory.K:
                with torch.no_grad():
                    prompt_i = self.memory.get_neighbours(low_freq_amp_i.cpu().numpy())
                    prompt_i = torch.tensor(prompt_i).to(self.device).squeeze()
            else:
                prompt_i = None
        
            self.model.init_freq_prompt(prompt_i)
            self.model.train()
          
            for step in range(self.args.optim_steps):
                self.all_steps += 1
                output, bn_f, img_ada_i = self.model(amp_i, pha_i, training=True)
                all_loss = warm_up_bn_loss(self.model, self.pretrained_params, bn_f, index=iter) 
                
                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
          
            self.memory.push(low_freq_amp_i.cpu().numpy(), self.model.freq_prompt.data.unsqueeze(0).cpu().numpy())
           
            self.model.eval()
            output, img_ada_i, freq_prompt = self.model(amp_i, pha_i) 
            output_sigmoid = torch.sigmoid(output)
            
            test_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            test_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0)) 
            case_name_list.append(name)

        mean_val_disc_dice = np.mean(test_disc_dice_list)
        mean_val_cup_dice = np.mean(test_cup_dice_list)
        std_val_disc_dice = np.std(test_disc_dice_list)
        std_val_cup_dice = np.std(test_cup_dice_list)
        self.logger.info(' Test disc dice: {:.5f}+{:.5f}; Cup dice: {:.5f}+{:.5f}'.format(mean_val_disc_dice, std_val_disc_dice, mean_val_cup_dice, std_val_cup_dice))
        
        saved_model = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }
        print('Saving model_{}.model...'.format('final'))
        torch.save(saved_model, join(self.model_folder, 'model_final.model'))

        # save the results
        time_str = time.strftime("%Y%m%d%H%M", time.localtime())
        data_frame = pd.DataFrame(
        data={'Case': case_name_list,
                        'Disc_Dice': test_disc_dice_list,
                        'Cup_Dice': test_cup_dice_list},
                index=range(len(case_name_list)))
        data_frame.to_csv(self.metrics_folder + '/'+ time_str + '_results.csv',
                                index_label='Index')
            
        return

    def evaluate(self):
        if self.dataset_name == 'RIGAPlus':
            self.evaluate_dis_cup_seg()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-TTA-VP", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_vptta", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=2, required=False,
                        help='Num of classes.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base3_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=5e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base3_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=1, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--save_pre', required=False,default=False,
                        help='Whether to save the predicted mask.')
     ### ema model

    args = parser.parse_args()
    adpater = VPTT_Adapter(args)
    adpater.evaluate()
