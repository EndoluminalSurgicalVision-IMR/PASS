"""
Shape-Prompt-Learning: The proposed SP-TTA for RIGA+ dataset in online setup.
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from time import time
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F
from models import *
from models.moment_tta.losses import *
from models.moment_tta.bounds import *
from loss_functions.bn_loss import bn_loss

def entropy_loss_sigmoid(pred):
    probs = pred.sigmoid()
    weights = torch.tensor([5, 1])
    log_p = (probs + 1e-10).log()
    mask = probs.type((torch.float32))
    mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, weights.to(mask.device)])
    loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
    loss /= mask.sum() + 1e-10
    return loss

class SPTTA_Online(BaseAdapter):
    def __init__(self, args):
        super(SPTTA_Online, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.all_steps = 0

    def evaluate_dis_cup_seg(self):
        
        for epoch in range(1):
            train_loss_list = list()
            test_disc_dice_list = []
            test_cup_dice_list = []
            for iter, batch in enumerate(self.ts_dataloader):
                self.model.train()
                data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                self.model.reset_online_network()
                self.source_model.eval()
                output_wo_ada = self.source_model(data)
                self.model.online_network.train()
                for step in range(self.args.optim_steps):
                    self.all_steps += 1
                    output, bn_f, _, _ = self.model.online_network(data, training=True)
                    loss_entropy_before = bn_loss(self.model.online_network, self.pretrained_params, bn_f)
                    all_loss = loss_entropy_before
                    print('step:{}, loss:{}'.format(step, all_loss.item()))
                    train_loss_list.append(all_loss.item())
                   
                    self.optimizer.zero_grad()
                    all_loss.backward()
                    self.optimizer.step() 
                    self.model.update_target_network()
                    
                self.model.online_network.eval()
                output = self.model.online_network(data) 
                output_entropy = entropy_loss_sigmoid(output) 
                output_wo_ada_entropy = entropy_loss_sigmoid(output_wo_ada)

                if output_entropy < output_wo_ada_entropy:
                    output_sigmoid = torch.sigmoid(output)
                else:
                    output_sigmoid = torch.sigmoid(output_wo_ada)

                test_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                test_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0)) 
                self.logger.info('Epoch: {}, Iter: {}, Loss: {:.5f}'.format(epoch, iter, all_loss.item()))  

            mean_val_disc_dice = np.mean(test_disc_dice_list)
            mean_val_cup_dice = np.mean(test_cup_dice_list)
            std_val_disc_dice = np.std(test_disc_dice_list)
            std_val_cup_dice = np.std(test_cup_dice_list)
            self.logger.info(' Test disc dice: {:.5f}+{:.5f}; Cup dice: {:.5f}+{:.5f}'.format(mean_val_disc_dice, std_val_disc_dice, mean_val_cup_dice, std_val_cup_dice))
        
        saved_model = {
                'target_model_state_dict': self.model.target_network.state_dict(),
                'model_state_dict': self.model.online_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }
        print('Saving model_{}.model...'.format('final'))
        torch.save(saved_model, join(self.model_folder, 'model_final.model'))
            
        
        pass

    def evaluate(self):
        self.evaluate_dis_cup_seg()
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-EMA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA'],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta", required=False,
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
    parser.add_argument('--initial_lr', type=float, default=5e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    
    parser.add_argument('-r', '--root', default='../Medical_TTA/data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['../Medical_TTA/data/RIGAPlus/MESSIDOR_Base3_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=2, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_RIGA/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the  model for each batch, choose True.')
    parser.add_argument('--save_pre', required=False,default=False,
                        help='Whether to save the predicted mask.')
     ### ema model
    parser.add_argument('--ema_decay', type=float, default=0.94, required=False,
                        help='ema decay.')
    parser.add_argument('--min_momentum_constant', type=float, default=0.01, required=False,help='min momentum constant.')

    args = parser.parse_args()
    adpater = SPTTA_Online(args)
    adpater.evaluate()
    
