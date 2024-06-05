"""
Official implementation for VPTTA on the RIGA+ dataset
"""
import torch
import numpy as np
import argparse, sys, datetime
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
from models.Official_VPTTA.ResUnet_VPTTA import ResUnet
from models.Official_VPTTA.vptta import *
from models.PromptTTA.vptta import UNet_BN
from loss_functions.warm_up_bn_loss import warm_up_bn_loss
import pandas as pd


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

    def run(self):
        test_disc_dice_list = []
        test_cup_dice_list = []
        case_name_list = []

        for iter, batch in enumerate(self.ts_dataloader):
            x = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            y = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            name = batch['name'][0]


            self.model.eval()
            self.prompt.train()
            # self.model.change_BN_status(new_sample=True)

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(x)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data)

            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_iter in range(self.iters):
                prompt_x, _ = self.prompt(x)
                output, bn_f, _ = self.model(prompt_x, training=True)
                loss = warm_up_bn_loss(self.model, self.pretrained_params, bn_f=bn_f, index=iter)

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
                prompt_x, low_freq = self.prompt(x)
                pred_logit = self.model(prompt_x)

            # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

            # Calculate the metrics
            output_sigmoid = torch.sigmoid(pred_logit)
            
            test_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (y[:, 0] > 0).cpu() * 1.0))
            test_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (y[:, 0] == 2).cpu() * 1.0)) 
            case_name_list.append(name)

        mean_val_disc_dice = np.mean(test_disc_dice_list)
        mean_val_cup_dice = np.mean(test_cup_dice_list)
        std_val_disc_dice = np.std(test_disc_dice_list)
        std_val_cup_dice = np.std(test_cup_dice_list)
        self.logger.info(' Test disc dice: {:.5f}+{:.5f}; Cup dice: {:.5f}+{:.5f}'.format(mean_val_disc_dice, std_val_disc_dice, mean_val_cup_dice, std_val_cup_dice))
        
           


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Tent-VPTTA", required=False,
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
    parser.add_argument('--tag', default="Base3_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base3_test.csv'], required=False, help='test csv file.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')

    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)
  
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

    config = parser.parse_args()

    TTA = VPTTA(config)
    TTA.run()