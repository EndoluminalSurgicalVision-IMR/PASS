"""
Shape-Prompt-Learning: The proposed SP-TTA for RIGA dataset in offline setup.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import argparse
from time import time
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models import *
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import target_collate_fn_tr_fda, collate_fn_ts,collate_fn_tr
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from torch.cuda.amp import autocast, GradScaler
from models.TENT.tent import configure_model
import models.moment_tta.losses as moment_tta_losses
import models.moment_tta.utils as moment_tta_utils
from models.moment_tta.bounds import *
from typing import Any, Callable, List, Tuple
from test_time_training_offline.base_tta import BaseAdapter
from loss_functions.bn_loss import bn_loss
from datasets.utils.normalize import normalize_image
import torch.nn.functional as F

#### Loss for adaptation

BN_sta = ['bn_loss',
         {'layers':5, 'alpha':0.01}]

TENT = ['Weighted_self_entropy_loss',
         {'weights':[1, 10], 'idc':[0, 1], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'2d'}]

LSIZE = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZECentroid = ['Constrain_prior_w_self_entropy_loss',
                 {'idc': [0, 1],
                  'weights_se':[0.8, 0.2],'lamb_se':1,
                  'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                  'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                  'mom_est':[[254.9747, 255.98499], [253.97656, 255.04263]],
                  'moment_fn':'soft_centroid', 'lamb_consprior':1,
                  'power': 1, 'act':'sigmoid'}]

LSIZEDistCentroid = ['Constrain_prior_w_self_entropy_loss',
                     {'idc':[0, 1],
                      'weights_se':[0.8, 0.2],'lamb_se':1,
                      'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                     'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                     'mom_est':[[39.157185, 37.18185 ],[17.430944,18.484102]],
                     'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,
                     'power': 1, 'act':'sigmoid'}]

class Offline_Adapter(BaseAdapter):
    def __init__(self, args):
        super(SPTTA_Offline, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.loss_fn = None
        if args.model == 'Moment-TTA':
            loss_name, loss_params = LSIZECentroid
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'Tent-TTA':
            loss_name, loss_params = TENT
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'RN-CR-TTA':
            loss_name, loss_params = RN_w_CR
            loss_class = getattr(moment_tta_losses, loss_name)
        elif args.model == 'BN-Sta-TTA':
            loss_name, loss_params = BN_sta
            loss_class = None
        else:
            loss_name, loss_params = LSIZE
            loss_class = getattr(moment_tta_losses, loss_name)

        self.logger.info('loss_name:{}'.format(loss_name))
        self.logger.info('Loss params:{}'.format(loss_params))
        if loss_class is not None:
            self.loss_fn = loss_class(**loss_params)

    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        if self.args.arch == 'unet_2d':
            self.model = UNet()
            self.model.load_state_dict(self.pretrained_params['model_state_dict'])
            self.model = self.model.to(self.device)
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info('normalization statistics updated')
            else:
                self.logger.info('WARNING all var updated')

            self.amp_grad_scaler = GradScaler()


        elif args.arch == 'unet_2d_sptta':
            self.model = UNet_SPTTA(pretrained_path=args.pretrained_model, num_classes=self.args.num_classes,  patch_size=self.args.patch_size)
            self.model = self.model.to(self.device)
            self.amp_grad_scaler = GradScaler()
            self.model = self.configure_model_2d()

        else:
            NotImplementedError('Not implemented optimizing step for this architecture')

        self.init_optimizer_and_scheduler()

    def configure_model_2d(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        # disable grad, to (re-)enable only what tent updates
        # self.model.requires_grad_(False)
        # # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None

        for name, param in self.model.named_parameters():
            if name.find('data') != -1 or name.find('prompt') != -1 :
                param.requires_grad = True

        return self.model


    def init_dataloader(self):
        tr_img_list, tr_label_list = convert_unlabeled_list( self.args.tr_csv, r=1)
        if tr_label_list is not None:
            self.logger.info('-----------Train:img-{}-label-{} -----------'.format(len(tr_img_list), len(tr_label_list)))
        else:
            self.logger.info('-----------Train:img-{}-----------'.format(len(tr_img_list)))
        tr_dataset = RIGA_unlabeled_set( self.args.root, tr_img_list, self.args.patch_size)
        ts_img_list, ts_label_list = convert_labeled_list(self.args.ts_csv, r=1)
        self.logger.info('-----------Test:img-{}-label-{} -----------'.format(len(ts_img_list), len(ts_label_list)))
        ts_dataset = RIGA_labeled_set( self.args.root, ts_img_list, ts_label_list,  self.args.patch_size)
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size= self.args.batch_size,
                                                    num_workers= self.args.num_threads,
                                                    shuffle= True,
                                                    pin_memory=True,
                                                    collate_fn=target_collate_fn_tr_fda)
        self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size= self.args.batch_size,
                                                    num_workers= self.args.num_threads // 2,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn_ts)

    def optimize_parameters(self, input, loss_fn=None):
        if args.model == 'BN-Sta-TTA':
            self.optimizer.zero_grad()
            pred, bn_f, _, _ = self.model(input, training=True) 
            loss = bn_loss(self.model, self.pretrained_params, bn_f)
            loss.backward()
            self.optimizer.step()
            return pred, loss
        else:
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = loss_fn(pred)
            loss.backward()
            self.optimizer.step()
            return pred, loss

    def evaluate(self):
        start_epoch = 0
        if args.continue_training:
            assert isfile(join(self.model_folder, 'model_latest.model')), 'missing model checkpoint!'
            params = torch.load(join(self.model_folder, 'model_latest.model'))
            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(start_epoch))
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print('Trainable', k)
        for epoch in range(start_epoch, args.num_epochs):
            self.do_disc_cup_seg_one_epoch(epoch, self.loss_fn)
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict()
            }
            print('  Saving model_{}.model...'.format('final'))
            torch.save(saved_model, join(self.model_folder, 'model_final.model'))
           

        # inference
        from inference.inference_nets.inference_tta import inference
        for ts_csv_path in self.ts_csv:
            inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
            self.logger.info("Running inference: {}".format(inference_tag))
            inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                      inference_tag, self.logger)

    def set_train_state(self):
        self.model.train()

    def do_disc_cup_seg_one_epoch(self, epoch, loss_fn):
        """
        Use another unlabelled shuffled ts datasets and test on the unshuffled tr dataset
         """
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, 20)
        self.logger.info('  lr: {}'.format(lr))

        train_loss_list = list()
        for iter, batch in enumerate(self.tr_dataloader):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            output, loss = self.optimize_parameters(data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())
        mean_tr_loss = np.mean(train_loss_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info('  Tr loss: {}\n'.format(mean_tr_loss))

        val_disc_dice_list = list()
        val_cup_dice_list = list()
        with torch.no_grad():
            self.model.eval()
            for iter, batch in enumerate(self.ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    output = self.model(data)
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
           
            
        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)


        self.writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        self.writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
        self.writer.add_scalar("Val Scalars/Class-Avg Dice", (mean_val_cup_dice+mean_val_disc_dice)/2, epoch)

   
        self.logger.info(' Val disc dice: {}; Cup dice: {}'.format( mean_val_disc_dice, mean_val_cup_dice))

        time_per_epoch = time() - start_epoch
        self.logger.info('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Moment-TTA", required=False,choices=['Moment-TTA', 'Tent-TTA', 'RN-CR-TTA', "BN-Sta-TTA"],
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_sptta", required=False,
                        help='Network architecture.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--num_classes', default=2, required=False,
                        help='Num of classes.')
    parser.add_argument('--manualseed', type=int, default=47, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=False, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Base3_test_sptta", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=5e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='sgd', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='../Medical_TTA/data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=False, default=['../Medical_TTA/data/RIGAPlus/MESSIDOR_Base3_test.csv'], help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['../Medical_TTA/data/RIGAPlus/MESSIDOR_Base3_test.csv'],
                        help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=20, required=False,
                        help='num_epochs.')

    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_RIGA/checkpoints/model_best.model',
                        required=False, help='pretrained model path.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')
    parser.add_argument("--norm_layer", type=str, default='batch_norm', help=
                        "select which normalization layer to be used in the model")

    args = parser.parse_args()
    adpater = Offline_Adapter(args)
    adpater.evaluate()

