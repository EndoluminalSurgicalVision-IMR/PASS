"""
Prompt-learning based TTA for RIGA and Prostate datasets.
"""
import argparse
from utils.init import init_random_and_cudnn, Recorder
from time import time
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
from models.PromptTTA.pls_fas import UNet_PLS
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import target_collate_fn_tr_fda, collate_fn_ts,collate_fn_tr, prostate_target_collate_fn_tr_fda
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
from torch import einsum
from inference.inference_nets.inference_tta import pseudo_label_refinement
from test_time_training_offline.outer_tta import *
from tqdm import tqdm


class PLS_FAS_TTA(BaseAdapter):
    def __init__(self, args):
        super(PLS_FAS_TTA, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)
        self.loss_fn = None
       
        
    def init_from_source_model(self):
        assert isfile(args.pretrained_model), 'missing model checkpoint!'
        self.pretrained_params = torch.load(args.pretrained_model)
        
        if args.arch == 'unet_2d_pls':
            self.model = UNet_PLS(pretrained_path=args.pretrained_model, num_classes=self.args.num_classes, patch_size=self.args.patch_size)
            self.model = self.model.to(self.device)
        else:
            NotImplementedError('Not implemented optimizing step for this architecture')


        self.amp_grad_scaler = GradScaler()

    def configure_model_2d(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        # self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # m.track_running_stats = False
                # m.running_mean = None
                # m.running_var = None

        return self.model

    def init_dataloader(self):
        if 'RIGA' in self.args.root:
            self.dataset_name = 'RIGA'
            tr_img_list, tr_label_list = convert_unlabeled_list(self.args.tr_csv, r=1)
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
        elif 'Prostate' in self.args.tag:
             self.dataset_name = 'Prostate'
             tr_img_list, tr_label_list = convert_labeled_list(args.tr_csv, r=-1)
             tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'tta2d', tuple(args.patch_size), img_normalize=True)
             self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers= args.num_threads,
                                                    shuffle= True,
                                                    pin_memory=True,
                                                    collate_fn=prostate_target_collate_fn_tr_fda)
            
             ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=-1)
             ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
             self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                        batch_size=1,
                                                        num_workers=args.num_threads//2,
                                                        shuffle=False,
                                                        pin_memory=True)

    def optimize_parameters(self, input, input_fda, loss_fn=None):
        criterion = nn.BCEWithLogitsLoss()
        
        with autocast():
            [output, feature], bn_f = self.model(input, training=True, get_bottleneck_fea=True)
            
            [output_fda, feature_fda],_ = self.model(input_fda, training=True, get_bottleneck_fea=True)
            sta_loss = bn_loss(self.model, self.pretrained_params, bn_f, alpha=self.args.alpha, i=self.args.layers)

            feature_alignment_loss = F.l1_loss(feature, feature_fda.detach()) + \
                            F.l1_loss(feature_fda, feature.detach())
            pseudo_seg = (output.sigmoid()>0.5).float()
            seg_alignment_loss = criterion(output_fda, pseudo_seg)


            loss = sta_loss + (feature_alignment_loss + seg_alignment_loss)

        self.amp_grad_scaler.scale(loss).backward()
        self.amp_grad_scaler.unscale_(self.optimizer)
        self.amp_grad_scaler.step(self.optimizer)
        self.amp_grad_scaler.update()
        return output, loss
       

    def evaluate(self):
        start_epoch = 0
        if args.continue_training:
            assert isfile(join(self.model_folder, 'model_latest.model')), 'missing model checkpoint!'
            params = torch.load(join(self.model_folder, 'model_latest.model'))
            self.model.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            start_epoch = params['epoch']
        self.logger.info('start epoch: {}'.format(start_epoch))
       
        for epoch in range(start_epoch, args.num_epochs):
            if self.dataset_name == 'RIGA':
                self.do_disc_cup_seg_one_epoch(epoch, self.loss_fn)
            elif self.dataset_name == 'Prostate':
                self.do_prostate_seg_one_epoch(epoch, self.loss_fn)
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict()
            }
            print('  Saving model_{}.model...'.format('final'))
            torch.save(saved_model, join(self.model_folder, 'model_final.model'))

        # inference
        if self.dataset_name == 'RIGA':
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                self.logger.info("Running inference: {}".format(inference_tag))
                inference(args, self.model, self.device, self.log_folder, [ts_csv_path],
                        inference_tag)
        elif self.dataset_name == 'Prostate':
            from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, save_pre=False, test_batch=16)

    def set_train_state(self):
        self.model.eval()

    def do_disc_cup_seg_one_epoch(self, epoch, loss_fn):
        """
        Use another unlabelled shuffled ts datasets and test on the unshuffled tr dataset
         """
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, self.args.num_epochs)
        self.logger.info('  lr: {}'.format(lr))

        train_loss_list = list()
        for iter, batch in enumerate(self.tr_dataloader):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            # seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(normalize_image(batch['fda_data'])).cuda().to(dtype=torch.float32)
            output, loss = self.optimize_parameters(data, fda_data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())
        mean_tr_loss = np.mean(train_loss_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info('  Tr loss: {}\n'.format(mean_tr_loss))

        val_disc_dice_list = list()
        val_cup_dice_list = list()
        val_refined_disc_dice_list = list()
        val_refined_cup_dice_list = list()
        with torch.no_grad():
            self.model.eval()
            for iter, batch in enumerate(self.ts_dataloader):
                data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                with autocast():
                    output = self.model(data)

                refined_output = pseudo_label_refinement(data, output, idc=[0,1], uncertainty_threshold=0.3, alpha=0.6, beta=1.4)
                
                output_sigmoid = torch.sigmoid(output)
                val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
                val_refined_disc_dice_list.append(get_hard_dice(refined_output[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                val_refined_cup_dice_list.append(get_hard_dice(refined_output[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))

                print('org disc cup', get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0), get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
                
                print('refined disc cup', get_hard_dice(refined_output[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0), get_hard_dice(refined_output[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))

        mean_val_disc_dice = np.mean(val_disc_dice_list)
        mean_val_cup_dice = np.mean(val_cup_dice_list)

        mean_val_refined_disc_dice = np.mean(val_refined_disc_dice_list)
        mean_val_refined_cup_dice = np.mean(val_refined_cup_dice_list)

        self.writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
        self.writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
        self.writer.add_scalar("Val Scalars/Class-Avg Dice", (mean_val_cup_dice+mean_val_disc_dice)/2, epoch)

        self.writer.add_scalar("Val Scalars/Refined Disc Dice", mean_val_refined_disc_dice, epoch)
        self.writer.add_scalar("Val Scalars/Refined Cup Dice", mean_val_refined_cup_dice, epoch)
        self.writer.add_scalar("Val Scalars/Class-Avg Refined Dice", (mean_val_refined_cup_dice+mean_val_refined_disc_dice)/2, epoch)

        self.logger.info(' Val disc dice: {}; Cup dice: {}'.format( mean_val_disc_dice, mean_val_cup_dice))

        self.logger.info(' Val refined disc dice: {}; Cup dice: {}'.format( mean_val_refined_disc_dice, mean_val_refined_cup_dice))

        time_per_epoch = time() - start_epoch
        self.logger.info('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)


    def do_prostate_seg_one_epoch(self, epoch, loss_fn):
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, self.args.num_epochs)
        self.logger.info('  lr: {}'.format(lr))
    
        train_loss_list = list()
        train_dice_list = list()
        for iter, batch in tqdm(enumerate(self.tr_dataloader)):
            data = torch.from_numpy(normalize_image(batch['data'])).cuda().to(dtype=torch.float32)
            # print('data shape', data.shape)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
            fda_data = torch.from_numpy(normalize_image(batch['fda_data'])).cuda().to(dtype=torch.float32)
            # print('fda data shape', fda_data.shape)
            output, loss = self.optimize_parameters(data, fda_data, loss_fn)
            pred = torch.sigmoid(output)
            # (pred.size(), seg.size())
            train_dice_list.append(get_hard_dice(pred.cpu().squeeze(),seg.cpu().squeeze()))
        train_loss_list.append(loss.detach().cpu().numpy())
        mean_tr_loss = np.mean(train_loss_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info(' Tr loss: {}'.format(mean_tr_loss))
        
      
        mean_tr_dice = np.mean(train_dice_list)
        self.writer.add_scalar("Train Scalars/Dice", mean_tr_dice, epoch)
        self.logger.info('  Tr-dice: {}'.format(mean_tr_dice))
            
        time_per_epoch = time() - start_epoch
        self.logger.info('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="PLS_FAS_TTA", required=False,
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d_pls", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', type=int, default=1, required=False,
                        help='class number.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=47, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--only_bn_updated', default=False, required=False,
                        help='which part to be updated.')
    parser.add_argument('--tag', default="Prostate_UCL", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[384, 384], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='sgd', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/MRI_prostate', required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=False, default=['data/MRI_prostate/UCL_all.csv'], help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/MRI_prostate/UCL_all.csv'],
                        help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    # parser.add_argument('--optim_steps', type=int, default=1, required=False,
    #                     help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/unet_Prostate_baseline/SiteA_RUNMC_batch_aug_v4/20231007-230537/checkpoints/model_final.model',
                        required=False, help='pretrained model path.')
    # BN loss
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')

    args = parser.parse_args()
    from scripts.tta_configs import Base2_config, Prostate_RUNMC2BIDMC_config
    args.__dict__.update(Base2_config)
    adpater = PLS_FAS_TTA(args)
    adpater.evaluate()
   


