"""
Method: Test-time adaptable neural networks for robust medical image segmentation
Setup: Online-version, one-case one-step
"""
import os
import argparse
from test_time_training_online.base_tta import BaseAdapter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet, UNet_v2, Norm_Indentity_Net, U_Net_for_DAE
from time import time
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from datasets.utils.normalize import normalize_image


class DAE_Adapter(BaseAdapter):
    def __init__(self, args):
        super(DAE_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)


    def configure_model(self, model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model

    def init_from_source_model(self):
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        self.Norm_model = Norm_Indentity_Net()
        self.Seg_model = UNet_v2()
        self.DAE = U_Net_for_DAE(img_ch=self.args.num_classes, output_ch=self.args.num_classes)

        self.Norm_model = nn.DataParallel(self.Norm_model, device_ids=self.gpus).to(self.device)
        self.Seg_model = nn.DataParallel(self.Seg_model, device_ids=self.gpus).to(self.device)
        self.DAE = nn.DataParallel(self.DAE, device_ids=self.gpus).to(self.device)

        params = torch.load(self.args.pretrained_model)
        self.logger.info('Load source model from {}'.format(self.args.pretrained_model))
        self.Norm_model.load_state_dict(params['norm_model_state_dict'])
        self.Seg_model.load_state_dict(params['seg_model_state_dict'])
        self.DAE.load_state_dict(params['dae_state_dict'])


        self.Seg_model.requires_grad_(False)
        self.DAE.requires_grad_(False)
        self.Norm_model.requires_grad_(True)

        self.optimizer = torch.optim.SGD(list(self.Norm_model.parameters())
                                         + list(self.Seg_model.parameters())
                                         + list(self.DAE.parameters()), lr=self.args.initial_lr,
            momentum=0.99, nesterov=True)

        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = nn.BCELoss()

    def optimize_parameters(self, input, loss_fn):
        self.optimizer.zero_grad()
        norm_data = self.Norm_model(input)
        output = self.Seg_model(norm_data)
        output = torch.sigmoid(output)
        rec_output = self.DAE(output)
        rec_output = torch.clamp(rec_output, 0., 1.)
        # loss = self.loss_fn(rec_output, (output > 0.5).float())
        loss = self.loss_fn(rec_output, output)
        print(loss)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def set_train_state(self):
        self.Norm_model.train()
        self.Seg_model.eval()
        self.DAE.eval()
      
    def evaluate(self):
        start_epoch = 0
        start = time()
        for epoch in range(start_epoch, args.num_epochs):
            if self.tag in ['Base1_test', 'Base2_test', 'Base3_test', 'MESSIDOR', 'BinRushed']:
                self.do_disc_cup_seg_one_epoch(epoch)
            else:
                raise NotImplemented
            
        saved_model = {
            'epoch': 1,
            'seg_model_state_dict': self.Seg_model.state_dict(),
            'norm_model_state_dict': self.Norm_model.state_dict(),
            'dae_state_dict': self.DAE.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dae_optimizer_state_dict': self.dae_optimizer.state_dict()
        }
        print('Saving model_{}.model...'.format('final'))
        torch.save(saved_model, join(self.model_folder, 'model_final.model'))


    def do_disc_cup_seg_one_epoch(self, epoch):
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, self.args.num_epochs)
        self.logger.info('  lr: {}'.format(lr))

        train_loss_list = list()
        test_disc_dice_list = []
        test_cup_dice_list = []
        for iter, batch in enumerate(self.ts_dataloader):
            data = torch.from_numpy(batch['data']).to(self.device).to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).to(self.device).to(dtype=torch.float32)
            self.Norm_model.train()
            self.Seg_model.eval()
            for i in range(self.args.optim_steps):
                output, loss = self.optimize_parameters(data, self.loss_fn)

            self.Seg_model.eval()
            self.Norm_model.eval()
            pred_tta = self.Seg_model(self.Norm_model(data))

            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(pred_tta)
            test_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            test_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg



        mean_tr_loss = np.mean(train_loss_list)
        mean_val_disc_dice = np.mean(test_disc_dice_list)
        mean_val_cup_dice = np.mean(test_cup_dice_list)
        std_val_disc_dice = np.std(test_disc_dice_list)
        std_val_cup_dice = np.std(test_cup_dice_list)
          
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info(' Test disc dice: {:.5f}+{:.5f}; Cup dice: {:.5f}+{:.5f}'.format(mean_val_disc_dice, std_val_disc_dice, mean_val_cup_dice, std_val_cup_dice))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="DAE-TTA", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--arch', default="norm_unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=2, required=False,
                        help='Class number.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Base1_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=1e-4, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+', default=['data/RIGAPlus/MESSIDOR_Base1_test.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=1, required=False,
                        help='num_epochs.')
    parser.add_argument('--optim_steps', type=int, default=5, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='log_dir/DAE_pred_w_seg_Source_RIGA/checkpoints/model_best.model', required=False,
                        help='pretrained model path.')

    args = parser.parse_args()
    adaptor = DAE_Adapter(args)
    adaptor.evaluate()


