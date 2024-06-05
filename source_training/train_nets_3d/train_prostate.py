"""
Trainer for CT Lung Lobe dataset: class-0-5
"""
from source_training.train_nets_3d.base_trainer import BaseTrainer
import os
from time import time
from tqdm import tqdm
from models.unet import UNet
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
from datasets.utils.transform import source_collate_fn_tr, collate_fn_ts
from utils.lr import adjust_learning_rate, step_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from utils.losses.seg_loss import DiceLoss, CELoss, Dice_CE_Loss, BCEDiceLoss
from utils.metrics.dice import get_hard_dice, SegMetric_Numpy
from utils.tools import AverageMeter, create_one_hot_from_2d_label, create_one_hot_from_3d_label
from tqdm import tqdm
import medpy.metric.binary as medpy_binay_metric


class Prostate_Trainer(BaseTrainer):
    def __init__(self, args):
        super(Prostate_Trainer, self).__init__(args)

        
    def init_criterion(self):
        """
        Define loss functions here.
        """
        self.criterion = BCEDiceLoss().to(self.device)
        
    def init_dataloader(self):
        self.tr_img_list, self.tr_label_list = convert_labeled_list(self.args.tr_csv, r=-1)
        tr_dataset = Prostate_labeled_set(self.root, self.tr_img_list, self.tr_label_list, 'train2d', self.patch_size, img_normalize=True)
        self.ts_img_list, self.ts_label_list = convert_labeled_list(self.args.ts_csv, r=-1)
        ts_dataset = Prostate_labeled_set(self.root, self.ts_img_list, self.ts_label_list, 'test3d', self.patch_size, img_normalize=True)
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                    batch_size=self.args.batch_size,
                                                    num_workers=self.args.num_threads,
                                                    shuffle=True,
                                                    pin_memory=True)
        self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=self.args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)
    
    def train(self):
        amp_grad_scaler = GradScaler()
        start = time()
        best_metric = 0
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.logger.info('  Epoch {}:'.format(epoch))
            self.model.train()
            #lr = adjust_learning_rate(self.optimizer, epoch, cur_lr, self.args.num_epochs)
            cur_lr = step_learning_rate(self.optimizer, epoch, self.get_lr(), [100, 150])
            self.logger.info('  lr: {}'.format(cur_lr))

            train_loss_list = list()
            train_dice_list = list()
            for iter, batch in tqdm(enumerate(self.tr_dataloader)):
                data, seg = batch
                data = data.to(self.device)
                seg = seg.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, seg, sigmoid=True)
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.unscale_(self.optimizer)
                amp_grad_scaler.step(self.optimizer)
                amp_grad_scaler.update()
                train_loss_list.append(loss.detach().cpu().numpy())
                # cal 2d dice
                pred = torch.sigmoid(output)
                # print(pred.size(), seg.size())
                train_dice_list.append(get_hard_dice(pred.cpu().squeeze(),seg.cpu().squeeze()))
                del seg
            mean_tr_loss = np.mean(train_loss_list)
            self.writer.add_scalar("Train Scalars/Learning Rate", cur_lr, epoch)
            self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
            self.logger.info(' Tr loss: {}'.format(mean_tr_loss))
            
            # 对每个类别的dice求平均， 0类为背景
            mean_tr_dice = np.mean(train_dice_list)
          
            self.writer.add_scalar("Train Scalars/Dice", mean_tr_dice, epoch)
            self.logger.info('  Tr-dice: {}'.format(mean_tr_dice))
                
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            print('Saving model_{}.model...'.format('final'))
            torch.save(saved_model, os.path.join(self.model_folder, 'model_final.model'))
            if os.path.isfile(os.path.join(self.model_folder, 'model_latest.model')):
                os.remove(os.path.join(self.model_folder, 'model_latest.model'))
            total_time = time() - start
            print("Running %d epochs took a total of %.2f seconds." % (args.num_epochs, total_time))
            
            if epoch % 10 == 0:
                dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = self.test(epoch)
                if dice_avg_all_classes > best_metric:
                    best_metric = dice_avg_all_classes
                    print('Saving model_{}.model...'.format('best'))
                    torch.save(saved_model, os.path.join(self.model_folder, 'model_best.model'))
                    
    def test_one_case(self, case_data, test_batch):
        # case data: [D, H, W], pred-all: [K, D, H, W]
        assert len(case_data.shape) == 3
        print('case-data', case_data.shape, case_data.shape[-2], case_data.shape[-2])
        assert case_data.shape[-2] == self.args.patch_size[0]
        assert case_data.shape[-1] == self.args.patch_size[1]
        pred_all = torch.zeros([self.args.num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
        s = 0
        while s < case_data.shape[0]:
            batch = min(test_batch, case_data.shape[0]-s)
            s_end = s+batch
            slice = case_data[s:s_end, :, :].unsqueeze(1)
            if len(slice.shape) ==3 :
                # test_batch  = 1
                slice = slice.unsqueeze(1)
            # repeat 3 channels for ResNet input
            pred_s = self.model(slice.repeat(1, 3, 1, 1))
            # pred_s: [B, 1, H, W]
            pred_all[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
            s = s_end
        return pred_all
    
    def test(self, epoch):
        # Calculate the dice of different class, respectively.
        Dice_Metrics = AverageMeter()
        ASSD_Metrics = AverageMeter()
        HD_Metrics = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(self.ts_dataloader)):
                case_data, label, case_name = sample
                self.logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(self.device).squeeze()
    
                pred_aggregated = self.test_one_case(case_data, self.args.batch_size)
                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()

                if target_array.sum() == 0 and pred_array.sum() == 0:
                    dc = 1
                    assd = 0
                    hd = 0

                elif target_array.sum() == 0 or pred_array.sum() == 0:
                    self.logger.warning(
                        'Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                    dc = 0
                    assd = 1
                    hd = 1
                else:
                    target_array = np.asarray(target_array > 0.5).astype(np.bool)
                    pred_array = np.asarray(pred_array > 0.5).astype(np.bool)

                    dc = medpy_binay_metric.dc(target_array, pred_array)
                    assd = medpy_binay_metric.assd(pred_array,
                                                target_array)
                    hd = medpy_binay_metric.hd95(pred_array,
                                                target_array)

                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)

                self.logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
                del pred_array
                del target_array
                del pred_aggregated

                # sys.stdout.flush()

        avg_dc = Dice_Metrics.avg
        avg_hd = HD_Metrics.avg
        avg_assd = ASSD_Metrics.avg

        self.logger.info(" Val-Epoch{} Avg dice:{:.4f} hd:{:.4f} assd:{:.4f}".format(epoch,  avg_dc,  avg_hd, avg_assd))

        
        return avg_dc, avg_hd, avg_assd
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser.add_argument('--model', default="unet", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Prostate_baseline", required=False,
                        help='Run identifier.')
    parser.add_argument('--method_tag', default="SiteE_BIDMC_batch_aug_v4", required=False,
                        help='Method identifier.')
    parser.add_argument('--num_classes', type=int, default=1, required=False,
                        help='class number.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[384, 384], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=32, required=False,
                        help='batch size.')
    parser.add_argument('--optimizer', type=str, default='adam', required=False,
                        help='optimizer method.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--save_interval', type=int, default=25, required=False,
                        help='save_interval.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/MRI_prostate', required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',default=['data/MRI_prostate/BIDMC_train.csv'],
                        required=False, help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',default=['data/MRI_prostate/BIDMC_test.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=800, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', required=False,
                        help='pretrained model path.')

    args = parser.parse_args()
    trainer = Prostate_Trainer(args)
    trainer.train()
