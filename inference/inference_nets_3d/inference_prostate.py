"""
Tester for MRI Prostate dataset: class-1
"""
from inference.inference_nets_3d.base_tester import BaseTester
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from time import time
from tqdm import tqdm
from models.unet import UNet
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set
from datasets.utils.convert_csv_to_list import convert_labeled_list
import torch
import numpy as np
from utils.metrics.dice import  get_dice_assd_hd, get_mean_dsc
from utils.tools import AverageMeter, save_np2nii, save_tensor2nii
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
from models import *
import models.TENT.dua as dua

def test3d_single_label_seg(method, model, ts_dataloader, logger, device,  visualization_folder, metrics_folder, num_classes=6, test_batch=32, save_pre=False):
    # Calculate the dice of different class, respectively.
    Dice_Metrics = AverageMeter(save_all=True)
    ASSD_Metrics = AverageMeter(save_all=True)
    HD_Metrics = AverageMeter(save_all=True)
    all_case_name = []


    if method == 'TENT' or method == 'TIPI' or \
        method == 'PTBN' or method == 'SAR' or method == 'CoTTA':
        logger.info('Testing with {} method'.format(method))
        all_case_name = []
        for iter, sample in tqdm(enumerate(ts_dataloader)):
            case_data, label, case_name = sample
            logger.info('Testing case-: {}'.format(case_name))
            case_data = case_data.to(device).squeeze()
            all_case_name.append(case_name[0])

            assert len(case_data.shape) == 3
            pred_aggregated = torch.zeros([num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
            s = 0
            while s < case_data.shape[0]:
                # if s == 0:
                #     reset = True
                # else:
                reset = False
                batch = min(test_batch, case_data.shape[0]-s)
                s_end = s+batch
                slice = case_data[s:s_end, :, :].unsqueeze(1)
                if len(slice.shape) ==3 :
                    # test_batch  = 1
                    slice = slice.unsqueeze(1)
                # repeat 3 channels for ResNet input
                data = slice.repeat(1, 3, 1, 1)
                pred_s = model(data, reset)
                # pred_s: [B, K, H, W]
                pred_aggregated[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
                s = s_end
            print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())
    
            pred_array = pred_aggregated.detach().squeeze().cpu().numpy()
            target_array = label.squeeze().numpy()

            dc, assd, hd = get_dice_assd_hd(target_array, pred_array, logger)

            Dice_Metrics.update(dc)
            ASSD_Metrics.update(assd)
            HD_Metrics.update(hd)

            logger.info(
                "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
            # del pred_array
            # del target_array
            # del pred_aggregated

            # sys.stdout.flush()

    elif method == 'DUA':
        logger.info('Testing with {} method'.format(method))
        model.eval()
        decay_factor = 0.94
        min_momentum_constant = 0.005
        mom_pre = 0.1
        all_case_name = []
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(ts_dataloader)):
                model.eval()
                case_data, label, case_name = sample
                logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(test_batch, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) ==3 :
                        # test_batch  = 1
                        slice = slice.unsqueeze(1)

                    data = slice.repeat(1, 3, 1, 1)
                    # print('min max', torch.min(data), torch.max(data))
                    ################# DUA change momentum ###################
                    mom_new = (mom_pre * decay_factor)
                    for m in model.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.train()
                            m.momentum = mom_new + min_momentum_constant
                    mom_pre = mom_new
                    ###################### AUG #################
                    inputs = dua.get_adaption_inputs_seg(data, device, aug_view_num=64)
                    _ = model(inputs)
                    with torch.no_grad():
                        model.eval()
                    # repeat 3 channels for ResNet input
                    pred_s = model(data)
                    # pred_s: [B, K, H, W]
                    pred_aggregated[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
                    s = s_end
                print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())

                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()

                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated

    elif 'DPG' in method:
        logger.info('Testing with {} method'.format(method))
        priormodel = model[0]
        segmodel = model[1]
       
        with torch.no_grad():
            priormodel.eval()
            segmodel.train()
            for iter, sample in tqdm(enumerate(ts_dataloader)):
                case_data, label, case_name = sample
                logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(test_batch, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) ==3 :
                        # test_batch  = 1
                        slice = slice.unsqueeze(1)
                    # repeat 3 channels for ResNet input
                    prior = priormodel(slice.repeat(1, 3, 1, 1), True)
                    pred_s = segmodel(slice.repeat(1, 3, 1, 1), prior)
                    # pred_s: [B, K, H, W]
                    pred_aggregated[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
                    s = s_end
                print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())

                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()
        
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated
             
                sys.stdout.flush()
            
           
               

    elif 'DAE' in method:
        logger.info('Testing with {} method'.format(method))
        Norm_model = model[0]
        Seg_model = model[1]
        DAE = model[2]
        Norm_model.eval()
        Seg_model.eval()
        DAE.eval()
        all_case_name = []
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(ts_dataloader)):
                case_data, label, case_name = sample
                logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(test_batch, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) ==3 :
                        # test_batch  = 1
                        slice = slice.unsqueeze(1)
                    # repeat 3 channels for ResNet input
                    norm_data = Norm_model(slice.repeat(1, 3, 1, 1))
                    pred_s = Seg_model(norm_data)
                    # pred_s: [B, K, H, W]
                    pred_s = torch.sigmoid(pred_s)
                    #采取2d-DAE后的结果
                    if '3' not in method:
                        pred_s = DAE(pred_s)
    
                    pred_aggregated[:, s:s_end, :, :] = pred_s.permute(1, 0, 2, 3)#torch.sigmoid(pred_s).permute(1, 0, 2, 3)
                    s = s_end
               
                print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())

                #采取3d-DAE后的结果
                if '3' in method:
                    pred_aggregated_pad = torch.cat((pred_aggregated.squeeze(), torch.zeros((16 - case_data.shape[0] % 16, case_data.shape[-2], case_data.shape[-1]))), dim=0)
                    pred_aggregated = DAE(pred_aggregated_pad.unsqueeze(0).unsqueeze(0)).squeeze()[:case_data.shape[0], :, :]


                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()
        
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated
             
                sys.stdout.flush()
            
    else:
        logger.info('Testing with {} method'.format(method))
        model.eval()
        all_case_name = []
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(ts_dataloader)):
                case_data, label, case_name = sample
                logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(device).squeeze()
                all_case_name.append(case_name[0])
    
                assert len(case_data.shape) == 3
                pred_aggregated = torch.zeros([num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
                s = 0
                while s < case_data.shape[0]:
                    batch = min(test_batch, case_data.shape[0]-s)
                    s_end = s+batch
                    slice = case_data[s:s_end, :, :].unsqueeze(1)
                    if len(slice.shape) ==3 :
                        # test_batch  = 1
                        slice = slice.unsqueeze(1)
                    # repeat 3 channels for ResNet input
                    pred_s = model(slice.repeat(1, 3, 1, 1))
                    # pred_s: [B, K, H, W]
                    pred_aggregated[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
                    s = s_end
                print('pred-agg', pred_aggregated.shape, pred_aggregated.min(), pred_aggregated.max())

                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()
        
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, logger)
               
                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
              
                del pred_array
                del target_array
                del pred_aggregated
             
                sys.stdout.flush()
    avg_dc = Dice_Metrics.avg
    avg_hd = HD_Metrics.avg
    avg_assd = ASSD_Metrics.avg

    std_dc = np.std(Dice_Metrics.all_data)
    std_hd = np.std(HD_Metrics.all_data)
    std_assd = np.std(ASSD_Metrics.all_data)

    logger.info(
            " Test Avg- dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
    

    # save results of each class
    data_frame = pd.DataFrame(
    data={'Case': all_case_name,
                    'Dice': Dice_Metrics.all_data,
                    'HD': HD_Metrics.all_data,
                    'ASSD': ASSD_Metrics.all_data},
            index=range(len(all_case_name)))
    data_frame.to_csv(metrics_folder + '/results.csv',
                            index_label='Index')

    sys.stdout.flush()
    return avg_dc, avg_hd, avg_assd

        
class Prostate_Tester(BaseTester):
    def __init__(self, args):
        super(Prostate_Tester, self).__init__(args)
        self.test_batch = args.test_batch

    def load_model(self):
        if self.args.model == 'unet_2d':
            self.model = UNet(num_classes=self.args.num_classes)
            self.model = self.model.to(self.device)
       
        elif self.args.model == 'unet_2d_sptta':
            self.model = UNet_SPTTA(pretrained_path=None,
                                          num_classes=self.args.num_classes, 
                                        patch_size=self.args.patch_size)
            self.model = self.model.to(self.device)
            
        else:
            raise NotImplementedError('model not implemented!')

        self.model = self.model.to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)
        assert os.path.isfile(self.model_path), 'missing model checkpoint {}!'.format(self.model_path)
        self.logger.info('Load model for inference:{}'.format(self.model_path))
        params = torch.load(self.model_path)
        self.model.load_state_dict(params['model_state_dict'])
        
  
    def init_dataloader(self):
        self.ts_img_list, self.ts_label_list = convert_labeled_list(self.args.ts_csv, r=-1)
        ts_dataset = Prostate_labeled_set(self.args.root, self.ts_img_list, self.ts_label_list, 'test3d', self.patch_size, img_normalize=True)
        self.ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=self.args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)
        
    def test_one_case(self, case_data):
         # case data: [D, H, W], pred-all: [K, D, H, W]
        assert len(case_data.shape) == 3
        print('case-data', case_data.shape, case_data.shape[-2], case_data.shape[-2])
        assert case_data.shape[-2] == self.args.patch_size[0]
        assert case_data.shape[-1] == self.args.patch_size[1]
        pred_all = torch.zeros([self.args.num_classes, case_data.shape[-3], case_data.shape[-2], case_data.shape[-1]])
        s = 0
        while s < case_data.shape[0]:
            batch = min(self.args.test_batch, case_data.shape[0]-s)
            s_end = s+batch
            slice = case_data[s:s_end, :, :].unsqueeze(1)
            if len(slice.shape) ==3 :
                # test_batch  = 1
                slice = slice.unsqueeze(1)
            # repeat 3 channels for ResNet input
            pred_s = self.model(slice.repeat(1, 3, 1, 1))
            # pred_s: [B, K, H, W]
            pred_all[:, s:s_end, :, :] = torch.sigmoid(pred_s).permute(1, 0, 2, 3)
            s = s_end
        return pred_all
    
    def test(self):
        # Calculate the dice of different class, respectively.
        Dice_Metrics = AverageMeter(save_all=True)
        ASSD_Metrics = AverageMeter(save_all=True)
        HD_Metrics = AverageMeter(save_all=True)
        Mean_Dice_Metrics = AverageMeter(save_all=True)
        all_case_name = []

        self.model.eval()
        with torch.no_grad():
            for iter, sample in tqdm(enumerate(self.ts_dataloader)):
                case_data, label, case_name = sample
                self.logger.info('Testing case-: {}'.format(case_name))
                case_data = case_data.to(self.device).squeeze()
                print('case-data', case_data.shape, label.shape)
                all_case_name.append(case_name[0])
    
                pred_aggregated = self.test_one_case(case_data)
                pred_array = pred_aggregated.squeeze().cpu().numpy()
                target_array = label.squeeze().numpy()

              
                dc, assd, hd = get_dice_assd_hd(target_array, pred_array, self.logger)
                mean_dsc = get_mean_dsc(target_array, pred_array)

                Dice_Metrics.update(dc)
                ASSD_Metrics.update(assd)
                HD_Metrics.update(hd)
                Mean_Dice_Metrics.update(mean_dsc)

                self.logger.info(
                    "Cur-patient {}  dice:{:.3f} assd:{:.3f} hd:{:.3f}".format(case_name, dc, assd, hd))
                if self.args.save_pre:
                    #将pred_aggregated存为.nii
                    saved_pred =  pred_array>0.5
                    saved_pred = saved_pred.astype(np.uint8)
                    # print('saved pred', pred_aggregated.shape, saved_pred.min(), saved_pred.max())
                    # saved_pred = sitk.GetImageFromArray(saved_pred)
                    # save_path = self.visualization_folder + '/' + case_name[0] + '_pre.nii.gz'
                    # sitk.WriteImage(saved_pred, save_path)
                    
                    saved_img = case_data.cpu().numpy()
                    print('saved-case_data',saved_img.shape)
                    # saved_img = sitk.GetImageFromArray(saved_img)
                    # save_path = self.visualization_folder + '/' + case_name[0] + '_img.nii.gz'
                    # sitk.WriteImage(saved_img, save_path)

                    save_np2nii(saved_pred, self.visualization_folder, case_name[0] + '_pre.nii.gz')
                    # save_np2nii(saved_img, self.visualization_folder, case_name[0] + '_img.nii.gz')
                del pred_array
                del target_array
                del pred_aggregated

                # sys.stdout.flush()

        avg_dc = Dice_Metrics.avg
        avg_hd = HD_Metrics.avg
        avg_assd = ASSD_Metrics.avg
        avg_mean_dsc = Mean_Dice_Metrics.avg

        std_dc = np.std(Dice_Metrics.all_data)
        std_hd = np.std(HD_Metrics.all_data)
        std_assd = np.std(ASSD_Metrics.all_data)
        std_mean_dsc = np.std(Mean_Dice_Metrics.all_data)

        # self.logger.info(
        #         " Test Avg-dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd))
        self.logger.info("Test Avg-dice:{:.4f}+{:.4f} hd:{:.4f}+{:.4f} assd:{:.4f}+{:.4f} mean-dsc:{:.4f}+{:.4f}".format(avg_dc,std_dc, avg_hd,std_hd, avg_assd, std_assd, avg_mean_dsc, std_mean_dsc))

    
        # save results of each class

        data_frame = pd.DataFrame(
        data={'Case': all_case_name,
                        'Dice': Dice_Metrics.all_data,
                        'HD': HD_Metrics.all_data,
                        'ASSD': ASSD_Metrics.all_data},
                index=range(len(all_case_name)))
        data_frame.to_csv(self.metrics_folder + '/results.csv',
                                index_label='Index')

        sys.stdout.flush()
        return avg_dc, avg_hd, avg_assd
    

    def test_v2(self):
        print('NNNEW Test function')
        dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, save_pre=False)
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet_2d", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Prostate_baseline", required=False,
                        help='Run identifier.')
    parser.add_argument('--method_tag', default="RUNMC2UCL", required=False,
                        help='Method identifier.')
    parser.add_argument('--num_classes', type=int, default=1, required=False,
                        help='class number.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[384, 384], required=False,
                        help='patch size.')
    parser.add_argument('--test_batch', type=int, default=32, required=False,
                        help='batch size.')
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/MRI_prostate', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+',default=['data/MRI_prostate/UCL_all.csv'],
                        required=False, help='test csv file.')
    parser.add_argument('--model_path', required=False, default='log_dir/unet_Prostate_baseline/SiteA_RUNMC_batch_aug_v4/20231007-230537/checkpoints/model_final.model', help='pretrained model path.')
    parser.add_argument('--save_pre', required=False,default=True,
                        help='Whether to save the predicted mask.')

    args = parser.parse_args()
    tester = Prostate_Tester(args)
    tester.test()


    
    
