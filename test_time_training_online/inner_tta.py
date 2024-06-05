"""
< inner_ttt >
Modifiying the forward pass of the model to perform TTT.
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from utils.init import init_random_and_cudnn, Recorder
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from utils.file_utils import *
import argparse
from test_time_training_online.base_tta import BaseAdapter


class Inner_Adapter(BaseAdapter):
    def __init__(self, args):
        super(Inner_Adapter, self).__init__(args)
        print('device:', self.device, 'gpus', self.gpus)

    def evaluate(self):
        if self.dataset_name == 'RIGAPlus':
            from inference.inference_nets.inference_tta import inference
            for ts_csv_path in self.ts_csv:
                inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
                self.logger.info("Running inference: {}".format(inference_tag))
                tta_model = inference(args, self.model, self.device, self.log_folder, [ts_csv_path], inference_tag, self.logger)
                

        elif self.dataset_name == 'Prostate':
            from inference.inference_nets_3d.inference_prostate import test3d_single_label_seg
            dice_avg_all_classes, hd_avg_all_classes, assd_avg_all_classes = test3d_single_label_seg(self.args.model, self.model, self.ts_dataloader, self.logger, self.device, self.visualization_folder, self.metrics_folder, num_classes=self.args.num_classes, test_batch=self.args.batch_size, save_pre=False)

        else:
            raise NotImplemented


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="TIPI", required=False,,choices=['PTBN', 'TIPI', 'TENT', 'CoTTA', 'SAR', 'DUA'],
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--manualseed', type=int, default=100, required=False,
                        help='random seed.')
    parser.add_argument('--arch', default="unet_2d", required=False,
                        help='Network architecture.')
    parser.add_argument('--num_classes', default=1, required=False,
                        help='Num of classes.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Prostate_BIDMC", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[384, 384], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--initial_lr', type=float, default=5e-3, required=False,
                        help='initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam', required=False,
                        help='optimizer method.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/MRI_prostate', required=False,
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/MRI_prostate/BIDMC_all.csv'],
                        help='test csv file.')
    parser.add_argument('--optim_steps', type=int, default=1, required=False,
                        help='optimization steps.')
    parser.add_argument('--pretrained_model', default='log_dir/unet_Prostate_baseline/SiteA_RUNMC_batch_aug_v4/20231007-230537/checkpoints/model_final.model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--model_episodic', type=bool, default=False, required=False,
                        help='To make adaptation episodic, and reset the model for each batch, choose True.')
    parser.add_argument('--only_bn_updated', default=True, required=False,
                        help='which part to be updated.')

    args = parser.parse_args()
    # evaluate(args)
    from scripts.tta_configs import Base1_tent_config,\
    Prostate_RUNMC2UCL_config, Prostate_RUNMC2BMC_config, \
    Prostate_RUNMC2HK_config, Prostate_RUNMC2BIDMC_config,Prostate_RUNMC2I2CVB_config ,Base2_tent_config,Base2_tent_config
    args.__dict__.update(Base2_tent_config)
    args.batch_size = 1
    args.optim_steps = 1
    adapter = Inner_Adapter(args)
    adapter.evaluate()

