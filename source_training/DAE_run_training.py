# -*- coding:utf-8 -*-
import os

import argparse
from utils.file_utils import gen_random_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="DAE", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--arch', default="norm_unet_2d", required=False,
                        help='Architecture name.')
    parser.add_argument('--manualseed', type=int, default=47, required=False,
                        help='random seed.')
    parser.add_argument('--log_folder', default='log_dir', required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="Source_RIGA", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
    parser.add_argument('--in_channel', type=int, default=3, required=False,
                        help='input channel.')
    parser.add_argument('--num_classes', type=int, default=2, required=False,
                        help='class number.')
    parser.add_argument('--initial_lr', type=float, default=1e-2, required=False,
                        help='initial learning rate.')
    parser.add_argument('--save_interval', type=int, default=25, required=False,
                        help='save_interval.')
    parser.add_argument('-c', '--continue_training', default=False, required=False, action='store_true',
                        help="restore from checkpoint and continue training.")
    parser.add_argument('--no_shuffle', default=False, required=False, action='store_true',
                        help="No shuffle training set.")
    parser.add_argument('--num_threads', type=int, default=0, required=False,
                        help="Threads number of dataloader.")
    parser.add_argument('-r', '--root', default='data/RIGAPlus/', required=False,
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=False, default=['data/RIGAPlus/BinRushed_train.csv',
                                                 'data/RIGAPlus/Magrabia_train.csv'], help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/RIGAPlus/MESSIDOR_Base1_test.csv'],
                        # required=False, default=['../RIGAPlus/BinRushed_test.csv',
                        #                          '../RIGAPlus/Magrabia_test.csv'],
                        help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', required=False,
                        help='pretrained model path.')
    args = parser.parse_args()
    from source_training.train_nets.train_dae import train_RIGA, train_Prosatate


    if 'RIGA' in args.root:
        train_RIGA(args)
    else:
         #### for prostate
        args.num_classes = 1
        args.root = 'data/MRI_prostate'
        args.tr_csv = ['data/MRI_prostate/RUNMC_all.csv']
        args.ts_csv = ['data/MRI_prostate/BIDMC_all.csv']
        args.patch_size = [384, 384]
        args.batch_size = 16
        args.num_epochs = 500
        args.tag  = 'Prostate_RUNMC'
        args.continue_training = True
        if 'DAE3D' in args.model:
            from source_training.train_nets_3d.train_dae_3d import train_Prosatate_v2
            train_Prosatate_v2(args)
        else:
            train_Prosatate(args)


if __name__ == '__main__':
    main()





