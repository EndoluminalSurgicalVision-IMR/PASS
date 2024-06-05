# -*- coding:utf-8 -*-
import os
import sys


import argparse
from utils.file_utils import gen_random_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="PLS", required=False,
                        help='Model name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0, 1], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder', required=False, default='log_dir',
                        help='Log folder.')
    parser.add_argument('--arch', default="unet_2d_pls", required=False,
                        help='Achitecture.')
    parser.add_argument('--tag', default="Base1_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='batch size.')
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
    parser.add_argument('-r', '--root', required=False, default='RIGAPlus/',
                        help='dataset root folder.')
    parser.add_argument('--tr_csv', nargs='+',
                        required=False, default=['RIGAPlus/MESSIDOR_Base1_test.csv'], help='training csv file.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['RIGAPlus/MESSIDOR_Base1_test.csv'],
                        help='test csv file.')
    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                        help='num_epochs.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model',
                        required=False, help='pretrained model path.')
    parser.add_argument('--alpha', type=float, default=0.01, required=False,
                        help='alpha in BN loss.')
    parser.add_argument('--layers', type=int, default=5, required=False,
                        help='layers to calculate bn loss.')
    parser.add_argument('--prompt_model_path', required=False,
                        help='domain-aware prompt model path.')
    parser.add_argument('--gamma', type=float, default=0.01, required=False,
                        help='gamma in feature alignment loss.')

    args = parser.parse_args()
    model_name = args.model

    if model_name == 'UNet':
        from source_training.train_nets.train_unet import train
    elif model_name == 'PLS':
        from source_training.train_nets.train_pls import train
    elif model_name == 'FAS':
        from source_training.train_nets.train_fas import train
    else:
        print('No model named "{}"!'.format(model_name))
        return
    train(args)


if __name__ == '__main__':
    main()
