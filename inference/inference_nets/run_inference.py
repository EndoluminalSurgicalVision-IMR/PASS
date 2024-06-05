# -*- coding:utf-8 -*-
import os
import argparse
from batchgenerators.utilities.file_and_folder_operations import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="UNet", required=False,
                        help='Model name.')
    parser.add_argument('--arch', default="unet_2d", required=False,
                        help='Model name.')
    parser.add_argument('--chk', default="model_best.model", required=False,
                        help='Checkpoint name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder', required=False, default='log_dir',
                        help='Log folder.')
    parser.add_argument('--tag', default="Base1_test", required=False,
                        help='Run identifier.')
    parser.add_argument('--inference_tag', default="all", required=False,
                        help='Inference tag.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('-r', '--root', required=False, default='data/RIGAPlus/',
                        help='dataset root folder.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=False, default=['data/RIGAPlus/MESSIDOR_Base1_test.csv'],
                        help='test csv file.')
    parser.add_argument('--pretrained_model', default='log_dir/UNet_Source_Model/checkpoints/model_best.model',
                        required=False, help='pretrained model path.')
    parser.add_argument('--prompt_model_path', required=False,
                        help='domain-aware prompt model path.')
    

    args = parser.parse_args()
    model_name = args.model
    chk_name = args.chk
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    root_folder = args.root
    ts_csv = tuple(args.ts_csv)
    inference_tag = args.inference_tag
    pretrained_model_path = args.pretrained_model
    prompt_model_path = args.prompt_model_path

    if model_name == 'UNet':
        from inference.inference_nets.inference_unet import inference
        inference(args, chk_name,  root_folder, log_folder,  ts_csv, inference_tag)
    else:
        print('No model named "{}"!'.format(model_name))
        return


if __name__ == '__main__':
    main()
