# -*- coding:utf-8 -*-
def convert_labeled_list(csv_list, r=1):
    img_pair_list = list()
    for csv_file in csv_list:
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    if len(img_pair_list[0].split(',')) == 1:
        label_list = None
    else:
        if r==1:
            label_list = [i.split(',')[-1].replace('.tif', '-{}.tif'.format(r)) for i in img_pair_list]
        else:
            label_list = [i.split(',')[-1] for i in img_pair_list]
    return img_list, label_list


def convert_unlabeled_list(csv_list, r=1):
    img_pair_list = list()
    print('******csv-list', csv_list)
    for csv_file in csv_list:
        print('******csv-file', csv_file)
        with open(csv_file, 'r') as f:
            img_in_csv = f.read().split('\n')[1:-1]
        img_pair_list += img_in_csv
    img_list = [i.split(',')[0] for i in img_pair_list]
    label_list = None
    return img_list, label_list
