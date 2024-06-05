"""
Common training steps for 2D Test-Time Adaptation
"""
from utils.init import init_random_and_cudnn, get_logger
from time import time
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data
import torch.nn as nn
from utils.file_utils import *
# from batchgenerators.utilities.file_and_folder_operations import *
from models.unet import UNet
from datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from datasets.dataloaders.Prostate_dataloader import Prostate_labeled_set, Prostate_labeled_set_one_shape
from datasets.utils.convert_csv_to_list import convert_labeled_list, convert_unlabeled_list
from datasets.utils.transform import collate_fn_tr, collate_fn_ts,  target_collate_fn_tr_fda, collate_fn_ts,collate_fn_tr
from utils.lr import adjust_learning_rate
from utils.metrics.dice import get_hard_dice
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.tools import AverageMeter, create_one_hot_from_2d_label, create_one_hot_from_3d_label


def get_test_dataloader(args):
    ### only test data loader for inference-time updating methods (TENT-Based methods).
    if args.tag in ['Base1_test', 'Base2_test', 'Base3_test', 'MESSIDOR', 'BinRushed']:
        ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=1)
        ts_dataset = RIGA_labeled_set(args.root, ts_img_list, ts_label_list, tuple(args.patch_size))
        test_batch = args.batch_size
        args.dataset_name = 'RIGAPlus'
        ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=test_batch,
                                                num_workers=args.num_threads // 2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts,
                                                drop_last=False)

           
    elif 'Prostate' in args.tag:
        ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=-1)
        ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
        args.dataset_name = 'Prostate'
        # get volumetric data
        ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)

    else:
        raise NotImplementedError


    return ts_dataloader


def get_shuffled_test_dataloader(args):
    ### a shuffled test data loader for TTA, and a test data loader for inference.
    if args.tag in ['Base1_test', 'Base2_test', 'Base3_test', 'MESSIDOR', 'BinRushed']:
        tr_img_list, tr_label_list = convert_unlabeled_list(args.ts_csv, r=1)
        tr_dataset = RIGA_unlabeled_set(args.root, tr_img_list, args.patch_size)
        ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=1)
        ts_dataset = RIGA_labeled_set(args.root, ts_img_list, ts_label_list, args.patch_size)
        train_batch = args.batch_size
        test_batch = args.batch_size
        if 'FAS' in args.model:
            train_collate_fn = target_collate_fn_tr_fda
        else:
            train_collate_fn = collate_fn_tr
        test_collate_fn = collate_fn_ts
        args.dataset_name = 'RIGAPlus'

        tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=train_batch,
                                                num_workers= args.num_threads,
                                                shuffle=True,
                                                pin_memory=True,
                                                collate_fn=train_collate_fn)
   
        ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=test_batch,
                                                num_workers=args.num_threads // 2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=test_collate_fn,
                                                drop_last=False)

    elif 'Prostate' in args.tag:
        args.dataset_name = 'Prostate'
        tr_img_list, tr_label_list = convert_labeled_list(args.tr_csv, r=-1)
        tr_dataset = Prostate_labeled_set(args.root, tr_img_list, tr_label_list, 'val2d', tuple(args.patch_size), img_normalize=True)
        # tr_dataset =  Prostate_labeled_set_one_shape(args.root, tr_img_list, tr_label_list,  tuple(args.patch_size), img_normalize=True)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=args.batch_size,
                                                num_workers= args.num_threads,
                                                shuffle= True,
                                                pin_memory=True)
        
        ts_img_list, ts_label_list = convert_labeled_list(args.ts_csv, r=-1)
        ts_dataset = Prostate_labeled_set(args.root, ts_img_list, ts_label_list, 'test3d', args.patch_size, img_normalize=True)
        ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_threads//2,
                                                    shuffle=False,
                                                    pin_memory=True)
       

    else:
        raise NotImplementedError

    if tr_label_list is not None:
        print('-----------Train:img-{}-label-{} -----------'.format(len(tr_img_list), len(tr_label_list)))
    else:
        print('-----------Train:img-{}-----------'.format(len(tr_img_list)))
 
    return tr_dataloader, ts_dataloader

class BaseAdapter(object):
    def __init__(
            self,
            args):
        """
       Steps:
           1、Init logger.
           2、Init device.
           3、Init seed.
           4、Init data_loader.
           5、Init model.
           6、Init optimizer and scheduler.

       After this call,
           All will be prepared for tta.
       """

        self.args = args
        self.gpus = tuple(args.gpu)
        self.tag = args.tag
        self.log_folder = os.path.join(args.log_folder, args.model + '_' + args.tag)
        self.patch_size = tuple(args.patch_size)
        self.ts_csv = tuple(args.ts_csv)

        self.tensorboard_folder, self.model_folder, self.visualization_folder, self.metrics_folder = check_folders(self.log_folder)
        self.writer = SummaryWriter(log_dir=self.tensorboard_folder)
        self.logger = get_logger(self.log_folder)
        print('RUNDIR: {}'.format(self.log_folder))
        self.logger.info('{}-TTA'.format(self.args.model))
        setting = {k: v for k, v in self.args._get_kwargs()}
        self.logger.info(setting)
        self.device = init_random_and_cudnn(self.gpus, manualseed=args.manualseed, bentchmark=True)

        self.init_from_source_model()
        if args.model == 'TENT' or args.model == 'TTN'  or args.model == 'TIPI' or args.model == 'EATA'  or args.model == 'CoTTA' or args.model == 'SAR'  or args.model == 'DUA' \
        or args.model == 'Source' or args.model == 'OCL-TTT-OnTheFly':
            # whether use another shuffled test data loader for TTA
            self.tr_data_shuffle = False
        elif  args.model == 'OCL-TTT' or args.model == 'DAE-TTA':
            self.tr_data_shuffle = True
        else:
            # Tent/Moment-TTA/BN-STa
            self.tr_data_shuffle = True
            self.init_optimizer_and_scheduler()

        self.init_dataloader()

    def get_lr(self) -> int:
        return self.optimizer.param_groups[0]['lr']

    def init_dataloader(self):
        if self.tr_data_shuffle:
            print('*** shuffle dataloader ***', self.args.tag)
            self.tr_dataloader, self.ts_dataloader = get_shuffled_test_dataloader(self.args)
        else:
            print('*** dataloader ***', self.args.tag)
            self.ts_dataloader = get_test_dataloader(self.args)

    def init_optimizer_and_scheduler(self):

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        # init optimizer
        if self.args.optimizer == 'sgd':
            print('******SGD************')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.initial_lr, momentum=0.99, nesterov=True)

        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.initial_lr, betas=(0.9, 0.999), weight_decay=0.0)

        else:
            raise NotImplementedError

        # init scheduler
        if hasattr(self.args, 'lr_scheduler'):
            if self.args.scheduler == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config.patience * 0.8), gamma=0.5)
        else:
            self.scheduler = None

    def init_from_source_model(self):

        self.recorder.logger.info('use: %d gpus', torch.cuda.device_count())
        assert isfile(self.args.pretrained_model), 'missing model checkpoint!'
        params = torch.load(self.args.pretrained_model)
        if self.args.arch == 'unet_2d':
            self.model = UNet()
            self.model.load_state_dict(params['model_state_dict'])
            if self.args.only_bn_updated:
                self.model = self.configure_model_2d()
                self.logger.info('normalization statistics updated')
            else:
                self.logger.info('WARNING all var updated')
        else:
            raise NotImplementedError

    def check_resume(self):
        pass

    def set_input(self, sample):
        input, target = sample
        self.input = input.to(self.device)
        self.target = target.to(self.device)

    def forward(self, input):
        """
        Define forward behavior here.
        Args:
            sample (tuple): an input-target pair
        """
        pred = self.model(input)
        return pred

    def backward(self, pred, loss_fn):
        """
        Compute the loss function
        Args:
            sample (tuple): an input-target pair
        """

        loss = loss_fn(pred)
        loss.backward()
        return loss

    def optimize_parameters(self, input, loss_fn):
        self.optimizer.zero_grad()
        pred = self.forward(input)
        loss = self.backward(pred, loss_fn)
        self.optimizer.step()
        return pred, loss

    def set_train_state(self):
        self.model.train()

    def configure_model_2d(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return self.model
    
    def do_lung_lobe_seg_one_epoch(self, epoch, loss_fn):
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, self.args.num_epochs)
        self.logger.info('  lr: {}'.format(lr))
    
        train_loss_list = list()
        train_dice_list = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
        for iter, batch in tqdm(enumerate(self.tr_dataloader)):
            data, seg = batch
            data = data.to(self.device)
            assert data.max() <= 1 and data.min() >= 0
            seg = seg.to(self.device)
            self.optimizer.zero_grad()
            output, loss = self.optimize_parameters(data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())

            # cal 2d dice
            output_softmax = torch.softmax(output, dim=1)
            one_hot_seg = torch.from_numpy(create_one_hot_from_2d_label(seg.cpu().numpy(), self.args.num_classes))
            for c in range(self.args.num_classes):
                pred = output_softmax[:, c]
                # print(pred.size(), one_hot_seg.size())
                train_dice_list[str(c)].append(get_hard_dice(pred.cpu(), one_hot_seg[:, c]))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.logger.info(' Tr loss: {}'.format(mean_tr_loss))
        
        # 对每个类别的dice求平均， 0类为背景
        mean_tr_dice = dict()
        for c in range(self.args.num_classes):
            mean_tr_dice[str(c)] = np.mean(train_dice_list[str(c)])
            self.writer.add_scalar("Train Scalars/Dice_{}".format(c), mean_tr_dice[str(c)], epoch)
            self.logger.info('  Tr class-{} dice: {}'.format(c, mean_tr_dice[str(c)]))
            
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
            data, seg = batch
            data = data.to(self.device)
            seg = seg.to(self.device)
            self.optimizer.zero_grad()
            output, loss = self.optimize_parameters(data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())
            # cal 2d dice
            pred = torch.sigmoid(output)
            # print(pred.size(), seg.size())
            train_dice_list.append(get_hard_dice(pred.cpu().squeeze(),seg.cpu().squeeze()))
            del seg
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

    def do_disc_cup_seg_one_epoch(self, epoch, loss_fn):
        self.logger.info('Epoch {}:'.format(epoch))
        start_epoch = time()
        self.set_train_state()
        lr = adjust_learning_rate(self.optimizer, epoch, self.args.initial_lr, self.args.num_epochs)
        self.logger.info('  lr: {}'.format(lr))

        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        for iter, batch in enumerate(self.ts_dataloader):
            data = torch.from_numpy(batch['data']).to(self.device).to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).to(self.device).to(dtype=torch.float32)
            output, loss = self.optimize_parameters(data, loss_fn)
            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg
        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        self.writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        self.writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        self.writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        self.writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        self.logger.info('  Tr loss: {}\n'
                         '  Tr disc dice: {}; Cup dice: {}'.format(mean_tr_loss, mean_tr_disc_dice, mean_tr_cup_dice))

        time_per_epoch = time() - start_epoch
        self.logger.info('  Durations: {}'.format(time_per_epoch))
        self.writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)





