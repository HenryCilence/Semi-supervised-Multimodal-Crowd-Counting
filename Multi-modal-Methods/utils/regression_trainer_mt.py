from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.multi_crowd_semi import Crowd
from losses.bay_loss import BL
from losses.evaluation import eval_game

from models.bm_mt import MeanTeacherWrapper
from models.bm import BM, BL as BL_Model
# from models.iadm import fusion_model as iadm
# from models.mc3net import Net as mc3net
# from models.defnet import DEFNet
# from models.CAGNet.CAGNet import MAINet as cagnet


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    gt_discretes = torch.stack(transposed_batch[3], 0)
    st_sizes = torch.FloatTensor(transposed_batch[4])
    label = transposed_batch[5]
    return images, points, targets, gt_discretes, st_sizes, label


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {
            'train': Crowd(args.train_dir, args.crop_size, args.downsample_ratio, args.is_gray, 'train', args.info),
            'val': Crowd(args.val_dir, args.crop_size, args.downsample_ratio, args.is_gray, 'val', args.info)
        }
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], collate_fn=train_collate, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers * self.device_count, pin_memory=True),
            'val': DataLoader(self.datasets['val'], collate_fn=default_collate, batch_size=1, 
                              shuffle=False, num_workers=args.num_workers * self.device_count, pin_memory=False),
        }
        self.model = MeanTeacherWrapper(BL_Model())
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)

        self.criterion = BL(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background, self.device)
        
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_game = [np.inf, np.inf, np.inf, np.inf]
        self.best_mse = [np.inf, np.inf, np.inf, np.inf]
        self.best_epoch = -1
        self.save_all = args.save_all
        self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.model.set_epoch(epoch)
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_s_loss = AverageMeter()
        epoch_c_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, gt_discrete, st_sizes, label) in enumerate(self.dataloaders['train']):

            if not (self.epoch >= self.args.unlabel_start or label[0]):
                continue

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            gt_discrete = gt_discrete.to(self.device)

            with torch.set_grad_enabled(True):
                if label[0]:
                    N = inputs[0].size(0)
                    # BL
                    metrics = self.model.training_step((inputs, targets, st_sizes, points), inputs, self.optimizer, self.criterion, isLabel=True)
                    outputs, s_loss, c_loss, loss = metrics["preds_l"], metrics["supervised_loss"], metrics["consistency_loss"], metrics["total_loss"]
                    epoch_loss.update(loss, N)
                    epoch_s_loss.update(s_loss, N)
                    epoch_c_loss.update(c_loss, N)

                    pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                    res = pre_count - gd_count
                    epoch_mse.update(np.mean(res * res), N)
                    epoch_mae.update(np.mean(abs(res)), N)

                else:
                    N = inputs[0].size(0)
                    # BL
                    metrics = self.model.training_step((inputs, targets, st_sizes, points), inputs, self.optimizer, self.criterion, isLabel=False)
                    c_loss, loss = metrics["consistency_loss"], metrics["total_loss"]
                    epoch_loss.update(loss, N)
                    epoch_c_loss.update(c_loss, N)                    

        logging.info('Epoch {} Train, Loss: {:.2f}, Sup Loss: {:.2f}, Const Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_s_loss.get_avg(), epoch_c_loss.get_avg(), np.sqrt(epoch_mse.get_avg()),\
                              epoch_mae.get_avg(), time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        # Iterate over data.
        for inputs, target, name in self.dataloaders['val']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs[0].shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                # outputs = self.model(inputs)[0:2]
                outputs = self.model.forward_supervised(inputs, None, None, None, None)
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error

        N = len(self.dataloaders['val'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        logging.info('Epoch {} Val, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f}, Cost {time:.1f} s'
                     .format(self.epoch, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], time=time.time()-epoch_start)
                     )

        model_state_dic = self.model.state_dict()
        if game[0] < self.best_game[0]:
            self.best_game = game
            self.best_mse = mse
            self.best_epoch = self.epoch
            logging.info("*** Save Best " \
            "GAME0 {:.2f} GAME1 {:.2f} GAME2 {:.2f} GAME3 {:.2f} MSE {:.2f} Model Epoch {}"
            .format(self.best_game[0], self.best_game[1], self.best_game[2], self.best_game[3], self.best_mse[0], self.best_epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        else:
            logging.info("Best " \
            "GAME0 {:.2f} GAME1 {:.2f} GAME2 {:.2f} GAME3 {:.2f} MSE {:.2f} Epoch {}"
            .format(self.best_game[0], self.best_game[1], self.best_game[2], self.best_game[3], self.best_mse[0], self.best_epoch))
