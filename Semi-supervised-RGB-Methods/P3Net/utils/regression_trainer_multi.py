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
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.multi_crowd_semi import Crowd
from losses.post import Post_Prob
from losses.count import unsupervised_loss, supervised_loss, de_forward
from losses.evaluation import eval_game
from math import ceil

from models.model import vgg19_trans



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
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = transposed_batch[4]
    return images, points, targets, st_sizes, label


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
                                 shuffle=True, num_workers=args.num_workers*self.device_count, pin_memory=True),
            'val': DataLoader(self.datasets['val'], collate_fn=default_collate, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers*self.device_count, pin_memory=False)
        } 
        self.model = vgg19_trans()
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

        self.post_prob = Post_Prob(args.sigma, args.crop_size,
                                   args.downsample_ratio, args.background_ratio,
                                   args.use_background, self.device)
        self.criterion_mse = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_game = [np.inf, np.inf, np.inf, np.inf]
        self.best_mse = [np.inf, np.inf, np.inf, np.inf]
        self.best_epoch = -1
        self.save_all = args.save_all
        self.best_count = 0
        idx_count2 = torch.tensor(
            [0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
             0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904,
             0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514,
             0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761,
             1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832])
        self.idx_count2 = idx_count2.unsqueeze(1).to(self.device)
        label_count2 = torch.tensor(
            [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765,
             0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087,
             0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755,
             0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993,
             1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832])
        self.label_count2 = label_count2.unsqueeze(1).to(self.device)
        idx_count = torch.tensor(
            [0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905,
             0.03335742127399603, 0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907,
             0.14175821026385504, 0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885,
             0.40938709885499597, 0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628,
             1.1967563985336944, 1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729])  # 25
        self.idx_count = idx_count.unsqueeze(1).to(self.device)
        label_count = torch.tensor(
            [0.00016, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
             0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904,
             0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514,
             0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761,
             1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676])  # 24
        self.label_count = label_count.unsqueeze(1).to(self.device)

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_unsup_loss = AverageMeter()
        epoch_sup_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes, label) in enumerate(self.dataloaders['train']):

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

            with torch.set_grad_enabled(True):
                N = inputs[0].size(0)
                N_s = 0
                N_us = 0

                outputs = self.model(inputs)
                if label[0]:
                    unsup_loss = torch.tensor(0.0, device=self.device)
                    sup_loss, _ = supervised_loss(outputs[0], outputs[1], self.label_count, self.label_count2, points, st_sizes, self.post_prob)
                    loss = unsup_loss + sup_loss
                    N_s += 1
                else:
                    sup_loss = torch.tensor(0.0, device=self.device)
                    unsup_loss = unsupervised_loss(outputs[0], outputs[1], self.idx_count, self.idx_count2, thresh=0.5, beta=0.1)
                    loss = sup_loss + unsup_loss
                    N_us += 1

                bay_outputs = de_forward(outputs[0], outputs[1], self.idx_count, self.idx_count2)

                epoch_loss.update(loss.item(), N)
                epoch_unsup_loss.update(unsup_loss.item(), N_us)
                epoch_sup_loss.update(sup_loss.item(), N_s)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_count = torch.sum(bay_outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count

                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, SupLoss: {:.2f}, UnsupLoss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_sup_loss.get_avg(), epoch_unsup_loss.get_avg()*100, 
                             np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), time.time() - epoch_start))
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
                outputs = self.model(inputs)
                bay_outputs = de_forward(outputs[0], outputs[1], self.idx_count, self.idx_count2)
                for L in range(4):
                    abs_error, square_error = eval_game(bay_outputs, target, L)
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
