from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import cv2


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', info=None):

        self.root_path = root_path
        self.rgb_list = sorted(glob(os.path.join(self.root_path, '*_RGB.jpg')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")  
        self.label_list = []         
        if method in ['train']:
            try:
                with open(os.path.join('label_list', info+'.txt')) as f:
                    for i in f:
                        self.label_list.append(i.strip())
            except:
                raise Exception("please give right info")

        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, item):
        rgb_path = self.rgb_list[item]
        gd_path = rgb_path.replace('jpg', 'npy').replace('_RGB', '_GT')
        t_path = rgb_path.replace('_RGB', '_T')
        rgb = Image.open(rgb_path).convert('RGB')
        t = Image.open(t_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            label = (os.path.basename(rgb_path) in self.label_list)
            return self.train_transform(rgb, t, keypoints, label)
        elif self.method == 'val' or self.method == 'test':
            keypoints = np.load(gd_path)
            rgb = self.trans(rgb)
            t = self.trans(t)
            name = os.path.basename(rgb_path).split('.')[0]

            shape = cv2.imread(t_path)[..., ::-1].copy().shape
            gt = keypoints
            k = np.zeros((shape[0], shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < shape[0] and int(gt[i][0]) < shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1

            return rgb, t, len(keypoints), name, k

    def train_transform(self, rgb, t, keypoints, label):
        """random crop image patch and find people in it"""
        wd, ht = rgb.size
        # assert len(keypoints) > 0
        if random.random() > 0.88:
            rgb = rgb.convert('L').convert('RGB')
            t = t.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd * re_size)
        htt = (int)(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            rgb = rgb.resize((wd, ht))
            t = t.resize((wd, ht))
            keypoints = keypoints * re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        rgb = F.crop(rgb, i, j, h, w)
        t = F.crop(t, i, j, h, w)
        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate

        if len(keypoints) > 0:
            if random.random() > 0.5:
                rgb = F.hflip(rgb)
                t = F.hflip(t)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                rgb = F.hflip(rgb)
                t = F.hflip(t)
        return self.trans(rgb), self.trans(t), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size, label
