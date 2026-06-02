from PIL import Image
import cv2
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


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


def gen_discrete_map(im_height, im_width, points):
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train', info=None):

        self.root_path = root_path
        self.gt_list = sorted(glob(os.path.join(self.root_path, '*.npy')))
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
            self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
            ])
            self.T_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.492, 0.168, 0.430],
                    std=[0.317, 0.174, 0.191]),
            ])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')
        if self.method == 'train':
            rgb = Image.open(rgb_path).convert('RGB')
            t = Image.open(t_path).convert('RGB')
            keypoints = np.load(gt_path)
            label = (os.path.basename(rgb_path) in self.label_list)
            return self.train_transform(rgb, t, keypoints, label)
        elif self.method == 'val' or self.method == 'test':
            rgb = cv2.imread(rgb_path)[..., ::-1].copy()
            t = cv2.imread(t_path)[..., ::-1].copy()
            keypoints = np.load(gt_path)
            gt = keypoints
            k = np.zeros((t.shape[0], t.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < t.shape[0] and int(gt[i][0]) < t.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            target = k

            rgb = self.RGB_transform(rgb)
            t = self.T_transform(t)
            name = os.path.basename(rgb_path).split('.')[0]
            
            return [rgb, t], target, name

    def train_transform(self, rgb, t, keypoints, label):
        """random crop image patch and find people in it"""
        assert rgb.size == t.size
        wd, ht = rgb.size
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
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 40.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.5)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate

            gt_discrete = gen_discrete_map(h, w, keypoints)
            down_w = w // self.d_ratio
            down_h = h // self.d_ratio
            gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
            gt_discrete = np.expand_dims(gt_discrete, 0)


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
        return [self.RGB_transform(rgb), self.T_transform(t)], torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), torch.from_numpy(gt_discrete.copy()).float(), st_size, label
