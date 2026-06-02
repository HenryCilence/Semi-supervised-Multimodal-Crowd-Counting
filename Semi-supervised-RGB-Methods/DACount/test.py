import torch
import os
import numpy as np
from datasets.crowd_semi import Crowd
from models.vgg import vgg19_trans
import argparse
import math
from glob import glob
from datetime import datetime
import cv2

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--test-dir', default='/media/dataset/person_dataset/multi-modal_crowd_counting/RGBT-CC/test',
                        help='testing data directory')
    parser.add_argument('--save-dir', default='/home/home/menghaoliang/code/count3/SSMMCC-OpenSource/Semi-supervised-RGB-Methods/DACount/check/0602-190012',
                        help='model directory')
    parser.add_argument('--device', default='2', help='assign device')
    args = parser.parse_args()
    return args


def eval_game(output, target, L=0):
    output = output[0][0].cpu().numpy()
    target = target[0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio)
    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]
            target_block = target[i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]

            abs_error += abs(output_block.sum() - target_block.sum().float())
            square_error += (output_block.sum() - target_block.sum().float()).pow(2)

    return abs_error, square_error


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(args.test_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    model_list = sorted(glob(os.path.join(args.save_dir, '*.pth')))
    if len(model_list) > 3:
        model_list = model_list[-3:]
    device = torch.device('cuda')
    model = vgg19_trans()
    model.to(device)
    model.eval()
    log_list = []

    for model_path in model_list:
        epoch_minus = []
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        model.load_state_dict(torch.load(model_path, device))
        i = 1
        for rgb, t, count, name, target in dataloader:
            rgb = rgb.to(device)
            t = t.to(device)
            b, c, h, w = rgb.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            with torch.set_grad_enabled(False):
                outputs = model([rgb, t])[0]
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                res = count[0].item() - torch.sum(outputs).item()
                print(i, name, count[0].item(), res)
                i += 1
                epoch_minus.append(res)

        N = len(dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f}'\
            .format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0])
        print(log_str)
        
        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'model_name {}, mae {}, mse {}'.format(os.path.basename(model_path), mae, mse)
        log_list.append(log_str)
        print(log_str)
