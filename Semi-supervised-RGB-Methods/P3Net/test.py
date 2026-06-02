import torch
import os
import numpy as np
from datasets.multi_crowd_semi import Crowd
from models.model import vgg19_trans
import argparse
from glob import glob
from torch.utils.data.dataloader import default_collate
from losses.count import de_forward
from losses.evaluation import eval_game

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--test-dir', default='/media/dataset/person_dataset/multi-modal_crowd_counting/RGBT-CC/test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/home/menghaoliang/code/count3/SSMMCC-OpenSource/Semi-supervised-RGB-Methods/DACount/check/0602-194908',
                        help='model directory')
    parser.add_argument('--device', default='2', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(args.test_dir, 224, 8, False, 'val', None)
    dataloader = torch.utils.data.DataLoader(datasets, collate_fn=default_collate, batch_size=1, 
                                             shuffle=False, num_workers=8, pin_memory=False)

    model_list = sorted(glob(os.path.join(args.save_dir, '*.pth')))
    device = torch.device('cuda')
    model = vgg19_trans()
    model.to(device)
    model.eval()
    log_list = []
    idx_count2 = torch.tensor([0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
                              0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833,
                              0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232,
                              0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038,
                              0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938,
                              1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832])
    idx_count2 = idx_count2.unsqueeze(1).to(device)
    idx_count = torch.tensor(
        [0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603,
         0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504,
         0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597,
         0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944,
         1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729])  # 25
    idx_count = idx_count.unsqueeze(1).to(device)

    for model_path in model_list:
        epoch_minus = []
        model.load_state_dict(torch.load(model_path, device))
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        # Iterate over data.
        for inputs, target, name in dataloader:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
            else:
                inputs = inputs.to(device)
            # inputs are images with different sizes
            b, c, h, w = inputs[0].shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                bay_outputs = de_forward(outputs[0], outputs[1], idx_count, idx_count2)
                for L in range(4):
                    abs_error, square_error = eval_game(bay_outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error

        N = len(dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        print('GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f}'
                     .format(game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0]))
