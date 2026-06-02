import torch
import os
import argparse
import numpy as np
from datasets.multi_crowd_semi import Crowd
from losses.evaluation import eval_game
from torch.utils.data.dataloader import default_collate

# from models.model import BL as BL_Model
from models.bm import BM, BL as BL_Model
# from models.iadm import fusion_model as iadm
# from models.mc3net import Net as mc3net
# from models.defnet import DEFNet
# from models.CAGNet.CAGNet import MAINet as cagnet

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--test-dir', default=r"",
                    help='testing data directory')
parser.add_argument('--save-dir', default=r"",
                    help='model directory')
parser.add_argument('--device', default='2', help='gpu device')
args = parser.parse_args()


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':

    datasets = Crowd(args.test_dir, 224, 8, False, 'test', None)
    dataloader = torch.utils.data.DataLoader(datasets, collate_fn=default_collate, batch_size=1, 
                                             shuffle=False, num_workers=8, pin_memory=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = BM()
    print('#Param.: {}'.format(count_parameters(model)))
    model.to(device)
    model_path = args.save_dir
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    i = 0
    epoch_minus = []

    for inputs, target, name in dataloader:
        i += 1
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            epoch_minus.append(torch.sum(outputs).item() - torch.sum(target).item())
            print(i, torch.sum(target).item(), torch.sum(outputs).item(),
                  torch.sum(outputs).item() - torch.sum(target).item())

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f}'. \
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0])

    print(log_str)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
