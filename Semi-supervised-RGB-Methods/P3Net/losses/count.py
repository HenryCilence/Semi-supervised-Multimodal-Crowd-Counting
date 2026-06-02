import torch
try:
    from post import Post_Prob
except:
    from .post import Post_Prob

mse_loss = torch.nn.MSELoss(reduction='sum')
# post_prob = Post_Prob(sigma=8.0, c_size=224, stride=8, background_ratio=0.15, use_background=True, device="cuda")


def unsupervised_loss(outputs, outputs2, idx_count, idx_count2, thresh=0.5, beta=0.01):
    outputs = torch.softmax(outputs, dim=1)
    outputs2 = torch.softmax(outputs2, dim=1)
    cross_outputs = outputs.flatten(2)[0].T
    cross_outputs2 = outputs2.flatten(2)[0].T
    bay_outputs1 = torch.einsum("bixy,io->boxy", outputs, idx_count)
    bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, idx_count2)
    mask1 = torch.max(cross_outputs, dim=1)[0] > thresh
    mask2 = torch.max(cross_outputs2, dim=1)[0] > thresh
    mask = (mask1 & mask2).detach()
    loss = mse_loss(bay_outputs1.flatten()[mask], bay_outputs2.flatten()[mask])
    return loss * beta


def supervised_loss(outputs, outputs2, label_count, label_count2, points, st_sizes, post_prob):
    outputs = torch.softmax(outputs, dim=1)
    outputs2 = torch.softmax(outputs2, dim=1)
    cross_outputs = outputs.flatten(2)[0]
    cross_outputs2 = outputs2.flatten(2)[0]

    cross_outputs = cross_outputs.T
    cross_outputs2 = cross_outputs2.T

    prob_list, gau = post_prob(points, st_sizes)
    _gau = gau
    gaum = gau.flatten().unsqueeze(0) - label_count
    gaum2 = gau.flatten().unsqueeze(0) - label_count2

    gaum = torch.sum(gaum > 0, dim=0)
    gaum2 = torch.sum(gaum2 > 0, dim=0)

    gau = gaum.long()
    gau2 = gaum2.long()
    one_hot = torch.zeros_like(cross_outputs).scatter_(1, gau.unsqueeze(-1), 1)
    one_hot = torch.cumsum(one_hot, dim=1)
    cross_outputs = torch.cumsum(cross_outputs, dim=1)
    loss = mse_loss(one_hot, cross_outputs)
    one_hot2 = torch.zeros_like(cross_outputs2).scatter_(1, gau2.unsqueeze(-1), 1)
    one_hot2 = torch.cumsum(one_hot2, dim=1)
    cross_outputs2 = torch.cumsum(cross_outputs2, dim=1)
    loss += mse_loss(one_hot2, cross_outputs2)

    return loss, _gau


def de_forward(outputs, outputs2, idx_count, idx_count2):
    outputs = torch.softmax(outputs, dim=1)
    outputs2 = torch.softmax(outputs2, dim=1)
    bay_outputs1 = torch.einsum("bixy,io->boxy", outputs, idx_count)
    bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, idx_count2)
    entro = torch.max(outputs, dim=1)[0]
    entro2 = torch.max(outputs2, dim=1)[0]
    mask = entro / (entro + entro2)
    mask = mask.unsqueeze(1).float()
    bay_outputs = (bay_outputs1 * mask + bay_outputs2 * (1 - mask))
    return bay_outputs


if __name__ == "__main__":
    idx_count = torch.tensor(
        [0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603,
         0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504,
         0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597,
         0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944,
         1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729])  # 25
    idx_count = idx_count.unsqueeze(1).to("cuda")
    idx_count2 = torch.tensor(
        [0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
         0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904,
         0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514,
         0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761,
         1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832])
    idx_count2 = idx_count2.unsqueeze(1).to("cuda")

    label_count = torch.tensor(
        [0.00016, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
         0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904,
         0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514,
         0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761,
         1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676])  # 24
    label_count = label_count.unsqueeze(1).to("cuda")
    label_count2 = torch.tensor(
        [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765,
         0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087,
         0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755,
         0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993,
         1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832])
    label_count2 = label_count2.unsqueeze(1).to("cuda")

    post_prob = Post_Prob(sigma=8.0, c_size=224, stride=8, background_ratio=0.15, use_background=True, device="cuda")

    outputs = torch.randn((1, 25, 28, 28)).to("cuda")
    outputs2 = torch.randn((1, 25, 28, 28)).to("cuda")

    points = [torch.randn((10, 2)).to("cuda")]
    st_sizes = [224]  # >= crop_size

    loss_us = unsupervised_loss(outputs, outputs2, idx_count, idx_count2)
    loss_s = supervised_loss(outputs, outputs2, label_count, label_count2, points, st_sizes, post_prob)
    print(loss_s, loss_us)
