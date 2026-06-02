from thop import profile
import torch

# from models.model import VGG_Trans
# from bm import BM, BL as Bayes
# from iadm import fusion_model as iadm
from mc3net import Net as mc3net
# from defnet import DEFNet
# from CAGNet.CAGNet import MAINet as cagnet

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params/1e6} M")
    return total_params

if __name__ == "__main__":
    model = mc3net().cuda()
    input_tensor = [torch.randn((1,3,224,224)).cuda(), torch.randn((1,3,224,224)).cuda()]
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")
    count_parameters(model)