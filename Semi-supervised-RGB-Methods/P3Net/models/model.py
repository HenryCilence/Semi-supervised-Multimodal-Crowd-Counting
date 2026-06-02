import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import math
try:
    from .transformer import TransformerDecoder, TransformerDecoderLayer
except:
    from transformer import TransformerDecoder, TransformerDecoderLayer



__all__ = ['vgg19', 'vgg19_mask', 'vgg19_mask_up', 'vgg_19_3D', 'vgg19_z', 'vgg19_y', 'vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}


class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features

        d_model = 512
        nhead = 4
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.decoder1 = TransformerDecoder(decoder_layer, num_layers, if_norm)
        self.decoder2 = TransformerDecoder(decoder_layer, num_layers, if_norm)

        self.count_query1 = nn.Parameter(torch.zeros(25, 1, 512, dtype=torch.float32))
        self.count_query2 = nn.Parameter(torch.zeros(25, 1, 512, dtype=torch.float32))

        self.g1 = nn.Conv2d(512, 512, kernel_size=1)
        self.g2 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        r, t = x
        b, c, h, w = r.shape
        rh = int(h) // 8
        rw = int(w) // 8
        r, t = self.features(r), self.features(t)
        x = r * torch.sigmoid(self.g1(r)) + t * torch.sigmoid(self.g2(t))
        b, c, h, w = x.shape
        z = x.flatten(2).permute(2, 0, 1)

        query1 = self.decoder1(self.count_query1, z)
        query2 = self.decoder2(self.count_query2, z)

        query1 = query1.permute(2, 3, 1, 0).view(c, 25)
        query2 = query2.permute(2, 3, 1, 0).view(c, 25)

        z = F.interpolate(x, size=(rh, rw))
        z1 = torch.einsum("bixy,io->boxy", z, query1)
        z2 = torch.einsum("bixy,io->boxy", z, query2)

        return z1, z2
    

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19_trans():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


if __name__ == "__main__":
    model = vgg19_trans()
    r = torch.randn((1,3,224,224))
    t = torch.randn((1,3,224,224))
    y1, y2 = model([r, t])
    print(y1.shape)