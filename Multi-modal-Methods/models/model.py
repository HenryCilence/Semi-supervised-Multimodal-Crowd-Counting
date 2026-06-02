import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class BL(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = make_layers(cfg['E'])
        self.reg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'), strict=False)
    
    def forward(self, x):
        x = self.features(x[0]) + self.features(x[1])
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(x.size(2)*2, x.size(3)*2), mode='bilinear')        
        x = torch.abs(self.reg(x))
        return x


if __name__ == '__main__':
    model = BL()
    x = torch.randn((1,3,224,224))
    y = model([x, x])
    print(y.shape)