import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class IDEM(nn.Module):
    def __init__(self, in_C, out_C):
        super(IDEM, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main1 = BasicConv2d(in_C,out_C,kernel_size=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(rgb + depth)
        return self.fuse_main(self.res_main(feat) + feat)


class Resudiual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resudiual, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        out = x1 * x
        return out


class Tdc3x3_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_1, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_3, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_5, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=4, padding=4)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class Tdc3x3_8(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_8, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=8, padding=8)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x4

class EDFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EDFM, self).__init__()
        self.one = Tdc3x3_1(in_channel, out_channel)
        self.two = Tdc3x3_3(in_channel, out_channel)
        self.three = Tdc3x3_5(in_channel, out_channel)
        self.four = Tdc3x3_8(in_channel,out_channel)
        self.fusion = BasicConv2d(out_channel , out_channel, 1)
        self.fusion1 = BasicConv2d(out_channel * 7, out_channel, 1)

    def forward(self, rgb, rgb_aux):
        x1, x2 = self.one(rgb_aux)
        x3, x4 = self.two(rgb_aux, x1)
        x5, x6 = self.three(rgb_aux, x3)
        x7 = self.four(rgb_aux,x5)
        x2 = x2 * rgb
        x4 = x4 * rgb
        x5 = x5 * rgb
        x2_1 = x2 - x4
        x4_1 = x4 - x6
        x6_1 = x6 - x7
        out = self.fusion1(torch.cat([x2,x4,x6,x7,x2_1,x4_1,x6_1],dim=1))
        out = self.fusion(torch.abs(out - rgb))
        return out

class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            nn.Conv2d(32,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)


class FDM(nn.Module):
    def __init__(self,):
        super(FDM, self).__init__()
        self.basicconv1 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv2 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,32,1),
            nn.ReLU()
        )
        self.basicconv3 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)
        self.basicupsample2 = BasicUpsample(scale_factor=2)
        self.basicupsample1 = BasicUpsample(scale_factor=1)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(160,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            )


    def forward(self,out_data_1,out_data_2,out_data_4,out_data_8,out_data_16):
        out_data_16 = self.basicconv1(out_data_16)
        out_data_16 = self.basicconv3(out_data_16)

        out_data_8 = self.basicconv1(out_data_8)
        out_data_8 = torch.cat([out_data_8,self.upsample1(out_data_16)],dim=1)
        out_data_8 = self.basicconv4(out_data_8)

        out_data_4 = self.basicconv1(out_data_4)
        out_data_4 = torch.cat([out_data_4,self.upsample1(out_data_8)],dim=1)
        out_data_4 = self.basicconv4(out_data_4)

        out_data_2 = self.basicconv2(out_data_2)
        out_data_2 = torch.cat([out_data_2,self.upsample1(out_data_4)],dim=1)
        out_data_2 = self.basicconv4(out_data_2)


        out_data_1 = self.basicconv2(out_data_1)
        out_data_1 = torch.cat([out_data_1,self.upsample1(out_data_2)],dim=1)
        out_data_1 = self.basicconv4(out_data_1)



        out_data_16 = self.basicupsample16(out_data_16)
        out_data_8 = self.basicupsample8(out_data_8)
        out_data_4 = self.basicupsample4(out_data_4)
        out_data_2 = self.basicupsample2(out_data_2)
        out_data_1 = self.basicupsample1(out_data_1)

        out_data = torch.cat([out_data_16,out_data_8,out_data_4,out_data_2,out_data_1],dim=1)

        out_data = self.reg_layer(out_data)


        return torch.abs(out_data)
    

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=True)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def Backbone_VGG_in1(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg16_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), *list(net.children())[0][1:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


def Backbone_VGG_in3(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = vgg16_bn(pretrained=pretrained, progress=True)
    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class DEFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DEFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)

        self.t_trans16 = IDEM(512, 64)
        self.t_trans8 = IDEM(512, 64)
        self.t_trans4 = IDEM(256, 64)
        self.t_trans2 = IDEM(128,32)
        self.t_trans1 = IDEM(64,64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)


        self.selfdc_16 = EDFM(64, 64)
        self.selfdc_8 = EDFM(64, 64)
        self.selfdc_4 = EDFM(64, 64)
        self.selfdc_2 = EDFM(32,32)
        self.selfdc_1 = EDFM(32,32)

        self.fdm = FDM()

    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]
        in_data_1 = self.encoder1(in_data)


        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)


        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)


        in_data_1_aux = self.t_trans1(in_data_1,in_data_1_d)
        in_data_2_aux = self.t_trans2(in_data_2,in_data_2_d)
        in_data_4_aux = self.t_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.t_trans8(in_data_8, in_data_8_d)
        in_data_16_aux = self.t_trans16(in_data_16, in_data_16_d)

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)

        out_data_16 = in_data_16
        out_data_16 = self.upconv16(out_data_16)  # 1024

        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        out_data_8 = self.upconv8(out_data_8)  # 512

        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        out_data_4 = self.upconv4(out_data_4)  # 256

        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        out_data_2 = self.upconv2(out_data_2)  # 64

        out_data_1 = self.upsample_add(self.selfdc_2(out_data_2,in_data_2_aux),in_data_1)
        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.fdm(out_data_1,out_data_2,out_data_4,out_data_8,out_data_16)

        return out_data


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    model = DEFNet(True)
    count_parameters(model)
    x = torch.randn(1,3,224,224)
    depth = torch.randn(1,3,224,224)
    fuse = model([x,depth])
    print(fuse.shape)