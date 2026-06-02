import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import logging


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNeXtt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=6, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnextt_small(pretrained=False, **kwargs):
    model = ConvNeXtt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


class APL(nn.Module):
    def __init__(self, channel):
        super(APL, self).__init__()

        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fus = nn.Sequential(
            nn.Conv2d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_size = x.size()[2:]
        b0 = F.interpolate(self.b0(x), x_size, mode='bilinear', align_corners=True)
        b1 = F.interpolate(self.b1(x), x_size, mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((b0, b1, x), 1))
        return out


class block1(nn.Module):
    def __init__(self, channels):
        super(block1, self).__init__()

        # self.cat = Channel_Att(channels)
        self.cat = CAT(channels)
        self.sat = SAT()

        self.d = CrossD(channels)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.fusion_total = BasicConv2d(channels, channels, 3, 1, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, infr, inft):
        infe = self.conv1(infr)
        inft = self.conv1(inft)
        inftotal = self.fusion_total(torch.mul(infe, inft) + torch.mul(inft, infe))
        infto = self.cat(inftotal) * inftotal
        inftod = self.d(inftotal)
        infin = self.conv(torch.cat((inftod, infto), 1))

        inres = self.sat(infin) * infin

        return inres


class block2(nn.Module):
    def __init__(self, channels):
        super(block2, self).__init__()

        self.mpltf = APL(channels)
        self.mpltrt = APL(channels)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
                      groups=channels, bias=False)
        )

    def forward(self, fea, feart):
        infea = self.mpltf(fea)
        infeart = self.mpltrt(feart)

        sinrt = self.conv1(infeart - infea)

        sinrtgate = torch.sigmoid(sinrt)

        new_sinrt = (infeart - infea) * sinrtgate

        fin = self.conv1(infeart + new_sinrt)

        return fin, self.conv1(new_sinrt)


class block3(nn.Module):
    def __init__(self, channels):
        super(block3, self).__init__()

        self.naccr = REH(channels)

    def forward(self, rgb, t, nrgbt):
        nrgb, nt = self.naccr(rgb, t, nrgbt)

        return nrgb, nt


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class CrossD(nn.Module):
    def __init__(self, channel):
        super(CrossD, self).__init__()

        self.convk3d3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3, dilation=3,
                      groups=channel, bias=False)
        self.convk3d5 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=5, dilation=5,
                      groups=channel, bias=False)
        self.convk3d7 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=7, dilation=7,
                      groups=channel, bias=False)
        self.convk3d9 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=9, dilation=9,
                      groups=channel, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 2, kernel_size=1),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(True),
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        c2 = self.convk3d3(x)
        c3 = self.convk3d5(x + c2)
        c4 = self.convk3d7(x + c3)
        c5 = self.convk3d9(x + c4)
        c2_2 = self.convk3d9(c2)
        c3_3 = self.convk3d7(c2_2 + c3)
        c4_4 = self.convk3d5(c3_3 + c4)
        c5_5 = self.convk3d3(c4_4 + c5)
        c2_22 = c2_2 + c3_3 + c4_4 + c5_5
        c3_33 = c3_3 + c4_4 + c5_5 + c2_2
        c4_44 = c4_4 + c5_5 + c3_3 + c2_2
        c5_55 = c5_5 + c4_4 + c2_2 + c3_3

        res = torch.cat((c2_22, c3_33, c4_44, c5_55), 1)

        return self.conv(res)


class CAT(nn.Module):
    def __init__(self, in_channels):
        super(CAT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SAT(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAT, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class REH(nn.Module):
    def __init__(self, channel):
        super(REH, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Conv2d(channel, 1, 3, padding=1),
        )
        self.channel = channel

    def forward(self, xr, xt, y):
        # print(y.shape)
        a = torch.sigmoid(-y)
        xr = a.expand(-1, self.channel, -1, -1).mul(xr)
        yr = self.conv2(xr)
        xt = a.expand(-1, self.channel, -1, -1).mul(xt)
        yt = self.conv2(xt)

        return yr, yt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.vgg_r = convnext_small(True)
        self.vgg_t = convnext_small(True)
        self.vgg_rt = convnextt_small(False)

        path = r"./convnext_small_1k_224_ema.pth"
        load_partial_state_dict(self.vgg_r, torch.load(path)['model'])
        load_partial_state_dict(self.vgg_t, torch.load(path)['model'])
        load_partial_state_dict(self.vgg_rt, torch.load(path)['model'])

        self.layer0_r = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_r = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_r = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_r = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_t = nn.Sequential(self.vgg_t.downsample_layers[0], self.vgg_t.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_t = nn.Sequential(self.vgg_t.downsample_layers[1], self.vgg_t.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_t = nn.Sequential(self.vgg_t.downsample_layers[2], self.vgg_t.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_t = nn.Sequential(self.vgg_t.downsample_layers[3], self.vgg_t.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_rt = nn.Sequential(self.vgg_rt.downsample_layers[0], self.vgg_rt.stages[0],
                                       LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_rt = nn.Sequential(self.vgg_rt.downsample_layers[1], self.vgg_rt.stages[1],
                                       LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_rt = nn.Sequential(self.vgg_rt.downsample_layers[2], self.vgg_rt.stages[2],
                                       LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_rt = nn.Sequential(self.vgg_rt.downsample_layers[3], self.vgg_rt.stages[3],
                                       LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.fusion11 = block1(96)
        self.fusion12 = block1(192)
        self.fusion13 = block1(384)
        self.fusion14 = block1(768)

        self.fusion21 = block2(96)
        self.fusion22 = block2(192)
        self.fusion23 = block2(384)
        self.fusion24 = block2(768)

        self.fusion31 = block3(96)
        self.fusion32 = block3(192)
        self.fusion33 = block3(384)
        self.fusion34 = block3(768)

        self.upsam = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.reg_layer1 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, RGBT):
        image = RGBT[0]
        t = RGBT[1]
        rgbt = torch.cat((image, image), 1)

        conv1_vgg_r = self.layer0_r(image)
        conv1_vgg_t = self.layer0_t(t)
        conv1_vgg_rt = self.layer0_rt(rgbt)

        # print(conv1_vgg_r.shape,conv1_vgg_t.shape)
        f1 = self.fusion11(conv1_vgg_r, conv1_vgg_t)
        ff1, g1 = self.fusion21(f1, conv1_vgg_rt)
        rgb1, t1 = self.fusion31(conv1_vgg_r, conv1_vgg_t, g1)
        resr1 = rgb1 + conv1_vgg_r
        rest1 = t1 + conv1_vgg_t
        resrt1 = ff1 + conv1_vgg_rt

        conv2_vgg_r = self.layer1_r(resr1)
        conv2_vgg_t = self.layer1_t(rest1)
        conv2_vgg_rt = self.layer1_r(resrt1)

        f2 = self.fusion12(conv2_vgg_r, conv2_vgg_t)
        ff2, g2 = self.fusion22(f2, conv2_vgg_rt)
        rgb2, t2 = self.fusion32(conv2_vgg_r, conv2_vgg_t, g2)
        resr2 = rgb2 + conv2_vgg_r
        rest2 = t2 + conv2_vgg_t
        resrt2 = ff2 + conv2_vgg_rt

        conv3_vgg_r = self.layer2_r(resr2)
        conv3_vgg_t = self.layer2_t(rest2)
        conv3_vgg_rt = self.layer2_r(resrt2)

        f3 = self.fusion13(conv3_vgg_r, conv3_vgg_t)
        ff3, g3 = self.fusion23(f3, conv3_vgg_rt)
        rgb3, t3 = self.fusion33(conv3_vgg_r, conv3_vgg_t, g3)
        resr3 = rgb3 + conv3_vgg_r
        rest3 = t3 + conv3_vgg_t
        resrt3 = ff3 + conv3_vgg_rt

        conv4_vgg_r = self.layer3_r(resr3)
        conv4_vgg_t = self.layer3_t(rest3)
        conv4_vgg_rt = self.layer3_r(resrt3)

        f4 = self.fusion14(conv4_vgg_r, conv4_vgg_t)
        ff4, g4 = self.fusion24(f4, conv4_vgg_rt)
        rgb4, t4 = self.fusion34(conv4_vgg_r, conv4_vgg_t, g4)
        resr4 = rgb4 + conv4_vgg_r
        rest4 = t4 + conv4_vgg_t
        resrt4 = ff4 + conv4_vgg_rt
        fin = resr4 + rest4 + resrt4

        fino = self.upsam(fin)
        fin = self.reg_layer1(fino)
        # print(fin)
        # rgb4 = F.interpolate(rgb4, (fin.size()[2], fin.size()[3]))
        return fin


def load_partial_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
            else:
                logging.info(f"Skipping parameter {name}, shape mismatch: {model_state_dict[name].shape} vs {param.shape}")
        else:
            logging.info(f"Skipping parameter {name}, not found in model state dict.")
    model.load_state_dict(model_state_dict)


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 3, 480, 640)
    # rgb = torch.randn(1, 3, 256, 256)
    # depth = torch.randn(1, 3, 256, 256)
    # a = torch.randn(1, 3, 32, 32)
    # b = F.interpolate(rgb, (a.size()[2], a.size()[3]))
    # print(b.shape)
    model = Net()
    res = model([rgb, depth])
    print(res.shape)
    # from models.ModelTwo.One.canshu.utils import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Net()
    #     flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    #     print('Flops:' + flops)
    #     print('Params:' + params)
    #
    # compute_speed(net, input_size=(1, 3, 480, 640), iteration = 500)