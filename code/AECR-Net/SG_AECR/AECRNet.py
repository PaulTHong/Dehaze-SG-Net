import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
# from deconv import FastDeconv
from dcn import DeformableConv2d


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, in_dim=None):
        super(DehazeBlock, self).__init__()
        if in_dim is None:
            in_dim = dim
            self.shortcut = nn.Sequential()
        elif in_dim != dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.conv1 = conv(in_dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + self.shortcut(x)
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += self.shortcut(x)
        return res


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class Dehaze(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        # self.dcn_block = DCNBlock(256, 256)
        self.dcn_block = DeformableConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):

        x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256]
        out = self.up3(x_up2) # [bs,  3, 256, 256]

        return out


class SGDehaze(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', seg_dim=16, num_classes=40):
        super(SGDehaze, self).__init__()
        self.seg_dim = seg_dim
        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.seg_attention_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=seg_dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3)
        self.SFblock = DehazeBlock(default_conv, ngf * 4, 3, in_dim=ngf*4+self.seg_dim)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        # self.dcn_block = DCNBlock(256, 256)
        self.dcn_block = DeformableConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

        self.pa = PALayer(256)

    def forward(self, input, seg):
        seg_insert = self.seg_conv(seg)
        x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        # x4 = self.block(x3)
        upsample = nn.Upsample(size=x3.size()[2:], mode='bilinear')
        seg_insert = upsample(seg_insert)
        x3 = torch.cat([x3, seg_insert], dim=1)
        x4 = self.SFblock(x3)

        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        # x_dcn2 = self.pa(x_dcn2)
        upseg = upsample(seg)
        seg_attention = self.seg_attention_layer(upseg)
        x_dcn2 = x_dcn2 * seg_attention

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256]
        out = self.up3(x_up2) # [bs,  3, 256, 256]

        return out


if __name__ == '__main__':
    # net = Dehaze(3, 3)
    # print(net)
    # num_para = torch.cat([p.view(-1) for p in net.parameters()]).size()
    # print('number of network parameters: ', num_para)
    # x = torch.randn(1, 3, 256, 256)
    # y = net(x)
    # print(y.size())

    from resnet import new_rf_lw50
    refinenet = new_rf_lw50(num_classes=40, pretrain=False)
    net = SGDehaze(3, 3)
    print(net)
    num_para = torch.cat([p.view(-1) for p in net.parameters()]).size()
    print('number of network parameters: ', num_para)
    x = torch.randn(1, 3, 256, 256)
    seg = refinenet(x)
    print('seg: ', seg.size())
    y = net(x, seg)
    print(y.size())
