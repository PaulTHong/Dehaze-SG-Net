import torch
import torch.nn as nn
from torchvision.transforms import Resize
import math
from resnet import new_rf_lw50


class PALayer(nn.Module):
    def __init__(self, channel, mid_channel=None):
        super(PALayer, self).__init__()
        if mid_channel is None:
            mid_channel = channel // 8
        self.pa = nn.Sequential(
            nn.Conv2d(channel, mid_channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, mid_channel=None, return_attention=False):
        super(CALayer, self).__init__()
        if mid_channel is None:
            mid_channel = channel // 8
        self.return_attention = return_attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, mid_channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return y if self.return_attention else x * y


# assert the in-channel and out-channel is same
class ResLayer(nn.Module):
    def __init__(self, channel, out_channel=None):
        super(ResLayer, self).__init__()
        if out_channel is None:
            out_channel = channel
        self.conv1 = nn.Conv2d(channel, out_channel, 1, 1, 0, bias=True)
        self.in1 = nn.InstanceNorm2d(out_channel, affine=True)
        # self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)
        # self.in2 = nn.InstanceNorm2d(out_channel, affine=True)
        # self.conv3 = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=True)
        # self.in3 = nn.InstanceNorm2d(out_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(out_channel, affine=True)

        self.short_cut = nn.Sequential()
        if out_channel is not None:  # or stride != 1
            self.short_cut = nn.Sequential(
                nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, bias=True),
                nn.InstanceNorm2d(out_channel, affine=True)
            )

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.in4(self.conv4(y))
        return self.relu(self.short_cut(x) + y)


class dehaze_net(nn.Module):
    def __init__(self, gps=3, dim=64, kernel_size=3, seg_dim=16, num_classes=40):
        super(dehaze_net, self).__init__()
        self.gps = gps
        self.dim = dim
        self.kernel_size = kernel_size
        self.seg_dim = seg_dim

        self.seg_conv = nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, self.dim, 3, 1, 1, bias=True)  # kernel_size=1
        self.in2 = nn.InstanceNorm2d(self.dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.res1 = ResLayer(self.dim)
        self.res2 = ResLayer(self.dim)
        self.res3 = ResLayer(self.dim)
        # self.res4 = ResLayer(self.dim)
        self.res4 = ResLayer(self.dim + self.seg_dim, self.dim)
        self.res5 = ResLayer(self.dim)
        self.res6 = ResLayer(self.dim)
        self.res7 = ResLayer(self.dim)

        self.ca = CALayer(self.dim * self.gps, mid_channel=self.dim // 16, return_attention=True)  # 8
        self.pa = PALayer(self.dim)

        self.conv3 = nn.Conv2d(self.dim, 32, 3, 1, 1, bias=True)
        self.in3 = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x, seg):
        seg = self.seg_conv(seg)

        y = self.relu(self.in1(self.conv1(x)))
        y1 = self.relu(self.in2(self.conv2(y)))
        y = self.res1(y1)
        y = self.res2(y)
        y_tmp = self.res3(y)
        # resize = Resize(y_tmp.size()[2:], interpolation=2)  # default: PIL.Image.NEAREST
        upsample = nn.Upsample(size=y_tmp.size()[2:], mode='bilinear')
        seg = upsample(seg)  # resize seg as y_tmp

        y = torch.cat([y_tmp, seg], dim=1)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        y = torch.cat([y1, y2, y3], dim=1)
        w = self.ca(y)
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        y = w[:, 0, ::] * y1 + w[:, 1, ::] * y2 + w[:, 2, ::] * y3
        y = self.pa(y)

        y = self.relu(self.in3(self.conv3(y)))
        # y = self.in4(self.conv4(y))
        y = self.conv4(y)
        y = x + y
        out = self.relu(y)

        return out


class multiseg_attention_dehaze_net(nn.Module):
    def __init__(self, gps=3, dim=64, kernel_size=3, seg_dim=16, num_classes=40):
        super(multiseg_attention_dehaze_net, self).__init__()
        self.gps = gps
        self.dim = dim
        self.kernel_size = kernel_size
        self.seg_dim = seg_dim

        if seg_dim != num_classes:
            self.seg_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
                # nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=3, stride=1, padding=1, bias=True),
                # nn.InstanceNorm2d(seg_dim, affine=True),
                nn.ReLU(inplace=True)
            )
            self.seg_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
                # nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=3, stride=1, padding=1, bias=True),
                # nn.InstanceNorm2d(seg_dim, affine=True),
                nn.ReLU(inplace=True)
            )
        else:
            self.seg_conv1 = nn.Identity()
            self.seg_conv2 = nn.Identity()
        self.seg_attention_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=seg_dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32 + self.seg_dim, self.dim, 3, 1, 1, bias=True)  # kernel_size=1
        self.in2 = nn.InstanceNorm2d(self.dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.res1 = ResLayer(self.dim)
        self.res2 = ResLayer(self.dim)
        self.res3 = ResLayer(self.dim)
        # self.res4 = ResLayer(self.dim)
        self.res4 = ResLayer(self.dim + self.seg_dim, self.dim)
        self.res5 = ResLayer(self.dim)
        self.res6 = ResLayer(self.dim)
        self.res7 = ResLayer(self.dim)
        # self.res7 = ResLayer(self.dim + self.seg_dim, self.dim)

        self.ca = CALayer(self.dim * self.gps, mid_channel=self.dim // 16, return_attention=True)  # 8
        self.pa = PALayer(self.dim)

        self.conv3 = nn.Conv2d(self.dim, 32, 3, 1, 1, bias=True)
        self.in3 = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x, seg):
        seg1 = self.seg_conv1(seg)
        seg2 = self.seg_conv2(seg)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear')
        upseg = upsample(seg)
        seg_attention = self.seg_attention_layer(upseg)

        y_combine1 = self.relu(self.in1(self.conv1(x)))
        upseg1 = upsample(seg1)
        y_combine1 = torch.cat([y_combine1, upseg1], dim=1)  # first seg insert (comment this line to abandon)

        y1 = self.relu(self.in2(self.conv2(y_combine1)))
        y = self.res1(y1)
        y = self.res2(y)
        y_combine2 = self.res3(y)
        upseg2 = upsample(seg2)
        y_combine2 = torch.cat([y_combine2, upseg2], dim=1)  # second seg insert
        y2 = self.res4(y_combine2)
        y = self.res5(y2)
        y_combine3 = self.res6(y)
        y3 = self.res7(y_combine3)

        y = torch.cat([y1, y2, y3], dim=1)
        w = self.ca(y)
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        y = w[:, 0, ::] * y1 + w[:, 1, ::] * y2 + w[:, 2, ::] * y3
        y = y * seg_attention

        y = self.relu(self.in3(self.conv3(y)))
        y = self.conv4(y)
        y = x + y
        out = self.relu(y)

        return out


if __name__ == '__main__':
    refinenet = new_rf_lw50(num_classes=40, pretrain=False)
    # net = dehaze_net()
    net = multiseg_attention_dehaze_net()
    print(net)
    num_para = torch.cat([p.view(-1) for p in net.parameters()]).size()
    print('number of network parameters: ', num_para)
    x = torch.randn(1, 3, 256, 256)
    seg = refinenet(x)
    print('seg: ', seg.size())
    y = net(x, seg)
    print(y.size())
    # from torchsummary import summary
    # net = net.cuda()
    # summary(net, ((3, 256, 256)))  # maybe summary must demand only one input of the network











