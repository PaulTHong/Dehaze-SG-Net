import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, channel_in=None, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        if channel_in is None:
            channel_in = channel_num
            self.shortcut = nn.Sequential()
        elif channel_in != channel_num:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channel_in, channel_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(channel_num, affine=True)
            )
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_in, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(self.shortcut(x) + y)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class GCANet(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=True, seg_dim=16, num_classes=40):
        super(GCANet, self).__init__()
        self.seg_dim = seg_dim

        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(seg_dim, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, channel_in=64+16, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)
        self.ca = CALayer(64 *3, mid_channel=4, return_attention=True)
        self.pa = PALayer(64)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = nn.Conv2d(64, out_c, 1)
        self.only_residual = only_residual

    def forward(self, x, seg):
        seg = self.seg_conv(seg)

        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y1)
        y = self.res2(y)
        # y = self.res3(y)
        # ================
        y_tmp = self.res3(y)
        upsample = nn.Upsample(size=y_tmp.size()[2:], mode='bilinear')
        seg = upsample(seg)
        y = torch.cat([y_tmp, seg], dim=1)
        # ================
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        # gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        # gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        fusion_y = torch.cat([y1, y2, y3], dim=1)
        w = self.ca(fusion_y)
        w = w.view(-1, 3, 64)[:, :, :, None, None]
        gated_y = w[:, 0, ::] * y1 + w[:, 1, ::] * y2 + w[:, 2, ::] * y3
        gated_y = self.pa(gated_y)

        y = F.relu(self.norm4(self.deconv3(gated_y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y


class seg_attention_GCANet(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=True, seg_dim=16, num_classes=40):
        super(seg_attention_GCANet, self).__init__()
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

        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, channel_in=64+16, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)
        self.ca = CALayer(64 *3, mid_channel=4, return_attention=True)
        self.pa = PALayer(64)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = nn.Conv2d(64, out_c, 1)
        self.only_residual = only_residual

    def forward(self, x, seg):
        seg_insert = self.seg_conv(seg)

        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y1)
        y = self.res2(y)
        y_tmp = self.res3(y)
        upsample = nn.Upsample(size=y_tmp.size()[2:], mode='bilinear')
        seg_insert = upsample(seg_insert)
        y = torch.cat([y_tmp, seg_insert], dim=1)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        fusion_y = torch.cat([y1, y2, y3], dim=1)
        w = self.ca(fusion_y)
        w = w.view(-1, 3, 64)[:, :, :, None, None]
        gated_y = w[:, 0, ::] * y1 + w[:, 1, ::] * y2 + w[:, 2, ::] * y3

        upseg = upsample(seg)
        seg_attention = self.seg_attention_layer(upseg)
        gated_y = gated_y * seg_attention

        y = F.relu(self.norm4(self.deconv3(gated_y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y


if __name__ == '__main__':
    from resnet import new_rf_lw50
    refinenet = new_rf_lw50(num_classes=40, pretrain=False)
    # net = GCANet(in_c=4)
    net = seg_attention_GCANet(in_c=4)
    print(net)
    num_para = torch.cat([p.view(-1) for p in net.parameters()]).size()
    print('number of network parameters: ', num_para)
    x = torch.randn(1, 4, 256, 256)
    x0 = x[:, :3, ::]
    seg = refinenet(x0)
    print('seg: ', seg.size())
    y = net(x, seg)
    print(y.size())

