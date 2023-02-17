"""
SF: G1 G2                                                                                                                                
SA: G3 + the last one
"""
import torch.nn as nn
import torch

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x, seg=None):
        if seg is None:
            y = self.pa(x)
        else:
            y = self.pa(seg)
        return x * y

class seg_PALayer(nn.Module):
    def __init__(self, channel):
        super(seg_PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x, seg):
        y = self.pa(seg)
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

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, concat_dim=None, PA_dim=None):
        super(Block, self).__init__()
        if concat_dim is None:
            self.conv1=conv(dim, dim, kernel_size, bias=True)
        else:
            self.conv1=conv(dim+concat_dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if concat_dim is not None:
            self.shortcut = nn.Sequential(
                conv(dim+concat_dim, dim, kernel_size=1, bias=True),
            )

        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        if PA_dim is None:
            self.palayer=PALayer(dim)
        else:
            self.palayer=PALayer(PA_dim)

    def forward(self, args):  # (x, seg=None)
        x, seg = args
        res=self.act1(self.conv1(x))
        res=res+self.shortcut(x)
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res, seg)
        res += self.shortcut(x)
        return (res, seg)

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks, concat_dim=None, PA_dim=None):
        super(Group, self).__init__()
        if concat_dim is None:
            modules = [Block(conv, dim, kernel_size, PA_dim=PA_dim) for _ in range(blocks)]
        else:
            modules = [Block(conv, dim, kernel_size, concat_dim=concat_dim, PA_dim=PA_dim)]
            for _ in range(blocks-1):
                modules.append(Block(conv, dim, kernel_size, PA_dim=PA_dim))
        self.gp = nn.Sequential(*modules)
        self.conv = conv(dim, dim, kernel_size)

        self.shortcut = nn.Sequential()
        if concat_dim is not None:
            self.shortcut = nn.Sequential(
                conv(dim+concat_dim, dim, kernel_size=1, bias=True),
            )

    def forward(self, x, seg=None):
        res, _ = self.gp((x, seg))
        res = self.conv(res)

        res += self.shortcut(x)
        return res

class seg_att_FFA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv, seg_dim=16, num_classes=40):
        super(seg_att_FFA, self).__init__()
        self.gps=gps
        self.dim=64
        self.seg_dim = seg_dim
        self.seg_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.seg_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=seg_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size, blocks=blocks, concat_dim=seg_dim, PA_dim=None)
        self.g2= Group(conv, self.dim, kernel_size, blocks=blocks, concat_dim=seg_dim, PA_dim=None)
        self.g3= Group(conv, self.dim, kernel_size, blocks=blocks, concat_dim=None, PA_dim=num_classes)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer = PALayer(num_classes)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1, seg):
        seg1 = self.seg_conv1(seg)
        seg2 = self.seg_conv2(seg)

        x = self.pre(x1)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear')
        seg1 = upsample(seg1)
        seg2 = upsample(seg2)
        upseg = upsample(seg)    

        combine1 = torch.cat([x, seg1], dim=1)
        res1=self.g1(combine1)
        combine2 = torch.cat([res1, seg2], dim=1)
        res2=self.g2(combine2)
        res3=self.g3(res2, upseg)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3

        out = self.palayer(out, upseg)

        x=self.post(out)
        return x + x1


if __name__ == "__main__":
    net = seg_att_FFA(gps=3, blocks=19)
    print(net)
    from resnet import new_rf_lw50
    refinenet = new_rf_lw50(num_classes=40, pretrain=False)
    x = torch.randn(1, 3, 256, 256)
    seg = refinenet(x)
    print('seg: ', seg.size())
    y = net(x, seg)
    print(y.size())
