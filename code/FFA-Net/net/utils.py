import os
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def edge_compute(x):
    x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])

    y = x.new(x.size())
    y.fill_(0)
    y[:, :, 1:] += x_diffx
    y[:, :, :-1] += x_diffx
    y[:, 1:, :] += x_diffy
    y[:, :-1, :] += x_diffy
    y = torch.sum(y, 0, keepdim=True) / 3
    y /= 4
    return y

def tensor_normalize(feat, mean, std):
    '''
    :param feat:  N x C x H x W
    :param mean:  1 dimension as C
    :param std: 1 dimension as C
    :return: normalized N x C x H x W
    '''
    mean = torch.as_tensor(mean, dtype=torch.float32, device=feat.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=feat.device)
    # feat.sub_(mean[:, None, None]).div_(std[:, None, None])
    feat = feat.sub(mean[None, :, None, None]).div(std[None, :, None, None])
    return feat

import torch.nn as nn


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=int(kernel_size / 2.0),
                groups=groups,
                bias=False,
            ),
            batchnorm(out_planes),
        )


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x




