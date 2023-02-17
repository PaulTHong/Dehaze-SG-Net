import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


class NonequalContrastLoss(nn.Module):
    def __init__(self, ablation=False):
        super(NonequalContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, teacher, student, neg):
        teacher_vgg, student_vgg, neg_vgg = self.vgg(teacher), self.vgg(student), self.vgg(neg)

        loss = 0

        # teacher_vgg: [5, N, n_feats, w, h]
        for i in range(len(teacher_vgg)):
            neg_i = neg_vgg[i].unsqueeze(0)  # [M, n_feats, w, h], M <= N, e.g. M=10, N=16
            neg_i = neg_i.repeat(student_vgg[i].shape[0], 1, 1, 1, 1)  # [N, M, n_feats, w, h]
            neg_i = neg_i.permute(1, 0, 2, 3, 4)  # [M, N, n_feats, w, h]

            # teacher_vgg[i], student_vgg[i]: [N, n_feats, w, h]
            d_ts = self.l1(teacher_vgg[i], student_vgg[i].detach())
            if not self.ab:
                # === way 1, no error, raise warning since the broadcast mechanism, but save RAM!
                d_sn = self.l1(teacher_vgg[i], neg_i.detach())  # [M, N, n_feats, w, h]

                # === way 2, match the dimension to exactly the same firstly
                # tea = teacher_vgg[i]
                # tea = tea.repeat(neg_i.shape[0], 1, 1, 1, 1)  # [N, n_feats, w, h] -> [M, N, n_feats, w, h]
                # d_sn = self.l1(tea, neg_i.detach())

                contrastive = d_ts / (d_sn + 1e-7)
            else:
                contrastive = d_ts

            loss += self.weights[i] * contrastive

        return loss


class JointCRLoss(nn.Module):
    def __init__(self, contrast_w=0, neg_num=0):  # contrast_w=0.1, neg_num=10
        super(JointCRLoss, self).__init__()
        self.neg_num = neg_num
        self.l1_loss = nn.L1Loss()
        # self.contrast_loss = ContrastLoss()
        self.contrast_w = contrast_w

    def forward(self, pred, clear, hazy):
        # neg = mindspore.numpy.flip(hazy, 0)
        neg = hazy
        if self.neg_num == 0:  # 1 positive vs 1 negative from the same sample
            self.contrast_loss = ContrastLoss()
        else:
            neg = neg[:self.neg_num, :, :, :]
            self.contrast_loss = NonequalContrastLoss()
        l1_loss = self.l1_loss(pred, clear)
        contras_loss = self.contrast_loss(pred, clear, neg)
        loss = l1_loss + self.contrast_w * contras_loss
        return loss
