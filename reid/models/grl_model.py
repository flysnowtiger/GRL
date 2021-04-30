
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

import torchvision
import numpy as np

from .basebranch import Backbone

__all__ = ['resnet50_grl']


# ===================
#   Initialization
# ===================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               # stride=stride,
                               # padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = x1 + x2
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class TRLBlock(nn.Module):
    def __init__(self, feat_num):
        super(TRLBlock, self).__init__()
        self.feat_num = feat_num
        self.feat_num_half = int(feat_num / 2)

        self.uncorr_memo_forward = BasicBlock(2048, 512)

        self.forward_f1 = nn.Sequential(nn.Conv2d(2048, 2048, 1, 1),
                                        nn.ReLU(),
                                        )

        self.forward_f2 = nn.Sequential(nn.Conv2d(2048, 2048, 1, 1),
                                        nn.ReLU(),
                                        )

        self.channel_atte_foreward_corr = nn.Sequential(
                            nn.Linear(2048, 2048 // 16, bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(2048// 16, 2048, bias=False),
                            nn.Sigmoid(),
        )


        ####################################################3

        self.uncorr_memo_backward = BasicBlock(2048, 512)

        self.backward_f1 = nn.Sequential(nn.Conv2d(2048, 2048, 1, 1),
                                         nn.ReLU(),
                                         )

        self.backward_f2 = nn.Sequential(nn.Conv2d(2048, 2048, 1, 1),
                                         nn.ReLU(),
                                         )

        self.channel_atte_backward_corr = nn.Sequential(
            nn.Linear(2048, 2048 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048// 16 , 2048, bias=False),
            nn.Sigmoid(),
        )


    def forward(self, x_uncorr, x_corr):
        b, t, c, h, w = x_corr.size()

        f_step_forward = []
        f_step_backward = []

        x_uncorr_memo_forward = x_uncorr.mean(dim=1) ## b*c*h*w
        x_uncorr_memo_backward = x_uncorr.mean(dim=1)


        for i in range(0,t,1):
            x_corr_forward = x_corr[:,i,:,:,:]
            x_uncorr_forward = x_uncorr[:,i,:,:,:]


            f11 = self.forward_f1(x_uncorr_memo_forward)
            f21 = self.forward_f2( x_corr_forward )#

            c_atte = self.channel_atte_foreward_corr((f11 - f21).pow(2).mean(dim=-1).mean(dim=-1))
            x_temp = x_corr_forward * c_atte.view(b, c, 1, 1).contiguous().expand(b, c, h, w) + x_corr_forward
            f_step_forward.append(x_temp.mean(dim=-1).mean(dim=-1))

            x_uncorr_memo_forward = self.uncorr_memo_forward(x_uncorr_memo_forward, x_uncorr_forward)

            #########################

            x_corr_backward = x_corr[:, t-1-i, :, :, :]
            x_uncorr_backward = x_uncorr[:, t-1-i, :, :, :]

            f12 = self.backward_f1( x_uncorr_memo_backward )
            f22 = self.backward_f2( x_corr_backward )#

            c_atte = self.channel_atte_backward_corr((f12 - f22).pow(2).mean(dim=-1).mean(dim=-1))
            x_temp = x_corr_backward * c_atte.view(b, c, 1, 1).contiguous().expand(b, c, h, w) + x_corr_backward
            f_step_backward.append(x_temp.mean(dim=-1).mean(dim=-1))

            x_uncorr_memo_backward = self.uncorr_memo_backward(x_uncorr_memo_backward, x_uncorr_backward)


        temp = []
        for i in range(t):
            temp.append(f_step_backward[t-1-i])
        f_step_backward = torch.stack(temp, dim=1)
        f_step_forward = torch.stack(f_step_forward, dim=1)

        f_corr = f_step_forward + f_step_backward

        f_uncorr = x_uncorr_memo_forward.mean(dim=-1).mean(dim=-1) + x_uncorr_memo_backward.mean(dim=-1).mean(dim=-1)

        return f_uncorr, f_corr



class ResNet50_GRL_Model(nn.Module):
    '''
    Backbone: ResNet-50 + GRL modules.
    '''

    def __init__(self, num_feat=2048, num_features=512, height=256, width=128, pretrained=True,
                 dropout=0, numclasses=0):
        super(ResNet50_GRL_Model, self).__init__()
        self.pretrained = pretrained
        self.num_feat = num_feat  # resnet output
        self.dropout = dropout
        self.num_classes = numclasses
        self.output_dim = num_features  # bnneck
        print('Num of features: {}.'.format(self.num_feat))

        self.backbone = Backbone(height=height, width=width)

        self.temporal_learning_block = TRLBlock(2048)
        # #
        self.corr_bn = nn.BatchNorm1d(2048)
        init.constant_(self.corr_bn.weight, 1)
        init.constant_(self.corr_bn.bias, 0)

        self.uncorr_bn = nn.BatchNorm1d(2048)
        init.constant_(self.uncorr_bn.weight, 1)
        init.constant_(self.uncorr_bn.bias, 0)

    def forward(self, inputs, training=True):
        b, t, c, h, w = inputs.size()
        im_input = inputs.view(b * t, c, h, w)  # 80, 3, 256, 128
        x_uncorr, x_corr,corr_map = self.backbone(im_input, b, t)  # b*t,2048,16,8

        ###########################
        x_corr = x_corr.view(b, t, x_corr.size(1), x_corr.size(2), x_corr.size(3))
        x_uncorr = x_uncorr.view(b, t, x_uncorr.size(1), x_uncorr.size(2), x_uncorr.size(3))

        x_uncorr, x_corr = self.temporal_learning_block(x_uncorr, x_corr)  #

        x_corr = self.corr_bn(x_corr.view(b * t, 2048)).view(b, t, 2048)
        x_corr = F.normalize(x_corr, p=2, dim=2)

        x_uncorr = self.uncorr_bn(x_uncorr.view(b, 2048)).view(b, 2048)
        x_uncorr = F.normalize(x_uncorr, p=2, dim=1)

        return x_uncorr, x_corr


def resnet50_grl(*args, **kwargs):
    return ResNet50_GRL_Model(*args, **kwargs)
