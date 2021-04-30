from __future__ import absolute_import
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, dropout=0, numclasses=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrain) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)

        self.classifier = nn.Linear(self.base.fc.in_features, numclasses)  # 2048, C
        init.kaiming_uniform_(self.classifier.weight, mode='fan_out')
        init.constant_(self.classifier.bias, 0)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.has_embedding = num_features > 0

            out_planes = self.base.fc.in_features
            self.feat_bn2 = nn.BatchNorm1d(out_planes)
            init.constant_(self.feat_bn2.weight, 1)
            init.constant_(self.feat_bn2.bias, 0)
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_uniform_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                self.num_features = out_planes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, imgs):
        # todo: change the base model
        img_size = imgs.size()
        # motion_size = motions.size()
        batch_sz = img_size[0]
        seq_len = img_size[1]
        imgs = imgs.view(-1, img_size[2], img_size[3], img_size[4])

        for name, module in self.base._modules.items():

            if name == 'conv1':
                # x = module(imgs) + self.conv0(motions)
                x = module(imgs)
                continue
            if name == 'avgpool':
                break
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])  # torch.Size([64, 2048, 1, 1])
        x = x.view(x.size(0), -1)  # torch.Size([64, 2048])
        raw = self.feat_bn2(x)
        raw = raw / raw.norm(2, 1).unsqueeze(1).expand_as(raw)
        raw = raw.squeeze(1)
        raw = raw.view(batch_sz, seq_len, -1)  # torch.Size([8, 8, 2048])

        x = self.feat(x)  # 64,128
        x = self.feat_bn(x)

        x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        x = x.squeeze(1)
        x = x.view(batch_sz, seq_len, -1)  # 8,8,128
        return x, raw

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def guiyihua(self, x):
        x_min = x.min()
        x_max = x.max()
        x_1 = (x - x_min) / (x_max - x_min)
        return x_1


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
