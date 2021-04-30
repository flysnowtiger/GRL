# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import sys
import os

import torch
import torch as th
from torch import nn
import torch.nn.functional as F

from .resnets1 import resnet50_s1


class Backbone(nn.Module):
	def __init__(self, height=256, width=128):
		super(Backbone, self).__init__()
		# resnet50
		resnet2d = resnet50_s1(pretrained=True)

		self.base = nn.Sequential(
			resnet2d.conv1,
			resnet2d.bn1,
			nn.ReLU(),
			resnet2d.maxpool,
			resnet2d.layer1,
			resnet2d.layer2,
			resnet2d.layer3,
			resnet2d.layer4,
		)

		self.glo_fc = nn.Sequential(nn.Linear(2048, 1024),
									  nn.BatchNorm1d(1024),
									  nn.ReLU())
		self.corr_atte = nn.Sequential(
			nn.Conv2d(2048 + 1024, 1024, 1, 1, bias=False),
			nn.BatchNorm2d(1024),
			nn.Conv2d(1024, 256, 1, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 1, 1, 1, bias=False),
			nn.BatchNorm2d(1),
		)

	def forward(self, x, b, t):

		x = self.base(x)

		###Global correlation estimation

		x_4 = x.view(b, t, x.size(1), x.size(2), x.size(3))
		x_glo = x_4.mean(dim=-1).mean(dim=-1).mean(dim=1)
		glo = self.glo_fc(x_glo).view(b,1, 1024, 1, 1).contiguous().expand(b,t, 1024, 16, 8).contiguous().view(b*t,1024, 16,8)

		x_corr = torch.cat((x, glo), dim=1)
		corr_map = self.corr_atte(x_corr)
		corr_map = F.sigmoid(corr_map).view(b * t, 1, 16, 8).contiguous()

		# disentanglement
		x_corr = x * corr_map
		x_uncorr = x*(1-corr_map)

		return x_uncorr, x_corr, corr_map
