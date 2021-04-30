# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -*-coding:utf-8-*-
import torch
from torch import nn
from torch.nn import functional as F

import pdb
import math

# ===================
#     RGA Module
# ===================


class RGA_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, cha_ratio=8, spa_ratio=8, down_ratio=8):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel  # 2048
		self.in_spatial = in_spatial  # 128=16*8

		self.spa_num = 6

		# 1 mask block size
		self.block_Hsize = 3
		self.block_Wsize = 3

		# Networks for learning attention weights

		self.W_spatial = nn.Sequential(
			nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel // down_ratio,
			          kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(self.in_channel // down_ratio),
			nn.ReLU(),
			nn.Conv2d(in_channels=self.in_channel // down_ratio, out_channels=self.spa_num,
			          kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(self.spa_num)
		)
		self.conv_erase = nn.Sequential(
			nn.Conv2d(self.in_channel, self.in_channel,
			          kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(self.in_channel)
		)

		# init

		for m in self.conv_erase:
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(0)
				m.bias.data.zero_()

	def forward(self, x, b, t):
		bt, c, h, w = x.size()  # bt, 2048, 16, 8

		# spatial attention
		soft_masks, reg = self.get_spatt(x, self.spa_num, bt, h, w)  # bt, num, 16, 8
		# Diversity Regularization todo
		temp = soft_masks[:, 0].unsqueeze(1)
		# Multiple Spatial Attention todo
		# mask = torch.zeros_like(temp)
		soft_mask = torch.zeros_like(temp)
		for i in range(self.spa_num):
			# mask_i, ease_i = self.block_binarization(soft_masks[:, i].unsqueeze(1))
			# mask = torch.max(mask, mask_i)
			soft_mask = torch.max(soft_mask, (soft_masks[:, i].unsqueeze(1)))
		# out = mask * x + soft_mask * x + x  # bt, 2048, 16,8
		out = soft_mask * x + x  # bt, 2048, 16,8
		out = out.contiguous().view(-1, c, h, w)
		return out, reg

	def get_spatt(self, f, num, bt, h, w):
		sp_att = self.W_spatial(f)  # bt, spa_num, 16, 8
		sp_att = sp_att.view(bt*num, -1)  # bt*n, 128
		a = F.softmax(sp_att, dim=-1).unsqueeze(1)   # b, 128
		reg = a.view(bt, num, -1)
		soft_masks = a.view(bt, num, h, w)  # bt,num,16,8
		return soft_masks, reg

	def erase_feature(self, x, masks, soft_masks):
		"""erasing the x with the masks, the softmasks for gradient back-propagation.
		"""
		bt, c, h, w = x.size()  # bt， 2048， 16， 8
		soft_masks = soft_masks - (1 - masks) * 1e8  # bt,1,16,8
		soft_masks = F.softmax(soft_masks.view(bt, h * w), dim=-1)  # bt， 128
		# 用mask对输入的图片特征进行擦除 [bt,2048,16,8]*[bt,1,16,8]=[bt,2048,16,8]
		inputs = x * masks  # bt,2048,16,8
		res = torch.bmm(x.view(bt, c, h * w), soft_masks.unsqueeze(-1))  # [bt,2048,128][bt,128,1] = [bt,2048,1]
		outputs = inputs + self.conv_erase(res.unsqueeze(-1))  # [bt,2048,16,8]+[bt,2048,1,1]  = bt,2048,16,8
		return outputs

	def block_binarization(self, f):
		"""
		generate the binary masks :二值化mask
		f: [bt, 1, 16, 8]
		"""
		soft_masks = f  # 16,1,16,8
		bs, t, h, w = f.size()  # 16,1,16,8
		# H
		f1 = torch.mean(f, 3)  # 16,1,16

		weight1 = torch.ones(1, 1, self.block_Hsize, 1).cuda()  # torch.Size([1, 1, 3, 1])  分成三块

		f1 = F.conv2d(input=f1.view(-1, 1, h, 1), weight=weight1,
		              padding=(self.block_Hsize // 2, 0))  # torch.Size([16, 1, 16, 1])

		if self.block_Hsize % 2 == 0:
			f1 = f1[:, :, :-1]
		# f = [16,16]
		index1 = torch.argmax(f1.view(bs * t, h), dim=1)  # torch.size([16]) 找到每一行的最大值index

		# generate the masks
		masks1 = torch.zeros(bs * t, h).cuda()  # torch.Size([16, 16])

		index_b1 = torch.arange(0, bs * t, dtype=torch.long)  # torch.Size([16])
		masks1[index_b1, index1] = 1  # 两个index相等的地方置1  [16,16]

		block_masks1 = F.max_pool2d(input=masks1[:, None, :, None], kernel_size=(self.block_Hsize, 1),
		                            stride=(1, 1), padding=(self.block_Hsize // 2, 0))  # torch.Size([16, 1, 16, 1])
		if self.block_Hsize % 2 == 0:
			block_masks1 = block_masks1[:, :, 1:]

		# W
		f2 = torch.mean(f, 2)  # 16,1,8

		weight2 = torch.ones(1, 1, 1, self.block_Wsize).cuda()  # torch.Size([1, 1, 1, 3])  分成三块
		f2 = F.conv2d(input=f2.view(-1, 1, 1, w),
		              weight=weight2,
		              padding=(0, self.block_Wsize // 2))  # torch.Size([16, 1, 1, 8])

		if self.block_Wsize % 2 == 0:
			f2 = f2[:, :, :-1]
		# f = [16,16]
		index2 = torch.argmax(f2.view(bs * t, w), dim=1)  # torch.size([16])

		# generate the masks
		masks2 = torch.zeros(bs * t, w).cuda()  # torch.Size([16, 8])
		index_b2 = torch.arange(0, bs * t, dtype=torch.long)  # torch.Size([16])
		masks2[index_b2, index2] = 1  # 两个index相等的地方置1  [16,8]

		block_masks2 = F.max_pool2d(input=masks2[:, None, None, :],
		                            kernel_size=(1, self.block_Wsize),
		                            stride=(1, 1),
		                            padding=(0, self.block_Wsize // 2))  # torch.Size([16, 1, 1, 8])
		if self.block_Wsize % 2 == 0:
			block_masks2 = block_masks2[:, :, 1:]
		block_masks1 = block_masks1.expand(bs, t, h, w)
		block_masks2 = block_masks2.expand(bs, t, h, w)
		block_masks = block_masks1 + block_masks2
		zero = torch.zeros_like(block_masks)
		one = torch.ones_like(block_masks)
		block_masks = torch.where(block_masks < 1.5, zero, block_masks)
		block_masks = torch.where(block_masks > 1.5, one, block_masks)
		erease_masks = 1 - block_masks  # torch.Size([16, 1, 16, 8])
		return block_masks, erease_masks
