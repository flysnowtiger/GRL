#!usr/bin/env python
# -*- coding:utf-8 _*-
# @author: ycy
# @contact: asuradayuci@gmail.com
# @time: 2019/9/7 下午2:53
import torch
from torch import nn
import torch.nn.functional as F
#!usr/bin/env python
# -*- coding:utf-8 _*-
# @author: ycy
# @contact: asuradayuci@gmail.com
# @time: 2019/9/7 下午2:53
import torch
from torch import nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
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
        # if m.bias:
        #     nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.bias, 0.0)


class Siamese_video(nn.Module):

    def __init__(self, input_num=2048, output_num=2048, class_num=2):
        super(Siamese_video, self).__init__()

        # self.input_num = 2048
        # self.output_num = 512
        self.class_num = class_num
        self.feat_num = input_num
        # linear_Q
        # self.featQ = nn.Linear(self.input_num, self.output_num)
        # self.featQ_bn = nn.BatchNorm1d(self.output_num)
        # self.featQ.apply(weights_init_kaiming)
        # self.featQ_bn.apply(weights_init_kaiming)
        #
        # # linear_K
        # self.featK = nn.Linear(self.input_num, self.output_num)
        # self.featK_bn = nn.BatchNorm1d(self.output_num)
        # self.featK.apply(weights_init_kaiming)
        # self.featK_bn.apply(weights_init_kaiming)
        #
        # # linear_V
        # self.featV = nn.Linear(self.input_num, self.output_num)
        # self.featV_bn = nn.BatchNorm1d(self.output_num)
        # self.featV.apply(weights_init_kaiming)
        # self.featV_bn.apply(weights_init_kaiming)
        #
        # # Softmax
        # self.softmax = nn.Softmax(dim=-1)
        #
        # # numti_head
        # self.d_k = 128
        # self.head = 4

        # BCE classifier
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        self.classifierlinear = nn.Linear(self.feat_num, self.class_num)
        self.classifierBN.apply(weights_init_kaiming)
        self.classifierlinear.apply(weights_init_classifier)
        self.muti_head = False

    def self_attention(self, probe_value, probe_base):
        pro_size = probe_value.size()  # torch.Size([4, 8, 128])
        pro_batch = pro_size[0]
        pro_len = pro_size[1]

        Qs = probe_base.view(pro_batch * pro_len, -1)  # 32 , 2048
        Qs = self.featQ(Qs)
        Qs = self.featQ_bn(Qs)  # 32, 128
        Qs = Qs / Qs.norm(2, 1).unsqueeze(1).expand_as(Qs)  # torch.Size([32, 256])
        if self.muti_head:
            Qs = Qs.contiguous().view(pro_batch, -1, self.head, self.d_k).transpose(1, 2)  # torch.Size([4, 4, 8, 64])
        else:
            Qs = Qs.contiguous().view(pro_batch, pro_len, -1)  # torch.Size([4, 8, 512])

        # generating Keys, key 不等于 value
        K = probe_base.view(pro_batch*pro_len, -1)
        K = self.featK(K)
        K = self.featK_bn(K)
        K = K / K.norm(2, 1).unsqueeze(1).expand_as(K)
        if self.muti_head:
            tmp_k = K.view(pro_batch, -1, self.head, self.d_k).transpose(1, 2)  # torch.Size([4, 4, 8, 64])
        else:
            tmp_k = K.view(pro_batch, pro_len, -1)  # torch.Size([4, 8, 512])

        # 1.single= [4,8, 512] * [4, 512, 8] = 4, 8, 8
        weights = torch.matmul(Qs, tmp_k.transpose(-1, -2))  # 2. muti:torch.Size([4, 4, 8, 8])

        weights = self.softmax(weights)  # 4 * 8 * 8  torch.Size([4, 4, 8, 8])

        if self.muti_head:
            V = probe_value.view(pro_batch, -1, self.head, self.d_k).transpose(1, 2)
        else:
            V = probe_value.view(pro_batch, pro_len, -1)

        pool_probe = torch.matmul(weights, V)  # ([4, 8, 8]) * ([4, 8, 512]) = 4 * 8 * 512   torch.Size([4, 4, 8, 64])
        if self.muti_head:
            pool_probe = pool_probe.transpose(1, 2).contiguous()  # torch.Size([4, 8, 4, 64])
            pool_probe = pool_probe.view(pro_batch, -1, self.head * self.d_k)  # torch.Size([4, 8, 512])

        pool_probe = pool_probe.sum(1)  # torch.Size([4, 128])
        # pool_probe = torch.mean(probe_value, dim=1)
        pool_probe = pool_probe / pool_probe.norm(2, 1).unsqueeze(1).expand_as(pool_probe)  # 单位向量
        pool_probe = pool_probe.squeeze(1)

        return pool_probe, pool_probe

    def forward(self, x):
        # xsize = x.size()  # 12,8,128
        # sample_num = xsize[0]  # 12
        #
        # if sample_num % 2 != 0:
        #     raise RuntimeError("the batch size should be even number!")
        #
        # seq_len = x.size()[1]  # 8
        # x = x.view(int(sample_num/2), 2, seq_len, -1)  # torch.Size([6, 2, 8, 128])
        # input = input.view(int(sample_num/2), 2, seq_len, -1)  # torch.Size([6, 2, 8, 2048])  => raw
        # probe_x = x[:, 0, :, :]
        # probe_x = probe_x.contiguous()  # torch.Size([6, 8, 128])
        # gallery_x = x[:, 1, :, :]
        # gallery_x = gallery_x.contiguous()  # torch.Size([6, 8, 128])
        #
        # probe_input = input[:, 0, :, :]
        # probe_input = probe_input.contiguous()  # torch.Size([6, 8, 2048])
        # gallery_input = input[:, 1, :, :]
        # gallery_input = gallery_input.contiguous()  # torch.Size([6, 8, 2048])
        #
        # # self-pooling  pooled_probe:torch.Size([6, 128])    hidden_probe:torch.Size([6, 128])
        # pooled_probe, probe_out_raw = self.self_attention(probe_x, probe_input)
        # # pooled_probe = probe_x.mean(dim=1)
        # # probe_out_raw = probe_input.mean(dim=1)
        # #
        # pooled_gallery, gallery_out_raw = self.self_attention(gallery_x, gallery_input)
        # # pooled_gallery = gallery_x.mean(dim=1)
        # # gallery_out_raw = gallery_input.mean(dim=1)

        batchsize = x.size(0)

        x = x.reshape(int(batchsize/2), 2, -1)
        pooled_probe = x[:,0,:]
        pooled_gallery = x[:,1,:]


        siamese_out = torch.cat((pooled_probe, pooled_gallery))
        probesize = pooled_probe.size()  # 4, 2048
        gallerysize = pooled_gallery.size()  # 4, 2048
        probe_batch = probesize[0]  # 4
        gallery_batch = gallerysize[0]  # 4

        # pooled_gallery: 4, 4, 2048
        pooled_gallery = pooled_gallery.unsqueeze(0)  # 1, 4, 2048

        pooled_probe = pooled_probe.unsqueeze(1)  # 4, 1, 2048

        diff = pooled_probe - pooled_gallery
        diff = torch.pow(diff, 2)  # torch.Size([4, 4, 2048])
        diff = diff.view(probe_batch * gallery_batch, -1).contiguous()  # torch.Size([16, 2048])
        diff = self.classifierBN(diff)
        # diff = diff / diff.norm(2, 1).unsqueeze(1).expand_as(diff)
        cls_encode = self.classifierlinear(diff)  # torch.Size([16, 2])
        cls_encode = cls_encode.view(probe_batch, gallery_batch, -1)  # torch.Size([4, 4, 2])

        return cls_encode, siamese_out
