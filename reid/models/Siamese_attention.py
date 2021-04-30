import torch
from torch import nn
import torch.nn.functional as F
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


class Siamese_attention(nn.Module):

    def __init__(self, input_num=2048, output_num=512, class_num=2):
        super(Siamese_attention, self).__init__()

        self.input_num = input_num
        self.output_num = output_num
        self.class_num = class_num
        self.feat_num = input_num
        # linear_Q
        self.featQ = nn.Linear(self.input_num, self.output_num)
        self.featQ_bn = nn.BatchNorm1d(self.output_num)
        self.featQ.apply(weights_init_kaiming)
        self.featQ_bn.apply(weights_init_kaiming)

        # linear_K
        self.featK = nn.Linear(self.input_num, self.output_num)
        self.featK_bn = nn.BatchNorm1d(self.output_num)
        self.featK.apply(weights_init_kaiming)
        self.featK_bn.apply(weights_init_kaiming)

        # linear_V
        self.featV = nn.Linear(self.input_num, self.output_num)
        self.featV_bn = nn.BatchNorm1d(self.output_num)
        self.featV.apply(weights_init_kaiming)
        self.featV_bn.apply(weights_init_kaiming)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

        # BCE classifier
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        self.classifierlinear = nn.Linear(self.feat_num, self.class_num)
        self.classifierBN.apply(weights_init_kaiming)
        self.classifierlinear.apply(weights_init_classifier)

    def self_attention(self, inputs):
        size = inputs.size()
        batch = size[0]
        len = size[1]

        Q = inputs.view(batch * len, -1)
        Q = self.featQ(Q)
        Q = self.featQ_bn(Q)
        Q = Q / Q.norm(2, 1).unsqueeze(1).expand_as(Q)
        Q = Q.contiguous().view(batch, len, -1)

        K = inputs.view(batch*len, -1)
        K = self.featK(K)
        K = self.featK_bn(K)
        K = K / K.norm(2, 1).unsqueeze(1).expand_as(K)
        K = K.contiguous().view(batch, len, -1)

        weights = torch.matmul(Q, K.transpose(-1, -2))
        weights = self.softmax(weights)

        V = inputs.view(batch, len, -1)
        pool_inputs = torch.matmul(weights, V)

        pool_inputs = pool_inputs.sum(1)
        pool_inputs = pool_inputs / pool_inputs.norm(2, 1).unsqueeze(1).expand_as(pool_inputs)
        pool_inputs = pool_inputs.squeeze(1)

        return pool_inputs

    def forward(self, x):
        xsize = x.size()
        sample_num = xsize[0]
        if sample_num % 2 != 0:
            raise RuntimeError("the batch size should be even number!")
        seq_len = x.size()[1]

        x = x.view(int(sample_num/2), 2, seq_len, -1)

        probe_x = x[:, 0, :, :]
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1, :, :]
        gallery_x = gallery_x.contiguous()

        probe_out = self.self_attention(probe_x)

        gallery_out = self.self_attention(gallery_x)

        siamese_out = torch.cat((probe_out, gallery_out))
        probesize = probe_out.size()
        gallerysize = gallery_out.size()
        probe_batch = probesize[0]
        gallery_batch = gallerysize[0]

        gallery_out = gallery_out.unsqueeze(0)
        probe_out = probe_out.unsqueeze(1)

        diff = probe_out - gallery_out
        diff = torch.pow(diff, 2)
        diff = diff.view(probe_batch * gallery_batch, -1).contiguous()
        diff = self.classifierBN(diff)
        cls_encode = self.classifierlinear(diff)
        cls_encode = cls_encode.view(probe_batch, gallery_batch, -1)

        return cls_encode, siamese_out
