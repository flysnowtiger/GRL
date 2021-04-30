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


class Siamese_video(nn.Module):

    def __init__(self, input_num=2048, output_num=2048, class_num=2):
        super(Siamese_video, self).__init__()

        self.output_num = output_num
        self.class_num = class_num
        self.feat_num = input_num

        # BCE classifier
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        self.classifierlinear = nn.Linear(self.feat_num, self.class_num)
        self.classifierBN.apply(weights_init_kaiming)
        self.classifierlinear.apply(weights_init_classifier)
        self.muti_head = False


    def forward(self, x):

        batchsize = x.size(0)

        x = x.reshape(int(batchsize/2), 2, -1)
        pooled_probe = x[:,0,:]
        pooled_gallery = x[:,1,:]

        siamese_out = torch.cat((pooled_probe, pooled_gallery))
        probesize = pooled_probe.size()
        gallerysize = pooled_gallery.size()
        probe_batch = probesize[0]
        gallery_batch = gallerysize[0]

        pooled_gallery = pooled_gallery.unsqueeze(0)
        pooled_probe = pooled_probe.unsqueeze(1)

        diff = pooled_probe - pooled_gallery
        diff = torch.pow(diff, 2)
        diff = diff.view(probe_batch * gallery_batch, -1).contiguous()
        diff = self.classifierBN(diff)

        cls_encode = self.classifierlinear(diff)
        cls_encode = cls_encode.view(probe_batch, gallery_batch, -1)

        return cls_encode, siamese_out
