from __future__ import absolute_import

import torch
from torch import nn

from reid.evaluator import accuracy


class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()

        # self.sigmod = nn.Sigmoid()
        self.BCE = nn.BCELoss()
        self.BCE.size_average = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, score, tar_probe, tar_gallery):
        cls_Size = score.size()  # torch.Size([4, 2])
        N_probe = cls_Size[0]  # 4
        N_gallery = cls_Size[0]

        tar_gallery = tar_gallery.unsqueeze(1)  # 6,1   tensor([[ 94],[ 10],[ 15],[ 16],[ 75],[ 39]])
        tar_probe = tar_probe.unsqueeze(0)  # 1,6 tensor([[ 94,  10,  15,  16,  75,  39]])
        mask = tar_probe.expand(N_probe, N_gallery).eq(tar_gallery.expand(N_probe, N_gallery))
        mask = mask.view(-1).cpu().numpy().tolist()
        # [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]

        score = score.contiguous()  # torch.Size([4, 4])
        samplers = score.view(-1)  # torch.Size([16])

        # samplers = self.sigmod(samplers)
        # labels = Variable(torch.Tensor(mask).cuda())
        labels = torch.Tensor(mask).to(self.device)

        loss = self.BCE(samplers, labels)

        samplers_data = samplers.data  # torch.Size([36])
        samplers_neg = 1 - samplers_data
        samplerdata = torch.cat((samplers_neg.unsqueeze(1), samplers_data.unsqueeze(1)), 1)  # torch.Size([36, 2])

        labeldata = torch.LongTensor(mask).to(self.device)
        prec, = accuracy(samplerdata, labeldata)

        return loss, prec
