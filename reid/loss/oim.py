from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut  # torch.Size([625, 128])
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)   # inputs: torch.Size([64, 128])
        outputs = inputs.mm(self.lut.t())  # (64, 128) * (128, 625)
        return outputs  # torch.Size([64, 625])

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features  # 512
        self.num_classes = num_classes  # 625
        self.momentum = momentum  # 0.5
        self.scalar = scalar  # 30
        self.weight = weight  # None
        self.register_buffer('lut', torch.zeros(num_classes, num_features))
        self.size_average = size_average  # True

    def forward(self, inputs, targets):#
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight)
        return loss, inputs
