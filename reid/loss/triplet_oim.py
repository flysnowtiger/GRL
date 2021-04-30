import torch
import torch.nn as nn


class TripletLoss_OIM(nn.Module):

    def __init__(self, margin=0, batch_hard=False, dim=2048):
        super(TripletLoss_OIM, self).__init__()
        self.batch_hard = batch_hard  # True
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin  # ‘soft’
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, lut, id=None, pos_mask=None, neg_mask=None, mode='id', dis_func='eu', n_dis=0):
        feat_OIM = []
        for i in id:
            feat_OIM.append(lut[i].unsqueeze(0))
        feat_oim = torch.cat(feat_OIM, dim=0)
        if dis_func == 'cdist':
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            dist = self.cdist(feat, feat)
        elif dis_func == 'eu':
            dist = self.cdist(feat, feat_oim)  # torch.Size([8, 8])

        if mode == 'id':
            if id is None:
                raise RuntimeError('foward is in id mode, please input id!')
            else:
                identity_mask = torch.eye(feat.size(0)).byte()  # torch.Size([8, 8])
                identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                negative_mask = same_id_mask ^ 1  # ^ 异或操作，同为0，异为1
                positive_mask = same_id_mask ^ identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                positive_mask = pos_mask
                same_id_mask = neg_mask ^ 1
                negative_mask = neg_mask
        else:
            raise ValueError('unrecognized mode')

        if self.batch_hard:
            if n_dis != 0:
                img_dist = dist[:-n_dis, :-n_dis]
                max_positive = (img_dist * positive_mask[:-n_dis, :-n_dis].float()).max(1)[0]
                min_negative = (img_dist + 1e5 * same_id_mask[:-n_dis, :-n_dis].float()).min(1)[0]
                dis_min_negative = dist[:-n_dis, -n_dis:].min(1)[0]
                z_origin = max_positive - min_negative
                # z_dis = max_positive - dis_min_negative
            else:
                max_positive = dist * positive_mask.float()
                max_positive = max_positive.max(1)[0]
                # tensor([11.2461, 11.0022, 11.2461, 11.1370,  8.9170,  8.4666,  8.9170,  8.4710])
                same_id_mask = 1e5 * same_id_mask.float()
                min_negative = dist + same_id_mask
                min_negative = min_negative.min(1)[
                    0]  # tensor([ 9.5545, 10.7909,  9.3813, 10.1063, 10.7909, 10.1282,  9.3813, 11.3685],
                z = max_positive - min_negative  # tensor([3.6010, 2.3646, 2.0217, 2.9059, 0.7020, 2.2440, 1.1459, 1.0037],
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1, 1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1, 1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative

        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            if n_dis != 0:
                b_loss = torch.log(1 + torch.exp(z_origin)) + -0.5 * dis_min_negative  # + torch.log(1+torch.exp(z_dis))
            else:
                b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")
        return b_loss

    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b

        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff ** 2).sum(2) + 1e-12).sqrt()
