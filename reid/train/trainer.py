from __future__ import print_function, absolute_import
import time
import torch
from torch import nn
from reid.evaluator import accuracy
from utils.meters import AverageMeter
import torch.nn.functional as F
from utils import to_numpy
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# import matplotlib.pyplot as plt
# mode decide how to train the model
from reid.loss import PairLoss, OIMLoss

from visualize import reverse_normalize
from cam_functions import visual_batch
# triplet
from reid.loss import TripletLoss, TripletLoss_OIM
criterion_triplet_oim = TripletLoss_OIM('soft', True)
criterion_triplet = TripletLoss('soft', True)


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.criterion_uncorr = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        precisions2 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)

            all_loss, uncorr_prec_id_vid, corr_prec_id_vid, corr_prec_id_frame = self._forward(inputs, targets, i, epoch)  # 1.前向传播 , loss_tri

            loss = all_loss

            losses.update(loss.item(), targets.size(0))

            precisions.update(uncorr_prec_id_vid, targets.size(0))
            precisions1.update(corr_prec_id_vid, targets.size(0))
            precisions2.update(corr_prec_id_frame, targets.size(0))

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 100
            num_step = len(data_loader)  # 1217
            num_iter = num_step * epoch + i

            self.writer.add_scalar('train/total_loss_step', losses.val, num_iter)
            self.writer.add_scalar('train/total_loss_avg', losses.avg, num_iter)

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'uncorr_vid {:.2%} ({:.2%})\t'
                      'corr_vid {:.2%} ({:.2%})\t'
                      'corr_frame {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader), losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              precisions1.val, precisions1.avg,
                              precisions2.val, precisions2.avg
                              ))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, i, epoch):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

        def __init__(self, cnn_model, siamese_model_corr, siamese_model_uncorr, criterion_veri, criterion_oim_corr, criterion_oim_uncorr, logdir):
            super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
            self.siamese_model_corr = siamese_model_corr
            self.siamese_model_uncorr = siamese_model_uncorr
            self.criterion_oim_corr = criterion_oim_corr
            self.criterion_oim_uncorr = criterion_oim_uncorr

            self.writer = SummaryWriter(log_dir=logdir)

        def _parse_data(self, inputs):
            imgs, pids, _ = inputs
            imgs = imgs.to(self.device)
            inputs = [imgs]
            targets = pids.to(self.device)
            return inputs, targets

        def _forward(self, inputs, targets, i, epoch):
            batch_size = inputs[0].size(0)
            seq_len = inputs[0].size(1)

            ### extract feature
            x_uncorr, x_corr = self.model(inputs[0])

            #### Uncorr Loss

            uncorr_id_loss_vid, output_id_uncorr = self.criterion_oim_uncorr(x_uncorr, targets)
            uncorr_prec_id_vid, = accuracy(output_id_uncorr.data, targets.data)

            ######
            # frame-wise label
            target_frame = targets.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
            target_frame = target_frame.expand(batch_size, seq_len)
            # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
            target_frame = target_frame.contiguous()
            target_frame = target_frame.view(batch_size * seq_len, -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
            target_frame = target_frame.squeeze(1)
            #######
            # verification label
            targets = targets.data
            targets = targets.view(int(batch_size / 2), -1)
            tar_probe = targets[:, 0]
            tar_gallery = targets[:, 1]
            target_video = torch.cat((tar_probe, tar_gallery))

            #### Corr features Loss

            # Frame-wise OIM loss for Corr features
            frames_corr = x_corr.view(batch_size * seq_len, -1)
            corr_id_loss_frame, output_id_corr1 = self.criterion_oim_corr(frames_corr, target_frame)
            corr_prec_id_frame, = accuracy(output_id_corr1.data, target_frame.data)

            encode_scores, siamese_out = self.siamese_model_corr(x_corr)

            # Video-wise OIM loss for Corr features
            corr_id_loss_vid, output_id_corr2 = self.criterion_oim_corr(siamese_out, target_video)
            corr_prec_id_vid, = accuracy(output_id_corr2.data, target_video.data)

            corr_loss_tri = criterion_triplet(siamese_out, target_video).mean()

            # verification loss for Corr features
            encode_size = encode_scores.size()
            encodemat = encode_scores.view(-1, 2)
            encodemat = F.softmax(encodemat, dim=-1)
            encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
            ver_cls = encodemat[:, :, 1]
            corr_loss_ver, corr_prec_ver = self.criterion(ver_cls, tar_probe, tar_gallery)

            # encode_scores, siamese_out = self.siamese_model_uncorr(x_uncorr)
            #
            # # Video-wise OIM loss for Uncorr features
            # uncorr_id_loss_vid, output_id_uncorr = self.criterion_oim_uncorr(siamese_out, target_video)
            # uncorr_prec_id_vid, = accuracy(output_id_uncorr.data, target_video.data)

            # # verification loss for Uncorr features
            # encode_size = encode_scores.size()
            # encodemat = encode_scores.view(-1, 2)
            # encodemat = F.softmax(encodemat, dim=-1)
            # encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
            # ver_cls = encodemat[:, :, 1]
            # uncorr_loss_ver, uncorr_prec_ver = self.criterion_uncorr(ver_cls, tar_probe, tar_gallery)

            corr_loss = corr_id_loss_frame + corr_id_loss_vid + corr_loss_ver*20 + corr_loss_tri
            uncorr_loss = uncorr_id_loss_vid

            all_loss = uncorr_loss + corr_loss

            return all_loss, uncorr_prec_id_vid, corr_prec_id_vid , corr_prec_id_frame

        def train(self, epoch, data_loader, optimizer1):
            self.siamese_model_corr.train()
            self.siamese_model_uncorr.train()

            super(SEQTrainer, self).train(epoch, data_loader, optimizer1)

