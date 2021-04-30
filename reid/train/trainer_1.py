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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
        losses4 = AverageMeter()
        losses5 = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        precisions2 = AverageMeter()
        precisions3 = AverageMeter()
        precisions4 = AverageMeter()

        accumulation_steps = 8
        # total_loss = 0

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            # , loss_tr
            all_loss, prec_id_uncorr_frame, prec_id_corr_frame = self._forward(inputs, targets, i, epoch)  # 1.前向传播 , loss_tri
            loss = all_loss
            # # temp_id_loss +, prec_id_all_frame

            losses.update(loss.item(), targets.size(0))
            # losses1.update(loss_id1.item(), targets.size(0))
            # losses2.update(loss_id2.item(), targets.size(0))
            # losses3.update(loss_ver.item(), targets.size(0))
            # losses4.update(loss_tr_oim.item(), targets.size(0))
            # losses5.update(loss_tri.item(), targets.size(0))

            # precisions.update(prec_id_uncorr_vid, targets.size(0))

            # precisions1.update(prec_id_all_frame, targets.size(0))
            precisions2.update(prec_id_uncorr_frame, targets.size(0))
            # precisions3.update(prec_id_corr_frame, targets.size(0))
            precisions4.update(prec_id_corr_frame, targets.size(0))

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 100
            num_step = len(data_loader)  # 1217
            num_iter = num_step * epoch + i

            self.writer.add_scalar('train/total_loss_step', losses.val, num_iter)
            # self.writer.add_scalar('train/loss_id1_step', losses1.val, num_iter)
            # self.writer.add_scalar('train/loss_id2_step', losses2.val, num_iter)
            # self.writer.add_scalar('train/loss_ver_step', losses3.val, num_iter)
            # self.writer.add_scalar('train/loss_tr_oim_step', losses4.val, num_iter)
            # self.writer.add_scalar('train/loss_tr_step', losses5.val, num_iter)

            self.writer.add_scalar('train/total_loss_avg', losses.avg, num_iter)
            # self.writer.add_scalar('train/loss_id1_avg', losses1.avg, num_iter)
            # self.writer.add_scalar('train/loss_id2_avg', losses2.avg, num_iter)
            # self.writer.add_scalar('train/loss_ver_avg', losses3.avg, num_iter)
            # self.writer.add_scalar('train/loss_tr_oim_avg', losses4.avg, num_iter)
            # self.writer.add_scalar('train/loss_tr_avg', losses5.avg, num_iter)

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'                      
                      # 'uncorr_vid {:.2%} ({:.2%})\t'                      
                      # 'all_frame {:.2%} ({:.2%})\t'
                      'uncorr_frame {:.2%} ({:.2%})\t'
                      # 'corr_frame {:.2%} ({:.2%})\t'
                      'corr_frame {:.2%} ({:.2%})\t'

                      .format(epoch, i + 1, len(data_loader), losses.val, losses.avg,
                              # precisions.val, precisions.avg,
                              # precisions1.val, precisions1.avg,
                              precisions2.val, precisions2.avg,
                              # precisions3.val, precisions3.avg,
                              precisions4.val, precisions4.avg,
                              ))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, i, epoch):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

        def __init__(self, cnn_model, siamese_model, criterion_veri, criterion_oim1, criterion_oim2, criterion_oim3, loss_id, logdir):
            super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
            self.siamese_model = siamese_model
            self.regular_criterion1 = criterion_oim1
            self.regular_criterion2 = criterion_oim2
            self.regular_criterion3 = criterion_oim3

            self.loss_id = loss_id
            self.writer = SummaryWriter(log_dir=logdir)

        def _parse_data(self, inputs):
            imgs, pids, _ = inputs
            imgs = imgs.to(self.device)
            # flows = flows.to(self.device)
            inputs = [imgs]

            targets = pids.to(self.device)
            return inputs, targets

        def _forward(self, inputs, targets, i, epoch):
            batch_size = inputs[0].size(0)
            seq_len = inputs[0].size(1)
            ### CNN extract frame-wise feature
            feats_corr, feats_uncorr, feats_kernel = self.model(inputs[0]) #, x_corr, x_uncorr,

            # expand the target label ID loss
            ### expand to get the frame-wise label

            targetX = targets.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
            targetX = targetX.expand(batch_size, seq_len)
            # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
            targetX = targetX.contiguous()
            targetX = targetX.view(batch_size * seq_len, -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
            targetX = targetX.squeeze(1)
            ####################3

            # feat_frame = feats_frame.view(batch_size * seq_len, -1)  # 96,128
            feat_corr = feats_corr.view(batch_size * seq_len, -1)
            feat_uncorr = feats_uncorr.view(batch_size * seq_len, -1)

            #########################
            ###########################
            ### vid-wise oim loss
            uncorr_id_loss_vid, output_id, lut = self.regular_criterion1(feats_uncorr.mean(dim=1), targets)
            prec_id_uncorr_vid, = accuracy(output_id.data, targets.data)

            corr_id_loss_vid, output_id, lut = self.regular_criterion2(feats_corr.mean(dim=1), targets)
            prec_id_corr_vid, = accuracy(output_id.data, targets.data)

            # kernel_id_loss_vid, output_id, lut = self.regular_criterion3(feats_kernel, targets)
            # prec_id_kernel_vid, = accuracy(output_id.data, targets.data)


            #########################
            ###########################
            #### oim loss for frame-wise,
            uncorr_id_loss_frame, output_id, lut = self.regular_criterion1(feat_uncorr, targetX)
            prec_id_uncorr_frame, = accuracy(output_id.data, targetX.data)

            corr_id_loss_frame, output_id, lut = self.regular_criterion2(feat_corr, targetX)
            prec_id_corr_frame, = accuracy(output_id.data, targetX.data)

            # all_id_loss_frame, output_id, lut = self.regular_criterion3(feat_frame, targetX)
            # prec_id_all_frame, = accuracy(output_id.data, targetX.data)

            # verification label
            targets = targets.data  # 6,2  tensor([[ 94,  94],[ 10,  10],[ 15,  15],[ 16,  16], [ 75,  75],[ 39,  39]])
            targets = targets.view(int(batch_size / 2), -1)
            tar_probe = targets[:, 0]  # tensor([ 94,  10,  15,  16,  75,  39], device='cuda:0')
            tar_gallery = targets[:, 1]  # tensor([ 94,  10,  15,  16,  75,  39], device='cuda:0')

            ###################
            kernel_ver_score, corr_ver_score = self.siamese_model(feats_kernel, feats_corr.mean(dim=1))

            loss_ver_kernel, prec_ver_uncorr = self.criterion(kernel_ver_score, tar_probe, tar_gallery)
            loss_ver_corr, prec_ver_corr = self.criterion(corr_ver_score, tar_probe, tar_gallery)
            ######################

            uncorr_loss =uncorr_id_loss_frame #uncorr_id_loss_vid # + #  + loss_ver_uncorr
            corr_loss =  corr_id_loss_frame +corr_id_loss_vid +loss_ver_corr #+

            all_loss =  corr_loss + uncorr_loss #+ kernel_id_loss_vid + loss_ver_kernel


            return all_loss, prec_id_uncorr_frame, prec_id_corr_frame#, prec_id_all_frame

        def train(self, epoch, data_loader, optimizer1):
            self.siamese_model.train()
            # self.rate = ratetrain
            super(SEQTrainer, self).train(epoch, data_loader, optimizer1)

        def guiyihua(self, x):
            x_min = x.min()
            x_max = x.max()
            x_1 = (x - x_min) / (x_max - x_min)
            return x_1

        def get_raw_images(self, images):
            b, s, c, h, w = images.size()
            image = images.view(b * s, c, h, w)
            imgs_vis = []
            for k in range(b*s):
                img = image[k].unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                img = reverse_normalize(img)
                imgs_vis.append(img)
            return imgs_vis

        def keshihua(self, cam, imgs, i, savedir):
            visual_batch(cam, imgs, i, savedir)
            return print('done')
