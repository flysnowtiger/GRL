from __future__ import print_function, absolute_import
import time
import torch
from reid.evaluator import accuracy
from utils.meters import AverageMeter
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from visualize import reverse_normalize
from cam_functions import visual_batch

from reid.loss import TripletLoss, TripletLoss_OIM
criterion_triplet_oim = TripletLoss_OIM('soft', True)
criterion_triplet = TripletLoss('soft', True)


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion_ver = criterion
        self.criterion_ver_uncorr = criterion
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

            all_loss, uncorr_prec_id_vid, corr_prec_id_vid, corr_prec_id_frame = self._forward(inputs, targets, i, epoch)
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
            num_step = len(data_loader)
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

        def __init__(self, cnn_model, siamese_model, siamese_model_uncorr, criterion_veri, criterion_corr, criterion_uncorr, logdir):
            super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
            self.siamese_model = siamese_model
            self.siamese_model_uncorr = siamese_model_uncorr

            self.criterion_uncorr = criterion_uncorr
            self.criterion_corr = criterion_corr

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

            x_uncorr, x_corr = self.model(inputs[0])

            # uncorr_id_loss_vid, output_id = self.criterion_uncorr(x_uncorr, targets)
            # uncorr_prec_id_vid, = accuracy(output_id.data, targets.data)

            # expand the target label ID loss
            frame_corr = x_corr.view(batch_size * seq_len, -1)

            targetX = targets.unsqueeze(1)
            targetX = targetX.expand(batch_size, seq_len)
            targetX = targetX.contiguous()
            targetX = targetX.view(batch_size * seq_len, -1)  #
            targetX = targetX.squeeze(1)

            #######
            corr_id_loss_frame, output_id = self.criterion_corr(frame_corr, targetX)
            corr_prec_id_frame, = accuracy(output_id.data, targetX.data)

            # verification label
            targets = targets.data
            targets = targets.view(int(batch_size / 2), -1)
            tar_probe = targets[:, 0]
            tar_gallery = targets[:, 1]

            target = torch.cat((tar_probe, tar_gallery))

            encode_scores, siamese_out = self.siamese_model(x_corr)
            corr_id_loss_vid, output_id = self.criterion_corr(siamese_out, target)
            corr_prec_id_vid, = accuracy(output_id.data, target.data)

            corr_loss_tri = criterion_triplet(siamese_out, target).mean()

            ### verification loss for pair-wise video feature
            encode_size = encode_scores.size()
            encodemat = encode_scores.view(-1, 2)
            encodemat = F.softmax(encodemat, dim=-1)
            encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
            encodemat0 = encodemat[:, :, 1]
            corr_loss_ver, corr_prec_ver = self.criterion_ver(encodemat0, tar_probe, tar_gallery)

            encode_scores, siamese_out = self.siamese_model_uncorr(x_uncorr)
            uncorr_id_loss_vid, output_id = self.criterion_uncorr(siamese_out, target)
            uncorr_prec_id_vid, = accuracy(output_id.data, target.data)
            
            # uncorr_loss_tri = criterion_triplet(siamese_out, target).mean()
            
            encode_size = encode_scores.size()  
            encodemat = encode_scores.view(-1, 2)  
            encodemat = F.softmax(encodemat, dim=-1)
            encodemat = encodemat.view(encode_size[0], encode_size[1], 2)  
            encodemat0 = encodemat[:, :, 1]  
            uncorr_loss_ver, uncorr_prec_ver = self.criterion_ver_uncorr(encodemat0, tar_probe, tar_gallery)


            corr_loss = corr_id_loss_frame + corr_id_loss_vid + corr_loss_ver*20 + corr_loss_tri
            uncorr_loss = uncorr_id_loss_vid #+ corr_loss_ver*10

            all_loss = uncorr_loss + corr_loss

            return all_loss, uncorr_prec_id_vid, corr_prec_id_vid , corr_prec_id_frame

        def train(self, epoch, data_loader, optimizer1):
            self.siamese_model.train()
            self.siamese_model_uncorr.train()

            super(SEQTrainer, self).train(epoch, data_loader, optimizer1)

