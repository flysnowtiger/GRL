# system tool
from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import sys

# computation tool
import torch
import numpy as np

# device tool
import torch.backends.cudnn as cudnn

from utils.logging import Logger
from utils.save_or_load import save_checkpoint, load_best_checkpoint
from utils.serialization import remove_repeat_tensorboard_files
from utils.adjust_lr import adjust_lr
from reid import models
from reid.loss import PairLoss, OIMLoss
from reid.data import get_data
from reid.train import SEQTrainer
from reid.evaluator import ATTEvaluator



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # log file 日志文件  防止重名覆盖
    run = 0
    if args.evaluate == 1:
        while osp.exists("%s" % (osp.join(args.logs_dir, 'log_test{}.txt'.format(run)))):
            run += 1

        sys.stdout = Logger(osp.join(args.logs_dir, 'log_test{}.txt'.format(run)))
    else:
        while osp.exists("%s" % (osp.join(args.logs_dir, 'log_train{}.txt'.format(run)))):
            run += 1

        sys.stdout = Logger(osp.join(args.logs_dir, 'log_train{}.txt'.format(run)))
    print("==========\nArgs:{}\n==========".format(args))

    # from reid.data import get_data ,根据get_data()函数,返回 数据集,行人数目,封装成batch的训练数据,查询数据,图库数据
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.seq_len, args.seq_srd,
                 args.workers, only_eval=False)

    # create CNN model  1.建立CNN模型,默认是resnet50 , num_features = 128
    cnn_model = models.create(args.arch1, num_features=args.features, dropout=args.dropout, numclasses=num_classes)
    # create Siamese modelm
    siamese_model_corr = models.create(args.arch2, input_num=args.features, output_num=512, class_num=2)
    siamese_model_uncorr = models.create(args.arch3, input_num=args.features, output_num=512, class_num=2)

    cnn_model = torch.nn.DataParallel(cnn_model).to(device)
    siamese_model_corr = siamese_model_corr.to(device)
    siamese_model_uncorr = siamese_model_uncorr.to(device)

    # Loss function 5.损失函数定义
    criterion_oim_corr = OIMLoss(2048, num_classes, scalar=args.oim_scalar, momentum=args.oim_momentum)
    criterion_oim_uncorr = OIMLoss(2048, num_classes, scalar=args.oim_scalar, momentum=args.oim_momentum)
    criterion_veri = PairLoss()

    criterion_oim_corr.to(device)
    criterion_oim_uncorr.to(device)
    criterion_veri.to(device)

    base_param_ids = set(map(id, cnn_model.module.backbone.parameters()))
    new_params = [p for p in cnn_model.parameters() if
                  id(p) not in base_param_ids]

    param_groups = [
        {'params': cnn_model.module.backbone.parameters(), 'lr_mult': 1},
        {'params': new_params, 'lr_mult': 2},
        {'params': siamese_model_corr.parameters(), 'lr_mult': 2},
        {'params': siamese_model_uncorr.parameters(), 'lr_mult': 2}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 nesterov=True)

    # Evaluator  测试
    evaluator = ATTEvaluator(cnn_model, siamese_model_corr)

    best_top1 = 0
    if args.evaluate == 1:  # 如果evaluate = 1 则进行测试
        load_best_checkpoint(args, cnn_model, siamese_model_corr)
        top1 = evaluator.evaluate(dataset.query, dataset.gallery, query_loader, gallery_loader, args.logs_dir, args.visual, args.rerank)
        print('best_rank1', top1)
    else:
        # Trainer  训练器,类的实例化
        tensorboard_train_logdir = osp.join(args.logs_dir, 'train_log')
        remove_repeat_tensorboard_files(tensorboard_train_logdir)

        trainer = SEQTrainer(cnn_model, siamese_model_corr, siamese_model_uncorr, criterion_veri, criterion_oim_corr, criterion_oim_uncorr,
                             tensorboard_train_logdir)

        for epoch in range(args.start_epoch, args.epochs):
            adjust_lr(args, optimizer, epoch)  # 根据epoch,调整学习率
            trainer.train(epoch, train_loader, optimizer)
            if (epoch+1) % 5 == 0 or (epoch+1) == args.epochs or ((epoch+1) > 30 and (epoch+1) % 3 == 0):
                top1 = evaluator.evaluate(dataset.query, dataset.gallery, query_loader, gallery_loader, args.logs_dir, args.visual, args.rerank)
                is_best = top1 > best_top1
                if is_best:
                    best_top1 = top1
                save_checkpoint(args, cnn_model, siamese_model_corr, epoch, best_top1, is_best)
                del top1
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training ResNet Model")

    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=['ilidsvidsequence', 'prid2011sequence', 'mars', 'duke'])
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--seq_srd', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    # MODEL
    # CNN model
    parser.add_argument('--arch1', type=str, default='resnet50_grl',
                        choices=['resnet50_grl', 'resnet50'])
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)
    # Siamese model
    parser.add_argument('--arch2', type=str, default='siamese_attention',
                        choices=models.names())
    parser.add_argument('--arch3', type=str, default='siamese_video',
                        choices=models.names())
    # Criterion model
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=30)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--sampling-rate', type=int, default=3)
    parser.add_argument('--sample_method', type=str, default='rrs')

    # OPTIMIZER
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step', type=float, default=15)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--cnn_resume', type=str, default='', metavar='PATH')

    # TRAINER
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    # EVAL
    parser.add_argument('--evaluate', type=int, default=0)
    parser.add_argument('--visual', type=int, default=0, help='visual the result')
    parser.add_argument('--rerank', type=int, default=0, help='rerank the result')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'log/1'))

    args = parser.parse_args()

    # main function
    main(args)
