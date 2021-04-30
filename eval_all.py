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
# from tensorboardX import SummaryWriter
# import adabound
# utilis
from utils.logging import Logger
from reid import models
from utils.serialization import load_checkpoint, save_cnn_checkpoint, save_siamese_checkpoint
from utils.serialization import remove_repeat_tensorboard_files
from reid.loss import PairLoss, OIMLoss
from reid.data import get_data
from reid.train import SEQTrainer
from reid.evaluator import ATTEvaluator


def save_checkpoint(cnn_model, siamese_model, epoch, best_top1, is_best):
    save_cnn_checkpoint({
        'state_dict': cnn_model.state_dict(),
        'epoch': epoch + 1,
        'best_top1': best_top1,
    }, is_best, fpath=osp.join(args.logs_dir, 'cnn_checkpoint.pth.tar'))

    save_siamese_checkpoint({
        'state_dict': siamese_model.state_dict(),
        'epoch': epoch + 1,
        'best_top1': best_top1,
    }, is_best, fpath=osp.join(args.logs_dir, 'siamese_checkpoint.pth.tar'))


def load_best_checkpoint(cnn_model, siamese_model):
    checkpoint0 = load_checkpoint(osp.join(args.logs_dir, 'cnnmodel_best.pth.tar'))
    cnn_model.load_state_dict(checkpoint0['state_dict'])

    checkpoint1 = load_checkpoint(osp.join(args.logs_dir, 'siamesemodel_best.pth.tar'))
    siamese_model.load_state_dict(checkpoint1['state_dict'])


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # log file 日志文件  防止重名覆盖
    run = 0
    if args.evaluate == 1:
        while osp.exists("%s" % (osp.join(args.logs_dir, 'log_testall{}.txt'.format(run)))):
            run += 1

        sys.stdout = Logger(osp.join(args.logs_dir, 'log_testall{}.txt'.format(run)))
    else:
        while osp.exists("%s" % (osp.join(args.logs_dir, 'log_train{}.txt'.format(run)))):
            run += 1

        sys.stdout = Logger(osp.join(args.logs_dir, 'log_train{}.txt'.format(run)))
    print("==========\nArgs:{}\n==========".format(args))

    # from reid.data import get_data ,根据get_data()函数,返回 数据集,行人数目,封装成batch的训练数据,查询数据,图库数据
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.seq_len, args.seq_srd,
                 args.workers, only_eval=True)

    # create CNN model  1.建立CNN模型,默认是resnet50 , num_features = 128
    cnn_model = models.create(args.a1, num_features=args.features, dropout=args.dropout, numclasses=num_classes)

    # create ATT model  2.建立注意力模型, 默认是attmodel
    input_num = 2048  # 2048，CNN backbone 输出的特征维度
    output_num = args.features  # 最后的特征向量维度
    class_num = 2  # 2分类,正确分类为1,错误分类为0  BCE的类别

    # create Siamese model
    siamese_model = models.create(args.a2, input_num, output_num, class_num)
    # create classifier model  3.建立一个分类模型

    # CUDA acceleration model 4.模型的CUDA声明

    cnn_model = torch.nn.DataParallel(cnn_model).to(device)
    siamese_model = siamese_model.to(device)

    tensorboard_train_logdir = osp.join(args.logs_dir, 'train_log')
    remove_repeat_tensorboard_files(tensorboard_train_logdir)
    # Evaluator  测试

    evaluator = ATTEvaluator(cnn_model, siamese_model, only_eval=True)

    load_best_checkpoint(cnn_model, siamese_model)
    top1 = evaluator.evaluate(dataset.query, dataset.gallery, query_loader, gallery_loader, args.logs_dir1, args.visul,
                              args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training ResNet Model")

    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=['ilidsvidsequence', 'prid2011sequence', 'mars'])
    parser.add_argument('-b', '--batch-size', type=int, default=32)

    parser.add_argument('-j', '--workers', type=int, default=8)

    parser.add_argument('--seq_len', type=int, default=8)

    parser.add_argument('--seq_srd', type=int, default=4)

    parser.add_argument('--split', type=int, default=0)

    # MODEL
    # CNN model
    parser.add_argument('--a1', '--arch_1', type=str, default='resnet50_rga',
                        choices=['resnet50_rga', 'resnet50'])
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Siamese model
    parser.add_argument('--a2', '--arch_2', type=str, default='siamese',
                        choices=models.names())

    # Criterion model
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=20)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--sampling-rate', type=int, default=3)
    parser.add_argument('--sample_method', type=str, default='rrs')

    # OPTIMIZER
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--lr3', type=float, default=1.0)

    parser.add_argument('--lr1step', type=float, default=15)
    parser.add_argument('--lr2step', type=float, default=20)
    parser.add_argument('--lr3step', type=float, default=40)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--cnn_resume', type=str, default='', metavar='PATH')

    # TRAINER
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    # EVAL
    parser.add_argument('--evaluate', type=int, default=1)
    parser.add_argument('--visul', type=int, default=0, help='visul the result')
    parser.add_argument('--rerank', type=int, default=0, help='rerank the result')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/ycy/data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'log/no_rga_6_8*4'))
    parser.add_argument('--logs-dir1', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'log/no_rga_6_8*4/split0'))

    args = parser.parse_args()

    # main function
    main(args)
