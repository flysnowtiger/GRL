import argparse
import os
import os.path as osp
import sys

from utils.serialization import load_checkpoint, save_cnn_checkpoint, save_siamese_checkpoint



def save_checkpoint(args, cnn_model, siamese_model, epoch, best_top1, is_best):
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


def load_best_checkpoint(args, cnn_model, siamese_model):
    checkpoint0 = load_checkpoint(osp.join(args.logs_dir, 'cnnmodel_best.pth.tar'))
    cnn_model.load_state_dict(checkpoint0['state_dict'])

    checkpoint1 = load_checkpoint(osp.join(args.logs_dir, 'siamesemodel_best.pth.tar'))
    siamese_model.load_state_dict(checkpoint1['state_dict'])