from __future__ import print_function, absolute_import
import time
import math
import torch
from utils.meters import AverageMeter
from utils import to_torch
from .eva_functions import cmc, mean_ap, evaluate, evaluate_zhengliang
from .rerank import re_ranking
from .visualize import visualize_ranked_results, visualize_in_pic
import numpy as np
from torch import nn
import scipy.io


def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, path, cmc_topk=[1, 5, 10, 20]):
    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    cmc_scores, mAP = evaluate(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    for r in cmc_topk:
        print("Rank-{:<3}: {:.1%}".format(r, cmc_scores[r-1]))
    print("------------------")

    return cmc_scores[0]


def pairwise_distance_tensor(query_x, gallery_x):
    m, n = query_x.size(0), gallery_x.size(0)
    x = query_x.view(m, -1)
    y = gallery_x.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) +\
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()  # torch.Size([1980, 9330])
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosin_dist(qf, gf):
    dist = -torch.mm(qf, gf.t())
    return dist


class ATTEvaluator(object):

    def __init__(self, cnn_model, Siamese_model_corr):
        super(ATTEvaluator, self).__init__()
        self.cnn_model = cnn_model
        self.siamese_model_corr = Siamese_model_corr
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def extract_feature(self, data_loader):  # 2
        # print_freq = 50
        self.cnn_model.eval()
        self.siamese_model_corr.eval()

        qf, q_pids, q_camids = [], [], []

        for i, inputs in enumerate(data_loader):
            imgs, pids, camids = inputs  # torch.Size([1, 8, 3, 256, 128])

            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b, s, c, h, w)
            imgs = to_torch(imgs)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            imgs = imgs.to(device)

            with torch.no_grad():

                x_uncorr, x_corr = self.cnn_model(imgs)
                x_corr_atte = self.siamese_model_corr.self_attention(x_corr)

                out_feat = torch.cat((x_uncorr, x_corr_atte, x_corr.mean(dim=1)), dim=1)

                qf.append(out_feat)
                q_pids.extend(pids)
                q_camids.extend(camids)
            torch.cuda.empty_cache()

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        # print(qf.size())
        return qf, q_pids, q_camids

    def evaluate(self, query, gallery, query_loader, gallery_loader, path, visual, rerank):
        # 1
        rerank = rerank
        path = path

        if visual:  # 节约时间，直接加载distmat，用于可视化
            result = scipy.io.loadmat(path+'dist.mat')
            distmat = result['distmat']
            save_dir = path + 'visual'
            visual_id = 4
            visualize_in_pic(distmat, query, gallery, save_dir, visual_id)

        else:
            qf, q_pids, q_camids = self.extract_feature(query_loader)  # 1980 * 128
            torch.cuda.empty_cache()
            print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

            gf, g_pids, g_camids = self.extract_feature(gallery_loader)  # torch.Size([9330, 128])
            gf = torch.cat((qf, gf), 0)
            g_pids = np.append(q_pids, g_pids)
            g_camids = np.append(q_camids, g_camids)
            print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

            print("Computing distance matrix")
            distmat = cosin_dist(qf, gf).cpu().numpy()  # torch.Size([1980, 9330])
            if rerank:
                print('Applying person re-ranking ...')
                distmat_qq = pairwise_distance_tensor(qf, qf).cpu().numpy()  # torch.Size([1980, 1980])
                distmat_gg = pairwise_distance_tensor(gf, gf).cpu().numpy()  # torch.Size([9330, 9330])
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        del query_loader
        del gallery_loader
        final = evaluate_seq(distmat, q_pids, q_camids, g_pids, g_camids, path)
        torch.cuda.empty_cache()
        return final
