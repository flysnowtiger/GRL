#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @time:2019/6/12上午11:09
# @Author: Yu Ci
__all__ = ['visualize_ranked_results', 'visualize_in_pic']

import torch
import numpy as np
import os
import os.path as osp
import shutil
# import matplotlib.pyplot as plt

from utils.osutils import mkdir_if_missing


def visualize_ranked_results(distmat, queryloader, galleryloader, save_dir='', visual_id=2, topk=10):
    """Visualizes ranked results. 存放在一个文件夹中

    Supports both image-reid and video-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        queryloader (tuple): tuples of (img_path(s), pid, camid).
        galleryloader (tuple): tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        visual_id(int, optional): only show 1 id
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query = queryloader  # 1980个tuple   (img_path(s), pid, camid)）
    gallery = galleryloader  # 9330个tuple (img_path(s), pid, camid)
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)  # <class 'tuple'>: (1980, 9330)
    mkdir_if_missing(save_dir)  # '/home/ying/Desktop/mars_rank/log/debug_for_eval/split0visual'

    def _cp_img_to(src, dst, rank, prefix):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory  # '/home/ying/Desktop/mars_rank/log/debug_for_eval/split0visual/0016C1T0006F001.jpg'
            rank: int, denoting ranked position, starting from 1
            prefix: string （query or gallery）
        """
        if isinstance(src, tuple) or isinstance(src, list):  # video reid
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))  # '/home/ying/Desktop/mars_rank/log/debug_for_eval/split0visual/0016C1T0006F001.jpg/query_top000'
            mkdir_if_missing(dst)
            for img_path in src:  # 将图片copy到目标文件夹中
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):  # 考虑到速度等因素，只输出1个id的rank结果。这个id不是实际的行人id，是在tuple中的顺序
        if q_idx == visual_id:  # 14
            qimg_path, qpid, qcamid = query[q_idx]   # qpid = 16， camid = 0

            if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):  # query_dir 保存Rank结果的文件夹名称 = query的第一张图片名称
                qdir = osp.join(save_dir, osp.basename(qimg_path[0]))  # '/home/ying/Desktop/mars_rank/log/debug_for_eval/split0visual/0016C1T0006F001.jpg'
            else:
                qdir = osp.join(save_dir, osp.basename(qimg_path))
            mkdir_if_missing(qdir)  # 新建这个保存rank结果的文件夹
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')  # 复制query的图片到结果文件夹中

            rank_idx = 1
            for g_idx in indices[q_idx, :]:  # 3291， 3288， 3289， 3290， 3293
                gimg_path, gpid, gcamid = gallery[g_idx]
                invalid = (qpid == gpid) & (qcamid == gcamid)  # true， 排除相同cam的情况
                if not invalid:
                    _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery')
                    rank_idx += 1
                    if rank_idx > topk:
                        break
    print("Done")


def visualize_in_pic(distmat, queryloader, galleryloader, save_dir='', visual_id=2, topk=9):
    """

        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        queryloader (tuple): tuples of (img_path(s), pid, camid).
        galleryloader (tuple): tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        visual_id(int, optional): only show 1 id
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk+1))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query = queryloader  # 1980个tuple   (img_path(s), pid, camid)）
    gallery = galleryloader  # 9330个tuple (img_path(s), pid, camid)
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)  # <class 'tuple'>: (1980, 9330)
    mkdir_if_missing(save_dir)  # '/home/ying/Desktop/mars_rank/log/debug_for_eval/split0visual'

    def imshow(path, title=None):
        """Imshow for Tensor."""
        im = plt.imread(path)
        plt.imshow(im)
        if title is not None:
            plt.title(title)
        # plt.pause(0.001)  # pause a bit so that plots are updated
    flag = 0
    for q_idx in range(num_q):  # 考虑到速度等因素，只输出1个id的rank结果。2,4,6,8,10..
        qimg_path, qpid, qcamid = query[q_idx]  # qpid = 16， camid = 0

        if qpid == visual_id:  # 14
            flag = 1
            fig = plt.figure(figsize=(25, 8))
            ax = plt.subplot(1, 11, 1)
            ax.axis('off')
            imshow(qimg_path[0], 'query, pid:{}'.format(qpid))

            rank_idx = 0
            for g_idx in indices[q_idx, :]:  # 3291， 3288， 3289， 3290， 3293
                gimg_path, gpid, gcamid = gallery[g_idx]
                # invalid = (qpid == gpid) & (qcamid == gcamid)  # true， 排除相同cam的情况
                invalid = False
                if not invalid:
                    rank_idx += 1
                    ax = plt.subplot(1, 11, rank_idx+1)
                    ax.axis('off')
                    imshow(gimg_path[0])
                    if qpid == gpid:
                        ax.set_title('rank:{},pid{}_{}'.format(rank_idx, gpid, gcamid), color='green')
                    else:
                        ax.set_title('rank:{},pid{}_{}'.format(rank_idx, gpid, gcamid), color='red')

                    if rank_idx > topk:
                        break
            fig.savefig("show_{}_{}.png".format(qpid, qcamid))
            break
    if flag == 1:
        print("Done")
    else:
        print("No matched person in query_dataset, try another id")
