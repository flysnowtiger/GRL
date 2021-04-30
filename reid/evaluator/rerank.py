#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @time:2019/6/12上午10:48
# @Author: Yu Ci

"""
Source: https://github.com/zhunzhong07/person-re-ranking

Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking

API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__all__ = ['re_ranking']

import numpy as np


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)  # <class 'tuple'>: (11310, 11310)
    original_dist = np.power(original_dist, 2).astype(np.float32)  # <class 'tuple'>: (11310, 11310)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))  # <class 'tuple'>: (11310, 11310)
    V = np.zeros_like(original_dist).astype(np.float32)  # <class 'tuple'>: (11310, 11310)
    initial_rank = np.argsort(original_dist).astype(np.int32)  # <class 'tuple'>: (11310, 11310)

    query_num = q_g_dist.shape[0]  # 1980
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]  # 11310
    all_num = gallery_num  # 11310

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]  # 21
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]  # <class 'tuple'>: (21, 21)
        fi = np.where(backward_k_neigh_index == i)[0]  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        k_reciprocal_index = forward_k_neigh_index[fi]  # <class 'tuple'>: (20,)
        k_reciprocal_expansion_index = k_reciprocal_index  # <class 'tuple'>: (20,)
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]  # 0
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]  # <class 'tuple'>: (11,)
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2.)) + 1]  # <class 'tuple'>: (11, 11)
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]  # <class 'tuple'>: (7,)
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]  # [   0 5238 5251 5245    1 5252    2]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # <class 'tuple'>: (23,)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # <class 'tuple'>: (23,)
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]  # <class 'tuple'>: (11310, 11310)
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)  # <class 'tuple'>: (11310, 11310)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe  # <class 'tuple'>: (11310, 11310)
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)  # <class 'tuple'>: (1980, 11310)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value  # <class 'tuple'>: (1980, 11310)
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]  # <class 'tuple'>: (1980, 9330)
    return final_dist
