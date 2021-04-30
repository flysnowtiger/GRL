from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from utils import to_torch, to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)  # <class 'tuple'>: (100, 100)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    num_q, num_g = distmat.shape  # 1980,9330
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)  # torch.Size([1980, 9330])

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # torch.Size([1980, 9330])

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # torch.Size([9330])
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel

        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # <class 'tuple'>: (50,)
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_zhengliang(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    cmc = np.zeros((distmat.shape[0], max_rank))  # (1980, 100)
    ap = np.zeros(distmat.shape[0])  # (1980,)

    junk_mask0 = (g_pids == -1)  # gallery中id为-1的样本是无意义的，忽略  9330
    num_valid_q = 0.
    for k in range(distmat.shape[0]):
        score = distmat[k, :]
        good_idx = np.where((q_pids[k] == g_pids) & (q_camids[k] != g_camids))[0]  # 18
        if len(good_idx) == 0:
            num_valid_q = num_valid_q
            continue
        else:
            num_valid_q += 1
        junk_mask1 = ((q_pids[k] == g_pids) & (q_camids[k] == g_camids))
        junk_idx = np.where(junk_mask0 | junk_mask1)[0]
        sort_idx = np.argsort(score)[:max_rank]
        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)

    all_cmc = np.asarray(cmc).astype(np.float32)
    CMC = all_cmc.sum(0) / num_valid_q  # <class 'tuple'>: (50,)

    mAP = np.mean(ap)
    return CMC, mAP


def Compute_AP(good_idx, junk_idx, index):
    cmc = np.zeros((len(index), ))
    num_real = len(good_idx)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):  # rank N
        flag = 0
        if np.any(good_idx == index[n]):
            cmc[n - njunk:] = 1
            flag = 1  # good image
            good_now += 1
        if np.any(junk_idx == index[n]):
            njunk += 1
            continue  # junk image

        if flag == 1:
            intersect_size += 1
        recall = intersect_size / num_real  # 1 / 21 = 0.047
        precision = intersect_size / (j + 1)  # 1
        ap += (recall - old_recall) * (old_precision + precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == num_real:
            return ap, cmc
    return ap, cmc