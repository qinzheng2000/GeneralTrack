import cv2
import numpy as np
import scipy
import lap
import torch
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time

import torch.nn.functional as F


def sig(x):
    if x < 1.5:
        return 1
    elif x < 5.5:
        return 2
    else:
        return 3

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q

def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    # Complete lifting of category restrictions
    class1 = np.array([track.cls for track in atracks])
    class2 = np.array([track.cls for track in btracks])
    class1_2d = np.tile(class1, (class2.shape[0], 1)).T
    class2_2d = np.tile(class2, (class1.shape[0], 1))
    # 对应位置上的元素是否相等
    matrix = (class1_2d == class2_2d).astype(int)

    _ious = _ious * matrix

    cost_matrix = 1 - _ious

    return cost_matrix

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix

def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def bbox_preprocess(strack_pool, detections, image_key, scale, img_h, img_w):
    detection1 = torch.Tensor([track.observation for track in strack_pool]).cuda()
    detection2 = torch.Tensor([track.tlwh for track in detections]).cuda()
    direction1 = torch.Tensor([track.observation_direction for track in strack_pool]).cuda()
    direction2 = torch.zeros(detection2.shape[0], 2).cuda()
    detection1 = torch.cat((detection1, direction1), 1)
    detection2 = torch.cat((detection2, direction2), 1)
    # detection1 *= scale
    # detection2 *= scale

    # Scale bbox
    h, w = image_key[0].shape[-2:]
    scale_h = h / float(img_h)
    scale_w = w / float(img_w)
    detection1[:, [0, 2, 4]] *= scale_w
    detection1[:, [1, 3, 5]] *= scale_h
    detection2[:, [0, 2, 4]] *= scale_w
    detection2[:, [1, 3, 5]] *= scale_h

    detection1 = detection1.unsqueeze(0)
    detection2 = detection2.unsqueeze(0)

    # Pad detections to match the maximum number of detections
    det_num = max(detection1.shape[1], detection2.shape[1]) + 1
    detection1 = F.pad(detection1, (2, 0, 0, 0, 0, 0), 'constant', 1).to(torch.float32) # placehold, align to gt
    detection2 = F.pad(detection2, (2, 0, 0, 0, 0, 0), 'constant', 1).to(torch.float32)
    detection1 = F.pad(detection1, (0, 0, 0, det_num - detection1.shape[1]), 'constant', -1)
    detection2 = F.pad(detection2, (0, 0, 0, det_num - detection2.shape[1]), 'constant', -1)

    image1 = image_key[0].to(torch.float32)
    image2 = image_key[1].to(torch.float32)

    # Handle out-of-bound detections
    for i in range(detection1.shape[1]):
        for j in range(detection1.shape[0]):
            if detection1[j, i, 2] < 0:  # 左边
                detection1[j, i, 4] += detection1[j, i, 2]
                detection1[j, i, 2] = 0
            if detection2[j, i, 2] < 0:  # 左边
                detection2[j, i, 4] += detection2[j, i, 2]
                detection2[j, i, 2] = 0
            if detection1[j, i, 3] < 0:  # 上
                detection1[j, i, 5] += detection1[j, i, 3]
                detection1[j, i, 3] = 0
            if detection2[j, i, 3] < 0:  # 上
                detection2[j, i, 5] += detection2[j, i, 3]
                detection2[j, i, 3] = 0
            if (detection1[j, i, 2] + detection1[j, i, 4]) > image1.shape[3]:
                detection1[j, i, 4] = image1.shape[3] - detection1[j, i, 2]
            if (detection2[j, i, 2] + detection2[j, i, 4]) > image2.shape[3]:
                detection2[j, i, 4] = image2.shape[3] - detection2[j, i, 2]
            if (detection1[j, i, 3] + detection1[j, i, 5]) > image1.shape[2]:
                detection1[j, i, 5] = image1.shape[2] - detection1[j, i, 3]
            if (detection2[j, i, 3] + detection2[j, i, 5]) > image2.shape[2]:
                detection2[j, i, 5] = image2.shape[2] - detection2[j, i, 3]

    # tlwh-->xywh
    detection1[:, :, 2] += detection1[:, :, 4] / 2
    detection2[:, :, 2] += detection2[:, :, 4] / 2
    detection1[:, :, 3] += detection1[:, :, 5] / 2
    detection2[:, :, 3] += detection2[:, :, 5] / 2

    return image1, image2, detection1, detection2


def classification_distance(strack_pool, detections, image_key, scale, model, img_h, img_w):
    if len(strack_pool)>0 and len(detections)>0:
        image1, image2, detection1, detection2 = bbox_preprocess(strack_pool, detections, image_key, scale, img_h, img_w)

        classification_score, _ = model(image1, image2, detection1, detection2)
        classification_score = torch.sigmoid(classification_score.squeeze()).detach().cpu().numpy().reshape((len(strack_pool), len(detections)))

        # Relaxation of category restrictions
        class1 = np.array([sig(track.cls) for track in strack_pool])
        class2 = np.array([sig(track.cls) for track in detections])
        # Complete lifting of category restrictions
        # a = np.array([track.cls for track in strack_pool])
        # b = np.array([track.cls for track in detections])

        class1_2d = np.tile(class1, (class2.shape[0], 1)).T
        class2_2d = np.tile(class2, (class1.shape[0], 1))
        matrix = (class1_2d == class2_2d).astype(int)

        classification_score = classification_score * matrix
        cost_matrix = 1 - classification_score
    else:
        cost_matrix = np.zeros((len(strack_pool) , len(detections)))

    return cost_matrix