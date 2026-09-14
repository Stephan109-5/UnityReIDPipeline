from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# NOT USED: DIoU is not used in the current implementation, was considered.
def calculate_diou(box1: Tuple[float, float, float, float],
                   box2: Tuple[float, float, float, float]) -> float:
    iou = calculate_iou(box1, box2)

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    center_x1 = (x1_min + x1_max) / 2.0
    center_y1 = (y1_min + y1_max) / 2.0
    center_x2 = (x2_min + x2_max) / 2.0
    center_y2 = (y2_min + y2_max) / 2.0

    center_distance_sq = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    enclosing_xmin = min(x1_min, x2_min)
    enclosing_ymin = min(y1_min, y2_min)
    enclosing_xmax = max(x1_max, x2_max)
    enclosing_ymax = max(y1_max, y2_max)

    enclosing_diag_sq = (enclosing_xmax - enclosing_xmin) ** 2 + (enclosing_ymax - enclosing_ymin) ** 2
    if enclosing_diag_sq <= 0:
        return iou

    return iou - (center_distance_sq / enclosing_diag_sq)


def match_detections(gt_objects: List[Dict], pred_objects: List[Dict],
                     iou_threshold: float = 0.5, match_mode: str = 'iou'):
    """Match ground-truth objects to predicted objects using an overlap metric.

    Args:
        gt_objects: list of dicts with at least a 'bbox' and 'name' key.
        pred_objects: list of dicts with at least a 'bbox' and 'track_id' key.
        iou_threshold: minimum overlap score to consider a match.
        match_mode: overlap metric to use. Supported values: 'iou', 'diou', 'containment'.

    Returns:
        matches: list of tuples (gt_idx, pred_idx, iou)
        unmatched_gt: list of gt indices not matched
        unmatched_pred: list of pred indices not matched
    """
    if len(gt_objects) == 0 or len(pred_objects) == 0:
        return [], list(range(len(gt_objects))), list(range(len(pred_objects)))

    gt_bboxes = [tuple(obj['bbox']) for obj in gt_objects]
    pred_bboxes = [tuple(obj['bbox']) for obj in pred_objects]

    iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=float)
    inter_area = np.zeros_like(iou_matrix)
    gt_area = np.zeros((len(gt_bboxes),), dtype=float)
    pred_area = np.zeros((len(pred_bboxes),), dtype=float)

    for i, gb in enumerate(gt_bboxes):
        x1, y1, x2, y2 = gb
        gt_area[i] = max(0.0, (x2 - x1) * (y2 - y1))
    for j, pb in enumerate(pred_bboxes):
        x1, y1, x2, y2 = pb
        pred_area[j] = max(0.0, (x2 - x1) * (y2 - y1))
    for i, gb in enumerate(gt_bboxes):
        for j, pb in enumerate(pred_bboxes):
            iou = calculate_iou(gb, pb)
            iou_matrix[i, j] = iou
            # compute intersection area via IoU*union = inter, but easier compute directly
            gx1, gy1, gx2, gy2 = gb
            px1, py1, px2, py2 = pb
            ix1 = max(gx1, px1)
            iy1 = max(gy1, py1)
            ix2 = min(gx2, px2)
            iy2 = min(gy2, py2)

            # if there is no intersection, area is zero. Otherwise compute intersection area as width*height of overlap region
            if ix2 < ix1 or iy2 < iy1:
                inter_area[i, j] = 0.0
            else:
                inter_area[i, j] = (ix2 - ix1) * (iy2 - iy1)

    matches = []
    mode = (match_mode or 'iou').lower()

    # select matching matrix and threshold depending on mode
    if mode == 'containment':
        # coverage: intersection over gt area
        coverage = np.zeros_like(iou_matrix)
        for i in range(inter_area.shape[0]):
            for j in range(inter_area.shape[1]):
                coverage[i, j] = inter_area[i, j] / gt_area[i] if gt_area[i] > 0 else 0.0
        metric_mat = coverage
        thresh = iou_threshold
    elif mode == 'diou':
        diou_matrix = np.zeros_like(iou_matrix)
        for i, gb in enumerate(gt_bboxes):
            for j, pb in enumerate(pred_bboxes):
                diou_matrix[i, j] = calculate_diou(gb, pb)
        metric_mat = diou_matrix
        thresh = iou_threshold
    else:
        metric_mat = iou_matrix
        thresh = iou_threshold

    # Hungarian matching for optimal global assignment, with IoU thresholding to filter out low-quality matches
    if linear_sum_assignment is not None:
        cost = -metric_mat
        gt_idx, pred_idx = linear_sum_assignment(cost)
        for i, j in zip(gt_idx, pred_idx):
            if i < metric_mat.shape[0] and j < metric_mat.shape[1]:
                val = metric_mat[i, j]
                if val >= thresh:
                    # report IoU in matches tuple for downstream use
                    matches.append((int(i), int(j), float(iou_matrix[i, j])))

    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}

    unmatched_gt = [i for i in range(len(gt_objects)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_objects)) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred