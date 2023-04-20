#coding: utf-8

import numpy as np

def bbox_iou_overlaps(b1, b2):
    '''
    :argument
        b1,b2: [n, k], k>=4, x1,y1,x2,y2,...
    :returns
        intersection-over-union pair-wise.
    '''

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
    union_area2 = (union_area1 - inter_area)
    return inter_area / np.maximum(union_area2, 1)


def bbox_iof_overlaps(b1, b2):
    '''
    :argument
        b1,b2: [n, k], k>=4 with x1,y1,x2,y2,....
    :returns
        intersection-over-former-box pair-wise
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    # area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:, np.newaxis], 1)


def center_to_corner(boxes):
    '''
    :argument
        boxes: [N, 4] of center_x, center_y, w, h
    :returns
        boxes: [N, 4] of xmin, ymin, xmax, ymax
    '''
    xmin = boxes[:, 0] - boxes[:, 2] / 2.
    ymin = boxes[:, 1] - boxes[:, 3] / 2.
    xmax = boxes[:, 0] + boxes[:, 2] / 2.
    ymax = boxes[:, 1] + boxes[:, 3] / 2.
    return np.vstack([xmin, ymin, xmax, ymax]).transpose()


def corner_to_center(boxes):
    '''
        inverse of center_to_corner
    '''
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    return np.vstack([cx, cy, w, h]).transpose()


def compute_loc_targets(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets


def compute_loc_bboxes(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, np.log(1000. / 16.))
    dh = np.minimum(dh, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h - 1)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w - 1)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h - 1)
    return bbox


def compute_recall(box_pred, box_gt):
    n_gt = box_gt.shape[0]
    if box_pred.size == 0 or n_gt == 0:
        return 0, n_gt
    # compute recall
    n_dt = box_pred.shape[0]
    ov = bbox_iou_overlaps(box_pred, box_gt)
    max_val = ov.max(1)
    max_ind = ov.argmax(1)
    detected = [False] * n_gt
    n_rc = 0
    for i in range(n_dt):
        # one gt should be matched only once
        if max_val[i] > 0.5 and detected[max_ind[i]] == False:
            detected[max_ind[i]] = True
            n_rc += 1
    assert (n_rc <= n_gt)
    return n_rc, n_gt


def test_bbox_iou_overlaps():
    b1 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5],
                   [100, 100, 200, 200]])
    overlaps = bbox_iou_overlaps(b1, b2)
    print(overlaps)


def test_bbox_iof_overlaps():
    b1 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5],
                   [100, 100, 200, 200]])
    overlaps = bbox_iof_overlaps(b1, b2)
    print(overlaps)


def test_corner_center():
    b1 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [5, 5, 5, 5]])
    b2 = corner_to_center(b1)
    b3 = center_to_corner(b2)
    print(b1)
    print(b2)
    print(b3)


def test_loc_trans():
    b1 = np.array([[0, 0, 4, 4], [1, 2, 3, 5], [4, 4, 5, 5]])
    tg = np.array([[1, 1, 5, 5], [0, 2, 4, 5], [4, 4, 5, 5]])
    deltas = compute_loc_targets(b1, tg)
    print(deltas)
    pred = compute_loc_bboxes(b1, deltas)
    print(pred)


def test_clip_bbox():
    b1 = np.array([[0, 0, 9, 29], [1, 2, 19, 39], [4, 4, 59, 59]])
    print(b1)
    b2 = clip_bbox(b1, (30, 35))
    print(b2)


if __name__ == '__main__':
    # test_corner_center()
    # test_loc_trans()
    test_clip_bbox()
