import numpy as np
import torch
from torch.autograd import Variable
from project.utils import bbox_helper

def to_np_array(x):
    if x is None: return x
    if isinstance(x, Variable): x = x.data
    return x.cpu().float().numpy() if torch.is_tensor(x) else x


def compute_pairs_targets(proposal_head, proposal_head_pre_nms, proposal_full, proposal_full_pre_nms,
                          cfg, ground_truth_bboxes, image_info, ignore_regions, **kwargs):
    B = len(ground_truth_bboxes)
    batch_rois_heads = []
    batch_rois_fulls = []
    batch_labels = []

    for b_ix in range(B):
        pros_head = proposal_head[b_ix][0][:, 0:4]
        pros_full = proposal_full[b_ix][0][:, 0:4]
        pros_full_pre_nms = proposal_full_pre_nms[b_ix][0][:, 0:4]
        gts_fulls = ground_truth_bboxes[b_ix]
        gts_fulls_id = kwargs['gt_ids'][b_ix]
        gts_fulls_id = gts_fulls_id[(gts_fulls[:,2] > gts_fulls[:,0]+1e-6) & (gts_fulls[:,3] > gts_fulls[:,1]+1e-6)]
        gts_fulls = gts_fulls[(gts_fulls[:,2] > gts_fulls[:,0]+1e-6) & (gts_fulls[:,3] > gts_fulls[:,1]+1e-6)]
        gts_fulls = to_np_array(gts_fulls)

        gts_heads = kwargs['gt_bboxes_head'][b_ix]
        gts_heads_id = kwargs['gt_ids_head'][b_ix]
        gts_heads_id = gts_heads_id[(gts_heads[:, 2] > gts_heads[:, 0] + 1e-6) & (gts_heads[:, 3] > gts_heads[:, 1] + 1e-6)]
        gts_heads = gts_heads[(gts_heads[:, 2] > gts_heads[:, 0] + 1e-6) & (gts_heads[:, 3] > gts_heads[:, 1] + 1e-6)]
        gts_heads = to_np_array(gts_heads)

        matched_heads = []
        matched_fulls = []
        overlaps = bbox_helper.bbox_iof_overlaps(pros_head, pros_full)
        matched_heads.append(pros_head[np.where(overlaps > cfg['iof_thresh'])[0]])
        matched_fulls.append(pros_full[np.where(overlaps > cfg['iof_thresh'])[1]])

        max_overlaps = overlaps.max(axis=1)
        mismatch_ix = np.where(max_overlaps < cfg['iof_thresh'])[0]
        mismatch_pros_head = pros_head[mismatch_ix]

        overlaps_pre_nms = bbox_helper.bbox_iof_overlaps(mismatch_pros_head, pros_full_pre_nms)
        matched_heads.append(mismatch_pros_head[np.where(overlaps_pre_nms > cfg['iof_thresh'])[0]])
        matched_fulls.append(pros_full_pre_nms[np.where(overlaps_pre_nms > cfg['iof_thresh'])[1]])

        matched_heads = np.vstack(matched_heads)
        matched_fulls = np.vstack(matched_fulls)

        if gts_heads.shape[0] > 0 or gts_fulls.shape[0] > 0:
            overlaps_heads_gts = bbox_helper.bbox_iou_overlaps(matched_heads, gts_heads)
            overlaps_fulls_gts = bbox_helper.bbox_iou_overlaps(matched_fulls, gts_fulls)

            labels = []
            for i in range(matched_heads.shape[0]):
                gt_head_ix = np.where(overlaps_heads_gts[i] > cfg['iou_thresh_match_gts'])
                gt_head_num = gts_heads_id[gt_head_ix]

                gt_full_ix = np.where(overlaps_fulls_gts[i] > cfg['iou_thresh_match_gts'])
                gt_full_num = gts_fulls_id[gt_full_ix]

                inter = set(to_np_array(gt_head_num)) & set(to_np_array(gt_full_num))
                if len(inter) == 0:
                    labels.append(0)
                else:
                    labels.append(1)

            labels = np.array(labels)
            pos_ix = np.where(labels == 1)[0]
            neg_ix = np.where(labels == 0)[0]
        else:
            pos_ix = np.array([])
            neg_ix = np.arange(matched_heads.shape[0])

        num_positives = pos_ix.shape[0]
        batch_size_per_image = cfg['batch_size']

        num_pos_sampling = int(cfg['positive_percent'] * batch_size_per_image)
        if (True and num_positives>0) or num_pos_sampling < num_positives:
            num_pos_sampling = min(num_pos_sampling, num_positives)
            keep_ix = np.random.choice(num_positives, size = num_pos_sampling, replace = False)
            pos_ix = pos_ix[keep_ix]
            num_positives = num_pos_sampling

        num_negatives = neg_ix.shape[0]
        num_neg_sampling = batch_size_per_image - num_positives
        num_neg_sampling = min(num_neg_sampling, num_negatives)
        if (True and num_negatives>0) or num_neg_sampling < num_negatives:
            keep_ix = np.random.choice(num_negatives, size = num_neg_sampling, replace = False)
            neg_ix = neg_ix[keep_ix]

        pos_ix = list(pos_ix)
        neg_ix = list(neg_ix)

        # gather positives, matched gts, and negatives
        pos_rois_head = matched_heads[pos_ix]
        neg_rois_head = matched_heads[neg_ix]
        pos_rois_full = matched_fulls[pos_ix]
        neg_rois_full = matched_fulls[neg_ix]
        rois_sampling_heads = np.vstack([pos_rois_head, neg_rois_head])
        rois_sampling_fulls = np.vstack([pos_rois_full, neg_rois_full])
        num_pos, num_neg = pos_rois_head.shape[0], neg_rois_head.shape[0]

        num_sampling = num_pos + num_neg

        if gts_heads.shape[0] > 0 or gts_fulls.shape[0] > 0:
            pos_labels = labels[pos_ix].astype(np.int32)
            neg_labels = labels[neg_ix]
            labels = np.concatenate([pos_labels, neg_labels]).astype(np.int32)
        else:
            # No gts
            # All rois are negatives
            labels = np.zeros(num_sampling).astype(np.int32)

        batch_ix = np.full(rois_sampling_heads.shape[0], b_ix)
        rois_sampling_heads = np.hstack([batch_ix[:, np.newaxis], rois_sampling_heads])
        rois_sampling_fulls = np.hstack([batch_ix[:, np.newaxis], rois_sampling_fulls])

        batch_rois_heads.append(rois_sampling_heads)
        batch_rois_fulls.append(rois_sampling_fulls)
        batch_labels.append(labels)

    batch_rois_heads = np.vstack(batch_rois_heads).astype(np.float32)
    batch_rois_fulls = np.vstack(batch_rois_fulls).astype(np.float32)
    batch_labels = np.concatenate(batch_labels).astype(np.int64)

    return batch_rois_heads, batch_rois_fulls, batch_labels

def get_pairs_matched(proposal_head, proposal_head_pre_nms, proposal_full, proposal_full_pre_nms,
                           image_info, cfg):

    B = len(proposal_full)

    batch_matched_heads = []
    batch_matched_heads_ix = []
    batch_matched_heads_pre_nms = []
    batch_matched_fulls = []
    batch_matched_fulls_ix = []
    batch_matched_fulls_pre_nms = []
    batch_recall_heads_ix = []
    batch_recall_fulls_ix = []

    for b_ix in range(B):

        pros_head = proposal_head[b_ix][0][:, 0:4]
        pros_full = proposal_full[b_ix][0][:, 0:4]
        pros_full_pre_nms = proposal_full_pre_nms[b_ix][0][:, 0:4]

        pros_head_ix= np.arange(pros_head.shape[0])
        pros_full_ix= np.arange(pros_full.shape[0])
        pros_full_pre_nms_ix = np.arange(pros_full_pre_nms.shape[0])

        overlaps = bbox_helper.bbox_iof_overlaps(pros_head, pros_full)
        # store the matched proposal pairs
        matched_heads = pros_head[np.where(overlaps > cfg['iof_thresh'])[0]]
        matched_heads_ix = np.where(overlaps > cfg['iof_thresh'])[0]
        matched_fulls = pros_full[np.where(overlaps > cfg['iof_thresh'])[1]]
        matched_fulls_ix = np.where(overlaps > cfg['iof_thresh'])[1]

        # make sure every dt_full have matched dt_head
        max_overlaps = overlaps.max(axis=1)
        mismatch_ix = np.where(max_overlaps < cfg['iof_thresh'])[0] # head that with no body higher than iof thresh

        matched_fulls_ix_u = np.unique(matched_fulls_ix, return_index=False)
        missed_fulls_ix = np.array(list(set(pros_full_ix) - set(matched_fulls_ix_u))).astype(int)
        max_iof_by_full = overlaps[:, missed_fulls_ix].max(axis=0)
        for i in range(max_iof_by_full.shape[0]):
            if max_iof_by_full[i] > 0.2:
                head_ix = np.where(overlaps[:, missed_fulls_ix[i]] == max_iof_by_full[i])[0]
                matched_fulls = np.concatenate([matched_fulls, pros_full[missed_fulls_ix[i]][np.newaxis, :]], axis=0)
                matched_fulls_ix = np.append(matched_fulls_ix, missed_fulls_ix[i])
                matched_heads = np.concatenate([matched_heads, pros_head[head_ix]], axis=0)
                matched_heads_ix = np.append(matched_heads_ix, head_ix)
                mismatch_ix = np.setdiff1d(mismatch_ix, head_ix) # Return the unique values in ar1 that are not in ar2.

        # check the position relationship between matched head and full body
        center_x_head = (matched_heads[:, 0] + matched_heads[:, 2]) / 2
        center_y_head = (matched_heads[:, 1] + matched_heads[:, 3]) / 2
        center_head = np.concatenate([center_x_head[np.newaxis, :], center_y_head[np.newaxis, :]], axis=0)
        assert center_head.shape[1] == matched_fulls_ix.shape[0]
        eliminate_ix = []
        for i in range(center_head.shape[1]):
            center_y = center_head[1, i]
            full = matched_fulls[i]
            center_y = (center_y - full[1]) / (full[3] - full[1])
            if center_y <= 0.5:
                continue
            else:
                eliminate_ix.append(i)  # head center falls in lower half of body
        eliminate_ix = np.array(eliminate_ix)
        if eliminate_ix.shape[0] > 0:
            matched_heads = np.delete(matched_heads, eliminate_ix, axis=0)
            matched_fulls = np.delete(matched_fulls, eliminate_ix, axis=0)
            matched_heads_ix = np.delete(matched_heads_ix, eliminate_ix, axis=0)
            matched_fulls_ix = np.delete(matched_fulls_ix, eliminate_ix, axis=0)
        mismatch_ix = np.setdiff1d(pros_head_ix, matched_heads_ix)

        # pick up the missing head proposals
        mismatch_pros_head = pros_head[mismatch_ix]

        # find the matched pair in the pre nms full proposals
        overlaps_pre_nms = bbox_helper.bbox_iof_overlaps(mismatch_pros_head, pros_full_pre_nms)
        matched_heads_pre_nms = mismatch_pros_head[np.where(overlaps_pre_nms > cfg['iof_thresh'])[0]]
        recall_heads_ix = pros_head_ix[mismatch_ix[np.where(overlaps_pre_nms > cfg['iof_thresh'])[0]]]
        matched_fulls_pre_nms = pros_full_pre_nms[np.where(overlaps_pre_nms > cfg['iof_thresh'])[1]]
        recall_fulls_ix = pros_full_pre_nms_ix[np.where(overlaps_pre_nms > cfg['iof_thresh'])[1]]

        # use the same position limitation as the former
        center_x_head = (matched_heads_pre_nms[:, 0] + matched_heads_pre_nms[:, 2]) / 2
        center_y_head = (matched_heads_pre_nms[:, 1] + matched_heads_pre_nms[:, 3]) / 2
        center_head = np.concatenate([center_x_head[np.newaxis, :], center_y_head[np.newaxis, :]], axis=0)
        assert center_head.shape[1] == recall_fulls_ix.shape[0]
        eliminate_ix = []
        for i in range(center_head.shape[1]):
            center_y = center_head[1, i]
            full = matched_fulls_pre_nms[i]
            center_y = (center_y - full[1]) / (full[3] - full[1])
            if center_y <= 0.5:
                continue
            else:
                eliminate_ix.append(i)
        eliminate_ix = np.array(eliminate_ix)
        if eliminate_ix.shape[0] > 0:
            matched_heads_pre_nms = np.delete(matched_heads_pre_nms, eliminate_ix, axis=0)
            matched_fulls_pre_nms = np.delete(matched_fulls_pre_nms, eliminate_ix, axis=0)
            recall_heads_ix = np.delete(recall_heads_ix, eliminate_ix, axis=0)
            recall_fulls_ix = np.delete(recall_fulls_ix, eliminate_ix, axis=0)

        batch_ix = np.full(matched_heads.shape[0], b_ix)
        matched_heads = np.hstack([batch_ix[:, np.newaxis], matched_heads])
        matched_fulls = np.hstack([batch_ix[:, np.newaxis], matched_fulls])
        batch_ix_pre = np.full(matched_heads_pre_nms.shape[0], b_ix)
        matched_heads_pre_nms = np.hstack([batch_ix_pre[:, np.newaxis], matched_heads_pre_nms])
        matched_fulls_pre_nms = np.hstack([batch_ix_pre[:, np.newaxis], matched_fulls_pre_nms])

        batch_matched_heads.append(matched_heads)
        batch_matched_heads_ix.append(matched_heads_ix)
        batch_matched_heads_pre_nms.append(matched_heads_pre_nms)
        batch_matched_fulls.append(matched_fulls)
        batch_matched_fulls_ix.append(matched_fulls_ix)
        batch_matched_fulls_pre_nms.append(matched_fulls_pre_nms)
        batch_recall_heads_ix.append(recall_heads_ix)
        batch_recall_fulls_ix.append(recall_fulls_ix)

    batch_matched_heads = np.vstack(batch_matched_heads).astype(np.float32)
    batch_matched_heads_ix = np.concatenate(batch_matched_heads_ix, axis=0)
    batch_matched_heads_pre_nms = np.vstack(batch_matched_heads_pre_nms).astype(np.float32)
    batch_matched_fulls = np.vstack(batch_matched_fulls).astype(np.float32)
    batch_matched_fulls_ix = np.concatenate(batch_matched_fulls_ix, axis=0)
    batch_matched_fulls_pre_nms = np.vstack(batch_matched_fulls_pre_nms).astype(np.float32)
    batch_recall_heads_ix = np.concatenate(batch_recall_heads_ix, axis=0)
    batch_recall_fulls_ix = np.concatenate(batch_recall_fulls_ix, axis=0)
    return batch_matched_heads, batch_matched_heads_ix, batch_matched_heads_pre_nms, \
           batch_matched_fulls, batch_matched_fulls_ix, batch_matched_fulls_pre_nms, \
           batch_recall_heads_ix, batch_recall_fulls_ix

def get_pairs_matched_alter(proposal_head, proposal_full_pre_nms, image_info, cfg):
    B = len(proposal_full_pre_nms)

    batch_matched_heads = []
    batch_matched_heads_ix = []
    batch_matched_fulls = []
    batch_matched_fulls_ix = []
    for b_ix in range(B):
        pros_head_ix = np.arange(proposal_head.shape[0])
        pros_head = proposal_head[:, 0:4]
        pros_full_pre_nms_ix = np.arange(proposal_full_pre_nms[b_ix][0].shape[0])
        pros_full_pre_nms = proposal_full_pre_nms[b_ix][0][:, 0:4]

        overlaps = bbox_helper.bbox_iof_overlaps(pros_head, pros_full_pre_nms)
        # store the matched proposal pairs
        matched_heads = pros_head[np.where(overlaps > cfg['iof_thresh'])[0]]
        matched_heads_ix = pros_head_ix[np.where(overlaps > cfg['iof_thresh'])[0]]
        matched_fulls = pros_full_pre_nms[np.where(overlaps > cfg['iof_thresh'])[1]]
        matched_fulls_ix = pros_full_pre_nms_ix[np.where(overlaps > cfg['iof_thresh'])[1]]

        # check the position relationship between matched head and full body
        center_x_head = (matched_heads[:, 0] + matched_heads[:, 2]) / 2
        center_y_head = (matched_heads[:, 1] + matched_heads[:, 3]) / 2
        center_head = np.concatenate([center_x_head[np.newaxis, :], center_y_head[np.newaxis, :]], axis=0)
        assert center_head.shape[1] == matched_fulls_ix.shape[0]
        eliminate_ix = []
        for i in range(center_head.shape[1]):
            center_y = center_head[1, i]
            full = matched_fulls[i]
            center_y = (center_y - full[1]) / (full[3] - full[1])
            if center_y <= 0.5:
                continue
            else:
                eliminate_ix.append(i)
        eliminate_ix = np.array(eliminate_ix)
        if eliminate_ix.shape[0] > 0:
            matched_heads = np.delete(matched_heads, eliminate_ix, axis=0)
            matched_fulls = np.delete(matched_fulls, eliminate_ix, axis=0)
            matched_heads_ix = np.delete(matched_heads_ix, eliminate_ix, axis=0)
            matched_fulls_ix = np.delete(matched_fulls_ix, eliminate_ix, axis=0)

        batch_ix = np.full(matched_heads.shape[0], b_ix)
        matched_heads = np.hstack([batch_ix[:, np.newaxis], matched_heads])
        matched_fulls = np.hstack([batch_ix[:, np.newaxis], matched_fulls])

        batch_matched_heads.append(matched_heads)
        batch_matched_heads_ix.append(matched_heads_ix)
        batch_matched_fulls.append(matched_fulls)
        batch_matched_fulls_ix.append(matched_fulls_ix)

    batch_matched_heads = np.vstack(batch_matched_heads)
    batch_matched_heads_ix = np.concatenate(batch_matched_heads_ix, axis=0)
    batch_matched_fulls = np.vstack(batch_matched_fulls)
    batch_matched_fulls_ix = np.concatenate(batch_matched_fulls_ix, axis=0)
    return batch_matched_heads, batch_matched_heads_ix, \
           batch_matched_fulls, batch_matched_fulls_ix