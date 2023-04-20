import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from project.functions.pair import compute_pairs_targets, get_pairs_matched, get_pairs_matched_alter
from project.functions import accuracy as A
from project.utils import bbox_helper

from copy import deepcopy

from collections import defaultdict

@HEADS.register_module()
class RDM(StandardRoIHead):
    def __init__(self,
                 bbox_roi_extractor=None,
                 relationship_classifier=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RDM, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.relationship_classifier = self.build_relationship_classifier(relationship_classifier) #TODO: initialize classifier
        self.index = 0

        self.heads_deleted = defaultdict(dict)
        self.heads = defaultdict(dict)


    def build_relationship_classifier(self, cfg):
        relationship_classifier = nn.Sequential(
            nn.Linear(cfg.roi_feat_size * cfg.roi_feat_size * cfg.in_channels,
                                 cfg.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fc_out_channels, cfg.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fc_out_channels, cfg.num_classes))

        return relationship_classifier

    def predict(self, x, roi_heads, roi_bodies):
        roi_heads_device = torch.from_numpy(roi_heads.astype(np.float32)).to(x[0].device)
        roi_bodies_devive = torch.from_numpy(roi_bodies.astype(np.float32)).to(x[0].device)

        head_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], roi_heads_device)
        body_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], roi_bodies_devive)
        x = head_bbox_feats + body_bbox_feats  # TODO: check concatenate behaviour
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)

        pred_cls = self.relationship_classifier(x)

        return pred_cls

    def forward_train(self,
                      x,
                      img_metas,
                      head_bbox_post,
                      body_bbox_post,
                      body_bbox_pre,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore,
                      gt_masks,
                      **kwargs):

        sampled_rois_heads, sampled_rois_bodies, cls_targets = \
            compute_pairs_targets(
                head_bbox_post, head_bbox_post, body_bbox_post, body_bbox_pre,
                self.train_cfg, gt_bboxes, img_metas, gt_bboxes_ignore, **kwargs)

        sampled_rois_heads = torch.from_numpy(sampled_rois_heads).to(x[0].device)
        sampled_rois_bodies = torch.from_numpy(sampled_rois_bodies).to(x[0].device)

        head_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], sampled_rois_heads)
        body_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], sampled_rois_bodies)
        x = head_bbox_feats + body_bbox_feats
        c = x.numel() // x.shape[0]
        x = x.view(-1, c)

        pred_cls = self.relationship_classifier(x)

        def f2(x):
            return Variable(torch.from_numpy(x)).cuda()
        [cls_targets] = map(f2, [cls_targets])

        pred_cls = pred_cls.float()
        loss_cls = F.cross_entropy(pred_cls, cls_targets)
        acc = A.accuracy(pred_cls.detach(), cls_targets)[0]

        losses = dict()
        losses['loss_rdm_cls'] = loss_cls
        losses['acc_rdm'] = acc

        return losses

    def simple_test(self, x, img_metas, head_bbox_post, body_bbox_post, body_bbox_pre):
        '''
        single image test
        :param x:
        :param img_metas:
        :param head_bbox_post:
        :param body_bbox_post:
        :param body_bbox_pre:
        :return:
        '''

        self.index += 1
        head_bbox_post_org = deepcopy(head_bbox_post)

        matched_heads, matched_heads_ix, matched_heads_pre_nms, \
        matched_fulls, matched_fulls_ix, matched_fulls_pre_nms, \
        recall_heads_ix, recall_fulls_ix = get_pairs_matched(head_bbox_post, head_bbox_post,
                                                             body_bbox_post, body_bbox_pre,
                                                             img_metas, self.test_cfg)
        assert matched_heads.shape[0] == matched_heads_ix.shape[0]
        assert matched_heads.shape[0] == matched_fulls.shape[0]
        assert matched_heads_pre_nms.shape[0] == recall_fulls_ix.shape[0]

        def to_np_array(x):
            if x is None: return x
            if isinstance(x, Variable): x = x.data
            return x.cpu().float().numpy() if torch.is_tensor(x) else x

        delete_head_restore_thresh = 0.9          # 0.9
        dt_head_fp_ix = []
        dt_head_fp_ix_pre_matched = []

        h_ix_matched = set(np.unique(matched_heads_ix, return_index=False)) | set(
            np.unique(recall_heads_ix, return_index=False))
        h_ix_mismatched = np.array(list(set(range(head_bbox_post[0][0].shape[0])) - h_ix_matched))
        store_idx = []
        for i in range(h_ix_mismatched.shape[0]):
            head_del = head_bbox_post[0][0][h_ix_mismatched[i], :]
            if head_del[4] > delete_head_restore_thresh:
                store_idx.append(i)
        h_ix_mismatched = np.delete(h_ix_mismatched, store_idx, axis=0)

        if matched_heads.shape[0] > 0:
            pred_cls_matched = self.predict(x, matched_heads, matched_fulls)
            pred_cls_matched = F.softmax(pred_cls_matched, dim=1)
            [pred_cls_matched] = map(to_np_array, [pred_cls_matched])

            # remove head fp
            for matched_h_ix in np.unique(matched_heads_ix, return_index=False):
                max_match_score = pred_cls_matched[np.where(matched_heads_ix == matched_h_ix)[0], 1].max()
                if max_match_score < self.test_cfg['rm_head_fp_thresh']:
                    dt_head_fp_ix.append(matched_h_ix)
            dt_head_fp_ix = np.array(dt_head_fp_ix).astype(int)

            # process the low match-score heads
            alter_heads = head_bbox_post[0][0][dt_head_fp_ix, :]
            alter_heads = alter_heads[alter_heads[:, 4] > 0.5]
            if alter_heads.shape[0] > 0:
                matched_heads_alter, matched_heads_ix_alter, matched_fulls_alter, matched_fulls_ix_alter = \
                    get_pairs_matched_alter(alter_heads, body_bbox_pre, img_metas, self.test_cfg)
                if matched_heads_alter.shape[0] > 0:
                    pred_cls_alter = self.predict(x, matched_heads_alter, matched_fulls_alter)
                    pred_cls_alter = F.softmax(pred_cls_alter, dim=1)
                    [pred_cls_alter] = map(to_np_array, [pred_cls_alter])
                    alter_matched_ix = np.where(pred_cls_alter[:, 1] > self.test_cfg['confidence_thresh'])[0]

                    # recall the pre-nms full boxes with low match-score heads
                    if alter_matched_ix.shape[0] > 0:
                        matched_h_pre_ix = matched_heads_ix_alter[alter_matched_ix]
                        matched_f_pre_ix = matched_fulls_ix_alter[alter_matched_ix]
                        dt_full_recall = []
                        dt_full_replace = []
                        dt_head_ix_matched_second = []
                        for dt_head_ix in np.unique(matched_h_pre_ix, return_index=False):
                            dt_full_pre = body_bbox_pre[0][0][matched_f_pre_ix][np.where(matched_h_pre_ix == dt_head_ix)[0]]

                            # recall the full box with the max iof
                            cor_head = alter_heads[dt_head_ix][np.newaxis, :][:, 0:4]
                            iof_h = bbox_helper.bbox_iof_overlaps(cor_head, dt_full_pre[:, 0:4])

                            alter_ix = np.where(iof_h > (iof_h.max(axis=1) - 1e-3))[1]
                            alter_recall_f = dt_full_pre[alter_ix]
                            dt_f_recall = alter_recall_f[alter_recall_f[:, 4].argmax(axis=0)]
                            if dt_f_recall[4] < 0.5:
                                continue
                            dt_full_recall.append(dt_f_recall)
                            iou = bbox_helper.bbox_iou_overlaps(dt_f_recall[0:4][np.newaxis, :], body_bbox_post[0][0][:, 0:4])
                            if np.where(iou > 0.7)[1].shape[0] > 0:
                                # dt_full_replace.append(np.where(iou > 0.7)[1])
                                dt_full_replace.extend(list(np.where(iou > 0.7)[1]))
                            dt_head_ix_matched_second.append(dt_head_ix)

                        if len(dt_full_recall) > 0:
                            dt_full_recall = np.vstack(dt_full_recall)
                            if len(dt_full_replace) > 0:
                                # dt_full_replace = np.vstack(dt_full_replace)
                                dt_full_replace = np.array(dt_full_replace)
                                # body_bbox_post[0][0] = np.delete(body_bbox_post[0][0], np.squeeze(dt_full_replace, axis=1), axis=0)
                                body_bbox_post[0][0] = np.delete(body_bbox_post[0][0],
                                                                 dt_full_replace, axis=0)
                            body_bbox_post[0][0] = np.vstack([body_bbox_post[0][0], dt_full_recall[:,:5]])

                            dt_head_ix_matched_second = np.array(dt_head_ix_matched_second)
                            dt_head_fp_ix = np.delete(dt_head_fp_ix, dt_head_ix_matched_second)

                store_idx = []
                for i in range(dt_head_fp_ix.shape[0]):
                    head_del = head_bbox_post[0][0][dt_head_fp_ix[i], :]
                    if head_del[4] > delete_head_restore_thresh:
                        store_idx.append(i)
                dt_head_fp_ix = np.delete(dt_head_fp_ix, store_idx, axis=0)

        if matched_heads_pre_nms.shape[0] > 0:
            pred_cls_pre_nms = self.predict(x, matched_heads_pre_nms, matched_fulls_pre_nms)
            pred_cls_pre_nms = F.softmax(pred_cls_pre_nms, dim=1)
            [pred_cls_pre_nms] = map(to_np_array, [pred_cls_pre_nms])

            # recall missed full bbox
            matched_h_pre_ix = recall_heads_ix[np.where(pred_cls_pre_nms[:, 1] > self.test_cfg['confidence_thresh'])[0]]
            matched_f_pre_ix = recall_fulls_ix[np.where(pred_cls_pre_nms[:, 1] > self.test_cfg['confidence_thresh'])[0]]
            dt_full_recall = []

            dt_head_fp_ix_pre_matched = list(set(recall_heads_ix) - set(matched_h_pre_ix))

            for dt_head_ix in np.unique(matched_h_pre_ix, return_index=False):
                dt_full_pre = body_bbox_pre[0][0][matched_f_pre_ix][np.where(matched_h_pre_ix == dt_head_ix)[0]]

                # recall the full box with the max iof
                cor_head = head_bbox_post_org[0][0][dt_head_ix][np.newaxis, :][:, 0:4]
                iof_h = bbox_helper.bbox_iof_overlaps(cor_head, dt_full_pre[:, 0:4])

                alter_ix = np.where(iof_h > (iof_h.max(axis=1) - 1e-3))[1]
                alter_recall_f = dt_full_pre[alter_ix]
                dt_f_recall = alter_recall_f[alter_recall_f[:, 4].argmax(axis=0)]
                if dt_f_recall[4] < 0.5:
                    dt_head_fp_ix_pre_matched.append(dt_head_ix)
                    continue

                dt_full_recall.append(dt_f_recall) # TODO: same pre-nms full body might be recalled twice

            dt_head_fp_ix_pre_matched = np.array(dt_head_fp_ix_pre_matched).astype(int)
            store_idx = []
            for i in range(dt_head_fp_ix_pre_matched.shape[0]):
                head_del = head_bbox_post[0][0][dt_head_fp_ix_pre_matched[i], :]
                if head_del[4] > delete_head_restore_thresh:
                    store_idx.append(i)

            dt_head_fp_ix_pre_matched = np.delete(dt_head_fp_ix_pre_matched, store_idx, axis=0)

            if len(dt_full_recall) > 0:
                dt_full_recall = np.vstack(dt_full_recall)

                body_bbox_post[0][0] = np.vstack([body_bbox_post[0][0], dt_full_recall])

        # delete fp head
        if len(dt_head_fp_ix_pre_matched) > 0 or len(dt_head_fp_ix) > 0 or len(h_ix_mismatched) > 0:

            dt_head_fp_to_be_deleted = list(set(dt_head_fp_ix_pre_matched) | set(dt_head_fp_ix) | set(h_ix_mismatched))
            dt_head_fp_to_be_deleted = np.array(dt_head_fp_to_be_deleted).astype(int)

            # delete heads
            proposal_head = np.delete(head_bbox_post[0][0], dt_head_fp_to_be_deleted, axis=0)
            head_bbox_post[0][0] = proposal_head

        return body_bbox_post, head_bbox_post

