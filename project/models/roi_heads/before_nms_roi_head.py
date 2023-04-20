# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead

from copy import deepcopy
import numpy as np

@HEADS.register_module()
class StandardRoIHeadPreNMS(StandardRoIHead):
    """Return both pre-nms and post-nms bboxes."""

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        # pre-nms
        test_cfg_pre_nms = deepcopy(self.test_cfg)
        test_cfg_pre_nms.nms['iou_threshold'] = 1
        test_cfg_pre_nms.max_per_img = 1000

        det_bboxes_pre, det_labels_pre = self.simple_test_bboxes(
            x, img_metas, proposal_list, test_cfg_pre_nms, rescale=rescale)

        bbox_results_pre = [
            bbox2result(det_bboxes_pre[i], det_labels_pre[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes_pre))
        ]

        # post-nms
        det_bboxes_post, det_labels_post = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results_post = [
            bbox2result(det_bboxes_post[i], det_labels_post[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes_post))
        ]

        if not self.with_mask:
            return bbox_results_post, bbox_results_pre
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes_post, det_labels_post, rescale=rescale)
            return list(zip(bbox_results_post, segm_results))


@HEADS.register_module()
class CascadeRoIHeadPreNMS(CascadeRoIHead):
    """Return both pre-nms and post-nms bboxes."""

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually

        # pre-nms
        test_cfg_pre_nms = deepcopy(rcnn_test_cfg)
        test_cfg_pre_nms.nms['iou_threshold'] = 1
        test_cfg_pre_nms.max_per_img = 1000
        det_bboxes_pre = []
        det_labels_pre = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=test_cfg_pre_nms)
            det_bboxes_pre.append(det_bbox)
            det_labels_pre.append(det_label)

        bbox_results_pre = [
            bbox2result(det_bboxes_pre[i], det_labels_pre[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        # post-nms
        det_bboxes_post = []
        det_labels_post = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes_post.append(det_bbox)
            det_labels_post.append(det_label)

        bbox_results_post = [
            bbox2result(det_bboxes_post[i], det_labels_post[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            raise NotImplementedError

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return bbox_results_post, bbox_results_pre


