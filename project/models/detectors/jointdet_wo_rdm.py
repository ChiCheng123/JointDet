import warnings

import torch

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.two_stage import TwoStageDetector

@DETECTORS.register_module()
# class JointDetV1(TwoStageDetector):
class JointDetWoRDM(TwoStageDetector):
    """
    Implementation of JointDet v1, without RDM (Relationship Discriminating Module)
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head_kopf=None,
                 roi_head_body=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head_kopf is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn.kopf if train_cfg is not None else None
            roi_head_kopf.update(train_cfg=rcnn_train_cfg)
            roi_head_kopf.update(test_cfg=test_cfg.rcnn)
            roi_head_kopf.pretrained = pretrained
            self.roi_head_kopf = build_head(roi_head_kopf)

        if roi_head_body is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn.body if train_cfg is not None else None
            roi_head_body.update(train_cfg=rcnn_train_cfg)
            roi_head_body.update(test_cfg=test_cfg.rcnn)
            roi_head_body.pretrained = pretrained
            self.roi_head_body = build_head(roi_head_body)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head_kopf') and self.roi_head_kopf is not None

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head_kopf') and self.roi_head_kopf.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs_kopf, rpn_outs_body = self.rpn_head(x)
            outs = outs + (rpn_outs_kopf, rpn_outs_body)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs_kopf = self.roi_head_kopf.forward_dummy(x, proposals)
        roi_outs_body = self.roi_head_body.forward_dummy(x, proposals)
        outs = outs + (roi_outs_kopf, roi_outs_body)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list_kopf, proposal_list_body = self.rpn_head.forward_train(
                x,
                img_metas,
                kwargs['gt_bboxes_head'],
                gt_labels=None,
                gt_bboxes_ignore=kwargs['gt_bboxes_ignore_head'],
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list_kopf = proposals
            proposal_list_body = proposals

        roi_losses_kopf = self.roi_head_kopf.forward_train(x, img_metas, proposal_list_kopf,
                                                 kwargs['gt_bboxes_head'], kwargs['gt_labels_head'],
                                                 kwargs['gt_bboxes_ignore_head'], gt_masks,
                                                 **kwargs)
        losses.update(roi_losses_kopf)

        # cascade roi head
        roi_losses_body = self.roi_head_body.forward_train(x, img_metas, proposal_list_body,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks)

        if len(roi_losses_kopf.keys()) == len(roi_losses_body.keys()):
            for key in roi_losses_body.keys():
                losses.update({key+'_body': roi_losses_body[key]})
        else:
            losses.update(roi_losses_body)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list_kopf, proposal_list_body = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list_kopf = proposals
            proposal_list_body = proposals

        kopf_test_results = self.roi_head_kopf.simple_test(
                    x, proposal_list_kopf, img_metas, rescale=rescale)
        body_test_results = self.roi_head_body.simple_test(
                    x, proposal_list_body, img_metas, rescale=rescale)

        return body_test_results, kopf_test_results
