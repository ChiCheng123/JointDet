from mmcv.runner import force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead


@HEADS.register_module()
class RPNHeadKopfAndBody(RPNHead):
    """
    Generate head region proposal and corresponding statistical body region proposal,
    refer to JointDetv1.
    """

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (kwargs['gt_bboxes_head'], img_metas)
        else:
            loss_inputs = outs + (kwargs['gt_bboxes_head'], kwargs['gt_labels_head'], img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=kwargs['gt_bboxes_ignore_head'])
        if proposal_cfg is None:
            return losses
        else:
            proposal_list_kopf, proposal_list_body = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)

            return losses, proposal_list_kopf, proposal_list_body

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):

        proposal_list_kopf = super(RPNHeadKopfAndBody, self).get_bboxes(cls_scores,
                                                               bbox_preds,
                                                               score_factors,
                                                               img_metas,
                                                               cfg,
                                                               rescale,
                                                               with_nms,
                                                               **kwargs)

        # generate statistical body proposal corresponding to head proposal
        proposal_list_body = [proposal.clone().detach() for proposal in proposal_list_kopf]

        for i in range(len(proposal_list_kopf)):
            height_head = proposal_list_body[i][:, 3] - proposal_list_body[i][:, 1] + 1
            width_head = proposal_list_body[i][:, 2] - proposal_list_body[i][:, 0] + 1
            height_full = height_head / 15 * 100
            width_full = width_head / 30 * 100
            proposal_list_body[i][:, 0] -= 0.35 * width_full
            proposal_list_body[i][:, 1] -= 0.05 * height_full
            proposal_list_body[i][:, 2] = proposal_list_body[i][:, 0] + width_full - 1
            proposal_list_body[i][:, 3] = proposal_list_body[i][:, 1] + height_full - 1

            proposal_list_body[i][..., 0::2].clamp_(min=0, max=img_metas[i]['img_shape'][1])
            proposal_list_body[i][..., 1::2].clamp_(min=0, max=img_metas[i]['img_shape'][0])

        return proposal_list_kopf, proposal_list_body
