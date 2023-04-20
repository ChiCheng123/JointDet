from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead


@HEADS.register_module()
class RPNHeadKopf(RPNHead):

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
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)

            return losses, proposal_list

