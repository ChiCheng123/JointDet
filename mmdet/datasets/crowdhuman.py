import os

import numpy as np
from mmcv.utils import print_log

from .ped_tools.crowdhuman.eval_AP_Recall_MR import _evaluate_predictions_on_crowdhuman
from .coco import CocoDataset
from .builder import DATASETS
import os.path as osp
import mmcv
from collections import defaultdict
import json
import shutil
from copy import deepcopy

@DATASETS.register_module
class CrowdHumanDataset(CocoDataset):

    CLASSES = ('person',)

    def __init__(self, ann_file, pipeline, eval_mode=0, **kwargs):
        super().__init__(ann_file, pipeline, **kwargs)

        self.eval_mode = eval_mode

        self.best_eval_ap_bymr = 0.0
        self.best_eval_mr_bymr = 100
        self.best_eval_rc_bymr = 0.0

        self.image_name_id_dict = self.image_name_to_coco_imgid()


    def image_name_to_coco_imgid(self):
        image_name_id_dict = defaultdict(dict)
        for img in self.coco.imgs.values():
            img_id = img['id']
            img_name = osp.splitext(osp.basename(img['file_name']))[0]

            image_name_id_dict[img_name]['ID'] = img_id
            image_name_id_dict[img_name]['width'] = img['width']
            image_name_id_dict[img_name]['height'] = img['height']

        return image_name_id_dict


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ids = []

        gt_bboxes_head = []
        gt_labels_head = []
        gt_bboxes_ignore_head = []
        gt_ids_head = []

        for i, ann in enumerate(ann_info):
            # if ann.get('ignore', False):
            #     continue
            x1, y1, w, h = ann['bbox']
            hx1, hy1, hw, hh = ann['hbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h <= 0:
                continue
            # if ann['area'] <= 0 or w < 1 or h < 1:
            #     continue
            if w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            hbox = [hx1, hy1, hx1 + hw, hy1 + hh]

            if ann.get('iscrowd_h', False):
                gt_bboxes_ignore_head.append(hbox)  # only add corresponding class ignore
            else:
                gt_bboxes_head.append(hbox)
                gt_labels_head.append(self.cat2label[ann['category_id']])
                gt_ids_head.append(ann['bbox_id'])

            if ann.get('iscrowd_f', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_ids.append(ann['bbox_id'])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ids = np.array(gt_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)

        if gt_bboxes_head:
            gt_bboxes_head = np.array(gt_bboxes_head, dtype=np.float32)
            gt_labels_head = np.array(gt_labels_head, dtype=np.int64)
            gt_ids_head = np.array(gt_ids_head, dtype=np.int64)
        else:
            gt_bboxes_head = np.zeros((0, 4), dtype=np.float32)
            gt_labels_head = np.array([], dtype=np.int64)
            gt_ids_head = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore_head:
            gt_bboxes_ignore_head = np.array(gt_bboxes_ignore_head, dtype=np.float32)
        else:
            gt_bboxes_ignore_head = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            ids=gt_ids,
            bboxes_ignore=gt_bboxes_ignore,
            bboxes_head=gt_bboxes_head,
            labels_head=gt_labels_head,
            ids_head=gt_ids_head,
            bboxes_ignore_head=gt_bboxes_ignore_head,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _parse_ann_info_org(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            # if ann['area'] <= 0 or w < 1 or h < 1:
            #     continue
            if w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)  # only add corresponding class ignore
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_ids.append(ann['bbox_id'])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ids = np.array(gt_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            ids=gt_ids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = defaultdict(dict)
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            json_results[img_id]['ID'] = img_id
            json_results[img_id]['dtboxes'] = list()
            for bboxes in result:
                for box in bboxes:
                    json_results[img_id]['dtboxes'].append({'box': self.xyxy2xywh(box), 'score': float(box[4])})
        return json_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        eval_MRs_dict = {}
        if self.eval_mode == 0 or self.eval_mode == 1:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            gt_path = self.format_gt(tmp_dir)

            # body
            eval_results = _evaluate_predictions_on_crowdhuman(gt_path, result_files['bbox'], mode=self.eval_mode)

            print_log('Checkpoint eval: [AP: %.2f%%], [mMR: %.2f%%], [Recall: %.2f%%]'
                      % (eval_results[0] * 100, eval_results[1] * 100, eval_results[2] * 100), logger=logger)

            if eval_results[1] * 100 < self.best_eval_mr_bymr:
                self.best_eval_ap_bymr = eval_results[0] * 100
                self.best_eval_mr_bymr = eval_results[1] * 100
                self.best_eval_rc_bymr = eval_results[2] * 100

            print_log('Best eval MR: [AP: %.2f%%], [mMR: %.2f%%], [Recall: %.2f%%]'
                      % (self.best_eval_ap_bymr, self.best_eval_mr_bymr, self.best_eval_rc_bymr), logger=logger)

            if tmp_dir is not None:
                tmp_dir.cleanup()
            eval_MRs_dict.update({'AP': eval_results[0], 'mMR': eval_results[1], 'Recall': eval_results[2], })

        else:
            assert self.eval_mode == 2

            results_body = results[::2]
            for i, res in enumerate(results_body):
                results_body[i][0] = results_body[i][0][0]
            results_head = results[1::2]
            for i, res in enumerate(results_head):
                results_head[i][0] = results_head[i][0][0]

            # body
            result_files, tmp_dir = self.format_results(results_body, jsonfile_prefix)
            gt_path = self.format_gt(tmp_dir)

            eval_results = _evaluate_predictions_on_crowdhuman(gt_path, result_files['bbox'], mode=0)

            print_log('Checkpoint eval body: [AP: %.2f%%], [mMR: %.2f%%], [Recall: %.2f%%]'
                      % (eval_results[0] * 100, eval_results[1] * 100, eval_results[2] * 100), logger=logger)

            if eval_results[1] * 100 < self.best_eval_mr_bymr:
                self.best_eval_ap_bymr = eval_results[0] * 100
                self.best_eval_mr_bymr = eval_results[1] * 100
                self.best_eval_rc_bymr = eval_results[2] * 100

            print_log('Best eval MR body: [AP: %.2f%%], [mMR: %.2f%%], [Recall: %.2f%%]'
                      % (self.best_eval_ap_bymr, self.best_eval_mr_bymr, self.best_eval_rc_bymr), logger=logger)

            if tmp_dir is not None:
                tmp_dir.cleanup()
            eval_MRs_dict.update({'AP': eval_results[0], 'mMR': eval_results[1], 'Recall': eval_results[2], })

            # head
            result_files, tmp_dir = self.format_results(results_head, jsonfile_prefix)
            gt_path = self.format_gt(tmp_dir)

            eval_results = _evaluate_predictions_on_crowdhuman(gt_path, result_files['bbox'], mode=1)

            print_log('Checkpoint eval head: [AP: %.2f%%], [mMR: %.2f%%], [Recall: %.2f%%]'
                      % (eval_results[0] * 100, eval_results[1] * 100, eval_results[2] * 100), logger=logger)

            if tmp_dir is not None:
                tmp_dir.cleanup()

            eval_MRs_dict.update({'AP_head': eval_results[0],
                                  'mMR_head': eval_results[1],
                                  'Recall_head':eval_results[2],})

        return eval_MRs_dict


    def format_gt(self, jsonfile_prefix):
        gt_path_odgt = "./datasets/CrowdHuman/annotation_val.odgt"
        with open(gt_path_odgt, "r") as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        for record in records:
            record_id_org = deepcopy(record['ID'])
            record['ID'] = self.image_name_id_dict[record_id_org]['ID']
            record.update({'width': self.image_name_id_dict[record_id_org]['width']})
            record.update({'height': self.image_name_id_dict[record_id_org]['height']})

        jsonfile_prefix = osp.join(jsonfile_prefix.name, 'results')

        mmcv.dump(records, f'{jsonfile_prefix}.gt.pkl')

        return f'{jsonfile_prefix}.gt.pkl'





