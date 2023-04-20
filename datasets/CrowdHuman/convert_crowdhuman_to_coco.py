import os
import datetime
import json
import numpy as np
import glob
from PIL import Image
import sys,inspect
import cv2
from collections import defaultdict
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
sys.path.append("../../pycococreator-0.2.1")
from pycococreatortools import pycococreatortools
import mmcv


INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2019,
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
]


def outside(bbox, img_h, img_w):
    '''
    check whether bbox center within image scope
    :param bbox: [x, y, w, h]
    :param img_h:
    :param img_w:
    :return:
    '''
    cx = (bbox[0] + bbox[0] + bbox[2] - 1) / 2
    cy = (bbox[1] + bbox[1] + bbox[3] - 1) / 2
    if cx < 0 or cx > img_w or cy < 0 or cy > img_h:
        return True
    else:
        return False


def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h - 1)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w - 1)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h - 1)
    return bbox


def parse_crowdhuman_odgt(file_path, phase):
    assert phase in ['train', 'val']
    with open(file_path) as f:
        lines = f.readlines()

    name_bbs_dict = defaultdict(list)

    count = 1

    for ll in lines:
        dat = json.loads(ll)
        img_name = dat['ID'] + '.jpg'
        root_path = os.path.join(os.path.dirname(file_path), phase)
        img = cv2.imread(os.path.join(root_path, img_name))
        img_h, img_w, _ = img.shape
        boxes = dat['gtboxes']

        for box in boxes:

            new_box = dict()
            new_box['hbox'] = box['hbox']
            new_box['fbox'] = box['fbox']
            new_box['vbox'] = box['vbox']

            if box['tag'] == 'person':
                new_box['isCrowd_f'] = False
                new_box['isCrowd_h'] = False
            else:
                new_box['isCrowd_f'] = True
                new_box['isCrowd_h'] = True
            if 'extra' in box:
                if 'ignore' in box['extra']:
                    if box['extra']['ignore'] != 0:
                        new_box['isCrowd_f'] = True
                        new_box['isCrowd_h'] = True
            if 'head_attr' in box:
                if 'ignore' in box['head_attr']:
                    if box['head_attr']['ignore'] != 0:
                        new_box['isCrowd_h'] = True

            bb = box['hbox']
            if outside(bb, img_h, img_w):
                new_box['isCrowd_h'] = True
            bb = box['fbox']
            if outside(bb, img_h, img_w):
                new_box['isCrowd_f'] = True

            new_box['boxID'] = count
            count += 1
            name_bbs_dict[img_name].append(new_box)

    print('Parsed {} bboxes in {} images average ({:.2f} boxes/img) '.format(count,
                                                                             len(name_bbs_dict.keys()),
                                                                             count / len(name_bbs_dict.keys())))
    return name_bbs_dict


def convert(phase, data_path, anno_mat_path):
    assert phase in ['train', 'val']

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    bbs_dict = parse_crowdhuman_odgt(os.path.join(data_path, anno_mat_path), phase)

    fn_lst = glob.glob('%s/%s/*.jpg' % (data_path, phase))

    for image_filename in fn_lst:
        base_name = os.path.basename(image_filename)

        if base_name in bbs_dict:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, image_filename.split(data_path)[-1][1:], image.size)
            coco_output["images"].append(image_info)

            boxes = bbs_dict[base_name]

            for box in boxes:
                annotation_info = pycococreatortools.create_annotation_info_crowdhuman(
                    annotation_id, image_id, box, image.size
                )
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1

        image_id = image_id + 1

    with open(os.path.join(data_path, phase + '.json'), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == '__main__':
    data_path = './'
    convert(phase='val', data_path=data_path, anno_mat_path="annotation_val.odgt")
    convert(phase='train', data_path=data_path, anno_mat_path="annotation_train.odgt")
