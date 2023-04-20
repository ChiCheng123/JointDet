# dataset settings
dataset_type = 'CrowdHumanDataset'
data_root = 'datasets/CrowdHuman/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithBoxID', with_bbox=True, with_head=True, with_boxid=True),
    dict(type='PhotoMetricDistortion', brightness_delta=180, contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Resize', img_scale=(1400, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='WithBoxIdDSFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_head', 'gt_labels', 'gt_labels_head',
                               'gt_bboxes_ignore', 'gt_bboxes_ignore_head', 'gt_ids', 'gt_ids_head']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1400, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox')
