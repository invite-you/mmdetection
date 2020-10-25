dataset_type = 'CocoDatasetCar'
data_root = '/content/gdrive/My Drive/Arirang/data/train/custom_coco_all_car/'
imgae_root = '/content/gdrive/My Drive/Arirang/data/train/coco_all/'
img_norm_cfg = dict(
    mean=[54.06, 53.295, 50.235], std=[36.72, 35.955, 33.915], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.1, 1.1),
         saturation_range=(0.1, 1.1),
         hue_delta=18),
    dict(type='CutOut', n_holes=(0, 1),
         cutout_shape=[(250, 100), (250, 250), (100, 250), (100, 100)]),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=imgae_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=imgae_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file= '/content/gdrive/My Drive/Arirang/data/test/instances_test2017.json',
        img_prefix= '/content/gdrive/My Drive/Arirang/data/test/images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
total_epochs = 1000
work_dir = '/content/gdrive/My Drive/Arirang/models/htc_without_semantic_r50_fpn_1x_dota1_5_car/'
load_from = './mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
resume_from = None
