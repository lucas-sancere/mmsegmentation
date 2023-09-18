_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/mc_sccskin184.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_SCC_segmenter_160k.py'
]

#checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'  # noqa

checkpoint = '/data/lsancere/Ada_Codes/mmsegmentation/checkpoints/segmenter_vit-l_SCCJohannes_mask_8x1_640x640_160k_ade20k_23-05-23.pth'  #We take the resulting checkpoints from
# the training with non melanoma dataset. Maybe the path is not correct (to test)

model = dict(
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(640, 640),
        embed_dims=1024,
        num_layers=24,
        num_heads=16),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_classes=2,
        out_channels=2,
        num_heads=16,
        dropout_ratio=0.0,
        embed_dims=1024,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)))

optimizer = dict(lr=0.001, weight_decay=0.0)


img_norm_cfg = dict( # This img_norm_cfg is widely used because it is mean and std of ImageNet 1K pretrained model
    mean=[127.5, 127.5, 127.5], std=[57.5, 57.5, 57.5], to_rgb=True)

crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(5120, 5120), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(5120, 5120),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(5120, 5120),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]), 
]


data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=1, #Bacth Size
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_pipeline),
    test=dict(pipeline=test_pipeline))
