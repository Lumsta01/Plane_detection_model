_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# Custom classes
classes = ('plane',)

# Dataset settings
dataset_type = 'CocoDataset'

metainfo = {
    'classes': classes,
    'palette': [(0, 255, 0)]  # green box for planes
}

# Data configuration (using full absolute paths correctly)
data = dict(
    train=dict(
        type=dataset_type,
        metainfo=metainfo,
       	ann_file='/home/luluma/Documents/ML_hackathon/mmdetection/data/DOTA/train/instances_train_small.json',
        data_prefix=dict(img='/home/luluma/Documents/ML_hackathon/mmdetection/data/datasets/part1/images'),
    ),
    val=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file='/home/luluma/Documents/ML_hackathon/mmdetection/data/DOTA/val/instances_val_small.json',
        data_prefix=dict(img='/home/luluma/Documents/ML_hackathon/mmdetection/data/datasets_val/images'),
    ),
)

# Output directory for checkpoints/logs
work_dir = './work_dirs/faster-rcnn_r50_fpn_plane'

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# Runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
)

# Evaluation metric
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/luluma/Documents/ML_hackathon/mmdetection/data/DOTA/val/instances_val_small.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/luluma/Documents/ML_hackathon/mmdetection/data/DOTA/val/instances_val_small.json',
    metric='bbox'
)


# Model configuration (1 class only)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1  # only 'plane'
        )
    )
)

# Dataloader config
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=data['train']
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=data['val']
)

test_dataloader = val_dataloader

# Environment and backend config
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

