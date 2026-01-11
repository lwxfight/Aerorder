_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/data'
data_root_val = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/data'

# ann_file_train = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/all_list/train.txt'
# ann_file_val = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/all_list/val.txt'
# ann_file_test = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/all_list/test.txt'

ann_file_train = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/view-all/train.txt'
ann_file_val = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/view-all/val.txt'
ann_file_test = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/view-all/test.txt'

# data_root = '/home/zhouzhuo/scratch/MOD/data'
# data_root_val = '/home/zhouzhuo/scratch/MOD/data'

# ann_file_train = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/level_list_new5/level1/train.txt'
# ann_file_val = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/level_list_new5/level1/val.txt'
# ann_file_test = '/home/public/Hidden_Intention/Zhouz/dataset/Drone/level_list_new5/level1/test.txt'

# ann_file_train = '/home/zhouzhuo/scratch/dataset/Drone/level_three2/view3/train.txt'
# ann_file_val = '/home/zhouzhuo/scratch/dataset/Drone/level_three2/view3/val.txt'
# ann_file_test = '/home/zhouzhuo/scratch/dataset/Drone/level_three2/view3/test.txt'

# ann_file_train = '/home/zhouzhuo/scratch/MOD/list/train.txt'
# ann_file_val = '/home/zhouzhuo/scratch/MOD/list/val.txt'
# ann_file_test = '/home/zhouzhuo/scratch/MOD/list/test.txt'

# ann_file_train = '/home/zhouzhuo/scratch/dataset/Drone/all/view-all/train.csv'
# ann_file_val = '/home/zhouzhuo/scratch/dataset/Drone/all/view-all/val.csv'
# ann_file_test = '/home/zhouzhuo/scratch/dataset/Drone/all/view-all/test.csv'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),

    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='PersonCrop'),
    dict(type='CenterCrop', crop_size=(224, 224)),
    # dict(type='RandomRotation', degrees=30),
    # dict(type='RandomResizedCrop'),
    # dict(type='FunctionWrapper',
    #      function=detect_and_adjust_cropping,
    #      keys=['imgs'],
    #      bbox_keys=['gt_bboxes']),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=300, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=1e-5),
    clip_grad=dict(max_norm=40, norm_type=2))
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4),
#     clip_grad=dict(max_norm=40, norm_type=2))
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=1e-5),
#     clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=300,
        eta_min=0,
        by_epoch=True,
    )
]
# load_from = "./checkpoints/x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth"
# load_from = "/home/zhouzhuo/project/mmaction2/work_dirs/view-all/best_acc_top1_epoch_59.pth"
load_from = "/home/public/Hidden_Intention/Zhouz/UAV/work_dirs/mod-x3d/best_acc_top1_epoch_34.pth"
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
