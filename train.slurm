#!/bin/bash

#SBATCH --partition=gpu      #申请分区 `hpxg` 的计算资源
#SBATCH --nodes=1            #申请 1 个节点
#SBATCH --ntasks-per-node=4  #申请每个节点上分配一个任务(进程)
#SBATCH --account=wangzheng  #请将 supervisor 替换为导师的账号名
#SBATCH --gres=gpu:1

#SBATCH --job-name=two-1-drop0.5-lr0.0005-w0.00001-nopre-class-person
#SBATCH --output=./two-1-drop0.5-lr0.0005-w0.00001-nopre-class-person.log
##%J-heatmap-all.log
##%J-ntu60-stgcn-ce_perceptual_sparse-spotlight.log

echo Start slurm job at `date`
# running
# module load anaconda/3
# module load cuda/10.2
# module load cudnn/8.0.4-cuda10.2
# module load gcc/5.4.0
source activate openmmlab

CUDA_VISIBLE_DEVICES=4 python tools/train.py  configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
    --work-dir work_dirs/UAV-four-view1-drop0.9-lr0.0005-w0.00001-class \
    --seed 0 --deterministic \
    > test.txt 2>&1

echo End slurm job at `date`