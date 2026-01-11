#!/bin/bash

#SBATCH --partition=gpu      #申请分区 `hpxg` 的计算资源
#SBATCH --nodes=1            #申请 1 个节点
#SBATCH --ntasks-per-node=4  #申请每个节点上分配一个任务(进程)
#SBATCH --account=wangzheng  #请将 supervisor 替换为导师的账号名
#SBATCH --gres=gpu:1

#SBATCH --job-name=view3-drop0.5-lr0.0005-w0.00001-nopre-test
#SBATCH --output=./view3-drop0.5-lr0.0005-w0.00001-nopre-test.log
##%J-heatmap-all.log
##%J-ntu60-stgcn-ce_perceptual_sparse-spotlight.log

echo Start slurm job at `date`
# running
# module load anaconda/3
# module load cuda/10.2
# module load cudnn/8.0.4-cuda10.2
# module load gcc/5.4.0
source activate openmmlab

CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
     work_dirs/view-all/best_acc_top1_epoch_59.pth --dump result/result-all.pkl

#python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
#    work_dirs/view2/best_acc_top1_epoch_14.pth --dump result/result-2.pkl

#python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
#    work_dirs/view1/best_acc_top1_epoch_38.pth --dump result/result-1.pkl

#python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
#   work_dirs/view3/best_acc_top1_epoch_24.pth --dump result/result-3.pkl

#python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
#    work_dirs/view4/best_acc_top1_epoch_71.pth --dump result/result-4.pkl

#python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py \
#    work_dirs/view5/best_acc_top1_epoch_59.pth --dump result/result-5.pkl


echo End slurm job at `date`