#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=${GPUS:-2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --work-dir /home/public/Hidden_Intention/Zhouz/UAV/work_dirs/MOD20-drop0.9-lr0.0005-w0.00001-class --seed 0 --deterministic
# Any arguments from the third one are captured by ${@:3}
