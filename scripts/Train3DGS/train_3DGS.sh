#!/bin/bash

# DATASET_PATH="/home/wucunqi/Desktop/datasets/face"
DATASET_PATH="/home/wucunqi/Desktop/3DEditor/datasets/kangaroo"

if [ ! -z "$1" ]; then
    DATASET_PATH="$1"
fi

python trainer/origin/train.py -s ${DATASET_PATH} --checkpoint_iterations 7000 --eval