#!/bin/bash

DATASET_PATH="/home/wucunqi/Desktop/datasets/face"

if [ ! -z "$1" ]; then
    DATASET_PATH="$1"
fi

python trainer/origin/train.py -s ${DATASET_PATH}