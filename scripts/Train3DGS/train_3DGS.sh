#!/bin/bash

DATASET_Path="/home/wucunqi/Desktop/datasets/face"

if [ ! -z "$1" ]; then
    DATASET_Path="$1"
fi

python trainer/origin/train.py -s ${DATASET_Path}