#!/bin/bash

SCENE_PATH="/home/wucunqi/Desktop/datasets/face"
OUTPUT_PATH="/home/wucunqi/Desktop/3DEditor/save/"

if [ ! -z "$1" ]; then
    SCENE_PATH="$1"
fi

if [ ! -z "$2" ]; then
    OUTPUT_PATH="$2"
fi

python trainer/origin/train.py -s ${SCENE_PATH} -m ${OUTPUT_PATH}