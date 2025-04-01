#!/bin/bash

SCENE_PATH="/home/wucunqi/Desktop/datasets/face/"
OUTPUT_PATH="/home/wucunqi/Desktop/results/face/3DGS+depth/"

if [ ! -z "$1" ]; then
    SCENE_PATH="$1"
fi

if [ ! -z "$2" ]; then
    OUTPUT_PATH="$2"
fi

python trainer/origin/render.py -s ${SCENE_PATH} -m ${OUTPUT_PATH}