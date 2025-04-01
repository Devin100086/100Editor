#!/bin/bash

GS_SOURCE="/home/wucunqi/Desktop/results/face/3DGS+depth/point_cloud/iteration_7000/point_cloud.ply"
COLMAP_DIR="/home/wucunqi/Desktop/datasets/face/"
SAVE_DIR="save/render_origin"

rm -rf ${SAVE_DIR}/*

if [ ! -z "$1" ]; then
    GS_SOURCE="$1"
fi

if [ ! -z "$2" ]; then
    COLMAP_DIR="$2"
fi

if [ ! -z "$3" ]; then
    COLMAP_DIR="$3"
fi

python EditorGS/GUIEditor/render.py --gs_source ${GS_SOURCE} --colmap_dir ${COLMAP_DIR} --save_dir ${SAVE_DIR} 