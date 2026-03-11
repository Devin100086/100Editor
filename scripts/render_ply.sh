#!/bin/bash

GS_SOURCE="outputs/Turn_it_into_basketball@2026_01_30_09_51/result.ply"
COLMAP_DIR="/home/wucunqi/Desktop/orange"
SAVE_DIR="outputs/Turn_it_into_basketball@2026_01_30_09_51"
# SAVE_DIR="save/render_edited"
USE_ORIGIN=1

# rm -rf ${SAVE_DIR}/*  

if [ ! -z "$1" ]; then
    GS_SOURCE="$1"
fi

if [ ! -z "$2" ]; then
    COLMAP_DIR="$2"
fi

if [ ! -z "$3" ]; then
    SAVE_DIR="$3"
fi

python EditorGS/render.py \
        --gs_source ${GS_SOURCE} \
        --colmap_dir ${COLMAP_DIR} \
        --save_dir ${SAVE_DIR} \
        --use_original_resolution ${USE_ORIGIN}