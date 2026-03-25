#!/bin/bash

GS_SOURCE="/home/wucunqi/Desktop/Deblur/outputs/edit-n2n/remove_degradation@20260323-111827/save/last.ply"
COLMAP_DIR="/media/wucunqi/data/3D_Datasets/20260303_cunqi"
SAVE_DIR="/home/wucunqi/Desktop/Deblur/outputs/edit-n2n/remove_degradation@20260323-111827"
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