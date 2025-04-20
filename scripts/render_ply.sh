#!/bin/bash

GS_SOURCE="outputs/edit-n2n/Turn_him_into_Dwayne_The_Rock_Johnson@20250415-103449/save/last.ply"
COLMAP_DIR="/home/wucunqi/Desktop/results//"
SAVE_DIR="save/render_edited"
USE_ORIGIN=0

rm -rf ${SAVE_DIR}/*

if [ ! -z "$1" ]; then
    GS_SOURCE="$1"
fi

if [ ! -z "$2" ]; then
    COLMAP_DIR="$2"
fi

if [ ! -z "$3" ]; then
    SAVE_DIR="$3"
fi

python EditorGS/GUIEditor/render.py --gs_source ${GS_SOURCE} --colmap_dir ${COLMAP_DIR} --save_dir ${SAVE_DIR} --use_original_resolution ${USE_ORIGIN}