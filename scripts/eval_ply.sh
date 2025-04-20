#!/bin/bash

CLIP_PROMPT_ORIGIN="a photo of a man"
CLIP_PROMPT_TARGET="a photo of a hulk"
ORIGIN_GS_SOURCE="/home/wucunqi/Desktop/results/face/3DGS/point_cloud/iteration_30000/point_cloud.ply"
EDITED_GS_SOURCE="/home/wucunqi/Desktop/results/face/edit/Hulk/save/last.ply"
COLMAP_DIR="/home/wucunqi/Desktop/results/face"
USE_ORIGIN=0

python EditorGS/eval_ply.py \
        --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}" \
        --clip_prompt_target "${CLIP_PROMPT_TARGET}" \
        --origin_gs_source ${ORIGIN_GS_SOURCE} \
        --edited_gs_source  ${EDITED_GS_SOURCE} \
        --colmap_dir ${COLMAP_DIR} \
        --use_original_resolution ${USE_ORIGIN}