#!/bin/bash
METRICS_DIR="src/eval/metrics"
ORIGIN_GS_SOURCE="/media/wucunqi/data/results/person-small/3DGS/point_cloud/iteration_7000/point_cloud.ply"
EDITED_GS_SOURCE="/media/wucunqi/data/100Editor/exp/Ablation/Turn_him_into_a_iron_man@2025_10_09_20_31/result.ply"
COLMAP_DIR="/media/wucunqi/data/results/person-small"
CLIP_PROMPT_ORIGIN="a photo of a man"
CLIP_PROMPT_TARGET="a photo of a iron man"
USE_ORIGIN=1

cd "${METRICS_DIR}"
python eval_ply.py \
        --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}" \
        --clip_prompt_target "${CLIP_PROMPT_TARGET}" \
        --origin_gs_source ${ORIGIN_GS_SOURCE} \
        --edited_gs_source  ${EDITED_GS_SOURCE} \
        --colmap_dir ${COLMAP_DIR} \
        --use_original_resolution ${USE_ORIGIN}