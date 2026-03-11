#!/bin/bash

CLIP_PROMPT_ORIGIN="a photo of an outdoor garden"
CLIP_PROMPT_TARGET="a photo of an outdoor garden in winter"
ORIGIN_IMAGE_DIR="/media/wucunqi/data/results/garden_8/rendered_origin"
EDITED_IMAGE_DIR="outputs/Make_it_winter@2026_01_27_22_26/rendered_origin"

python EditorGS/eval_image.py \
       --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}" \
       --clip_prompt_target "${CLIP_PROMPT_TARGET}" \
         --origin_image_dir ${ORIGIN_IMAGE_DIR} \
         --edited_image_dir  ${EDITED_IMAGE_DIR} \