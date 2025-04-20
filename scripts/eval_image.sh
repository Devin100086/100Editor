#!/bin/bash

CLIP_PROMPT_ORIGIN="a photo of a man"
CLIP_PROMPT_TARGET="a photo of a hulk"
ORIGIN_IMAGE_DIR="save/render_origin"
EDITED_IMAGE_DIR="save/render_edited"

python EditorGS/eval_image.py \
       --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}" \
       --clip_prompt_target "${CLIP_PROMPT_TARGET}" \
         --origin_image_dir ${ORIGIN_IMAGE_DIR} \
         --edited_image_dir  ${EDITED_IMAGE_DIR} \