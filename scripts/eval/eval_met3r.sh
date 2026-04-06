#!/bin/bash
METRICS_DIR="src/eval/metrics"

cd "${METRICS_DIR}"
python eval_met3r.py \
    --image-dir /media/wucunqi/data/100Editor/exp/Comparison/EditSplat/face_to_Kevin_Durant/point_cloud/iteration_7560/rendered_origin \
    --distance cosine \
    --img-size 256