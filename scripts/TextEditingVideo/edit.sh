SEG="face"
PROMPT="A man add a sunglasses"
DATASOURCE="/home/wucunqi/Desktop/results/face"
GSSOURCE="/home/wucunqi/Desktop/results/face/3DGS+depth/point_cloud/iteration_7000/point_cloud.ply"

if [ ! -z "$1" ]; then
    SEG="$1"
fi

if [ ! -z "$2" ]; then
    PROMPT="$2"
fi

if [ ! -z "$3" ]; then
    DATASOURCE="$3"
fi

if [ ! -z "$4" ]; then
    GSSOURCE="$4"
fi

python launch.py \
    --config configs/edit-n2n.yaml \
    --train --gpu 0 \
    data.max_view_num=20 \
    system.prompt_processor.prompt="${PROMPT}" \
    system.per_editing_step=100000 \
    system.max_densify_percent=0.01 \
    system.anchor_weight_init_g0=0.05 \
    system.anchor_weight_init=0.1 \
    system.anchor_weight_multiplier=1.3 \
    system.gs_lr_scaler=1 \
    system.seg_prompt="${SEG}" \
    system.gs_final_lr_scaler=1 \
    system.color_lr_scaler=1 \
    system.opacity_lr_scaler=1 \
    system.scaling_lr_scaler=1 \
    system.rotation_lr_scaler=1 \
    system.loss.lambda_anchor_color=0 \
    system.loss.lambda_anchor_geo=0 \
    system.loss.lambda_anchor_scale=0 \
    system.loss.lambda_anchor_opacity=0 \
    system.densify_from_iter=0 \
    system.densify_until_iter=200000 \
    system.densification_interval=100 \
    system.camera_update_per_step=500 \
    system.video=true \
    trainer.val_check_interval=500 \
    trainer.max_steps=1000 \
    data.source=${DATASOURCE} \
    system.gs_source=${GSSOURCE}
