python launch.py \
    --config configs/edit-n2n.yaml \
    --train --gpu 0 \
    trainer.max_steps=1000 \
    data.max_view_num=20 \
    system.prompt_processor.prompt="turn him into a spider man with mark" \
    system.per_editing_step=100000 \
    system.max_densify_percent=0.01 \
    system.anchor_weight_init_g0=0.05 \
    system.anchor_weight_init=0.1 \
    system.anchor_weight_multiplier=1.3 \
    system.gs_lr_scaler=5 \
    system.gs_final_lr_scaler=5 \
    system.color_lr_scaler=5 \
    system.opacity_lr_scaler=2 \
    system.scaling_lr_scaler=2 \
    system.rotation_lr_scaler=2 \
    system.loss.lambda_anchor_color=0 \
    system.loss.lambda_anchor_geo=0 \
    system.loss.lambda_anchor_scale=0 \
    system.loss.lambda_anchor_opacity=0 \
    system.densify_from_iter=100 \
    system.densify_until_iter=801 \
    system.densification_interval=100 \
    system.camera_update_per_step=500 \
    system.video=true \
    guidance.video=true \
    data.source="/home/wucunqi/Desktop/results/face" \
    system.gs_source="/home/wucunqi/Desktop/results/face/3DGS+depth/point_cloud/iteration_7000/point_cloud.ply"
