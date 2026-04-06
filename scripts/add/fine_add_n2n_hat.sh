python launch.py \
    --config configs/edit-n2n.yaml \
    --train --gpu 0 \
    trainer.max_steps=2000 \
    system.prompt_processor.prompt="A man wear a red hat" \
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
    system.seg_prompt="hat" \
    system.loss.lambda_anchor_color=0 \
    system.loss.lambda_anchor_geo=50 \
    system.loss.lambda_anchor_scale=50 \
    system.loss.lambda_anchor_opacity=50 \
    system.densify_from_iter=100 \
    system.densify_until_iter=1501 \
    system.densification_interval=100 \
    data.source="/home/wucunqi/Desktop/results/face" \
    system.gs_source="/home/wucunqi/Desktop/results/face/3DGS+depth/point_cloud/iteration_7000/point_cloud.ply"
