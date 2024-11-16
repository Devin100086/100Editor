from typing import Dict, Optional, Tuple
import torch

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


def rasterize_splats(
        splats,
        cfg,
        world_size,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=cfg.packed,
            absgrad=(
                cfg.strategy.absgrad
                if isinstance(cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=world_size > 1,
            camera_model=cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
