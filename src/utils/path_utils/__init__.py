import os
from pathlib import Path
from typing import Tuple, Union


def find_repo_root(start: Union[str, Path]) -> Path:
    cur = Path(start).resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "src").is_dir() and (parent / "third_party").is_dir():
            return parent
    return cur


def resolve_runtime_dir(start: Union[str, Path], create: bool = False) -> Path:
    runtime_dir = find_repo_root(start) / "runtime"
    if create:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def resolve_runtime_subdir(
    start: Union[str, Path], *parts: str, create: bool = False
) -> Path:
    runtime_dir = resolve_runtime_dir(start, create=create)
    target_dir = runtime_dir.joinpath(*parts)
    if create:
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def resolve_sam2_paths(start: Union[str, Path]) -> Tuple[str, str]:
    def to_sam2_config_name(path_like: Union[str, Path]) -> str:
        path_obj = Path(path_like)
        parts = path_obj.parts
        if "configs" in parts:
            idx = parts.index("configs")
            return "/".join(parts[idx:])
        return str(path_like).replace("\\", "/")

    repo_root = find_repo_root(start)
    runtime_root = repo_root / "runtime"

    model_cfg_env = os.getenv("SAM2_MODEL_CFG")
    if model_cfg_env:
        if model_cfg_env.replace("\\", "/").startswith("configs/"):
            model_cfg = model_cfg_env.replace("\\", "/")
        elif Path(model_cfg_env).is_file():
            model_cfg = to_sam2_config_name(Path(model_cfg_env).resolve())
        else:
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        # Prefer the current location under third_party/sam2, and keep legacy
        # extern/sam2 fallback for older checkouts.
        model_cfg_candidates = [
            repo_root
            / "third_party"
            / "sam2"
            / "sam2"
            / "configs"
            / "sam2.1"
            / "sam2.1_hiera_l.yaml",
            repo_root
            / "third_party"
            / "extern"
            / "sam2"
            / "sam2"
            / "configs"
            / "sam2.1"
            / "sam2.1_hiera_l.yaml",
        ]
        model_cfg_candidates += sorted(
            repo_root.glob(
                "third_party/sam2/build/lib.*/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            )
        )
        model_cfg_candidates += sorted(
            repo_root.glob(
                "third_party/extern/sam2/build/lib.*/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            )
        )
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        for candidate in model_cfg_candidates:
            if candidate.is_file():
                model_cfg = to_sam2_config_name(candidate.resolve())
                break

    checkpoint_env = os.getenv("SAM2_CHECKPOINT")
    checkpoint_candidates = []
    if checkpoint_env:
        checkpoint_candidates.append(Path(checkpoint_env))
    checkpoint_candidates += [
        runtime_root / ".cache" / "sam2" / "sam2.1_hiera_large.pt",
        runtime_root / "cache" / "sam2" / "sam2.1_hiera_large.pt",
        repo_root / "runtime" / "sam2" / "sam2.1_hiera_large.pt",
        repo_root / ".cache" / "sam2" / "sam2.1_hiera_large.pt",
    ]
    checkpoint_candidates += sorted(runtime_root.glob("**/sam2.1_hiera_large.pt"))

    checkpoint = checkpoint_candidates[0]
    for candidate in checkpoint_candidates:
        if candidate.is_file():
            checkpoint = candidate
            break

    return model_cfg, str(checkpoint.resolve())
