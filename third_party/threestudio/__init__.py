import os
import sys
from pathlib import Path

__modules__ = {}


def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


def _maybe_add_hundrededitor_src_to_path() -> None:
    candidates = []

    env_root = os.getenv("HUNDREDEDITOR_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "src")

    # Typical editable install in this repository: <repo>/third_party/threestudio
    candidates.append(Path(__file__).resolve().parents[2] / "src")
    # Running commands from project root without editable install.
    candidates.append(Path.cwd() / "src")

    for src_dir in candidates:
        src_dir = src_dir.resolve()
        if (src_dir / "editor").is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            break


_maybe_add_hundrededitor_src_to_path()


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, systems
