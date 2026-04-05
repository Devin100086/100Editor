import sys
from pathlib import Path


def _ensure_src_import_paths() -> None:
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent

    for path in (repo_root, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_src_import_paths()

