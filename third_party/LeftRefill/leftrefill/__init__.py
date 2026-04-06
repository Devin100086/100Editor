from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Optional


_PREDICT_FN: Optional[Callable] = None


def _resolve_predict() -> Callable:
    # Lazy import so package import itself does not immediately load model weights.
    global _PREDICT_FN
    if _PREDICT_FN is not None:
        return _PREDICT_FN

    try:
        from run import predict as _predict
    except ModuleNotFoundError:
        try:
            from LeftRefill.run import predict as _predict
        except ModuleNotFoundError:
            run_path = Path(__file__).resolve().parent.parent / "run.py"
            spec = spec_from_file_location("leftrefill_run", run_path)
            if spec is None or spec.loader is None:
                raise ModuleNotFoundError(f"Cannot load LeftRefill run.py at: {run_path}")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            _predict = module.predict

    _PREDICT_FN = _predict
    return _PREDICT_FN


def predict(*args, **kwargs):
    return _resolve_predict()(*args, **kwargs)
