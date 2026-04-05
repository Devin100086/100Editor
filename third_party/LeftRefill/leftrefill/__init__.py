def predict(*args, **kwargs):
    # Lazy import so package import itself does not immediately load model weights.
    try:
        from run import predict as _predict
    except ModuleNotFoundError:
        from LeftRefill.run import predict as _predict

    return _predict(*args, **kwargs)
