import torch
import numpy as np
import math


def equal_dicts(dict1, dict2):
    if dict1 is None or dict2 is None:
        return False

    if set(dict1.keys()) != set(dict2.keys()):
        return False

    for key in dict1.keys():
        if not _equal_values(dict1[key], dict2[key]):
            return False
    return True


def _equal_values(value1, value2):
    if isinstance(value1, torch.Tensor):
        return isinstance(value2, torch.Tensor) and torch.equal(value1, value2)

    if isinstance(value1, np.ndarray):
        return isinstance(value2, np.ndarray) and np.array_equal(value1, value2)

    if isinstance(value1, dict):
        if not isinstance(value2, dict):
            return False
        if set(value1.keys()) != set(value2.keys()):
            return False
        for key in value1.keys():
            if not _equal_values(value1[key], value2[key]):
                return False
        return True

    if isinstance(value1, (list, tuple)):
        if not isinstance(value2, (list, tuple)):
            return False
        if len(value1) != len(value2):
            return False
        for item1, item2 in zip(value1, value2):
            if not _equal_values(item1, item2):
                return False
        return True

    if isinstance(value1, float) and isinstance(value2, float):
        if math.isnan(value1) and math.isnan(value2):
            return True

    return value1 == value2

from typing import Any


class EasyDict(dict):

    @property
    def __name__(self):
        return self.__class__.__name__

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
