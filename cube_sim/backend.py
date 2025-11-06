"""Backend selection utilities for CPU/GPU execution."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as _cupy  # type: ignore
except Exception:  # pragma: no cover - cupy missing or broken
    _cupy = None  # type: ignore


def _detect_cuda() -> bool:
    """Return ``True`` if a CUDA device is available and usable."""

    if _cupy is None:
        return False
    if os.environ.get("CUBE_SIM_DISABLE_CUDA"):
        return False
    try:  # pragma: no cover - requires CUDA runtime
        device_count = _cupy.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - runtime failure
        return False
    return device_count > 0


HAS_CUDA = _detect_cuda()
xp = _cupy if HAS_CUDA else np


def to_numpy(array: Any) -> np.ndarray:
    """Convert an ``xp`` array to a NumPy array on the host."""

    if HAS_CUDA and _cupy is not None:
        return _cupy.asnumpy(array)
    return np.asarray(array)


def to_scalar(value: Any) -> Any:
    """Return a Python scalar from an ``xp`` scalar."""

    if HAS_CUDA and _cupy is not None and hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = ["HAS_CUDA", "to_numpy", "to_scalar", "xp"]

