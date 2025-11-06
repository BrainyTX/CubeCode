"""Mirror enforcement utilities."""
from __future__ import annotations

import numpy as np

from .config import MirrorConfig
from .grid import GridState


def _complement_value(value: int) -> int:
    return 10 - value if value >= 0 else value


def _antisym_phase(phase: int) -> int:
    return (-phase) % 8


def enforce_mirror(state: GridState, config: MirrorConfig) -> None:
    """Enforce mirror constraints on values and phases."""

    for group_id, coords in state.mirror_groups.items():
        values = state.values[tuple(coords.T)]
        phases = state.phase[tuple(coords.T)]

        if config.map == "identical":
            target_val = _most_common(values)
        else:
            complement = np.array([_complement_value(int(v)) for v in values], dtype=np.int16)
            target_val = _most_common(complement)
            target_val = _complement_value(int(target_val))

        if config.phase_map == "identical":
            target_phase = _most_common(phases)
        else:
            antisym = np.array([_antisym_phase(int(p)) for p in phases], dtype=np.int16)
            target_phase = _most_common(antisym)
            target_phase = _antisym_phase(int(target_phase))

        state.values[tuple(coords.T)] = target_val
        state.phase[tuple(coords.T)] = target_phase


def _most_common(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    unique, counts = np.unique(values, return_counts=True)
    idx = np.argmax(counts)
    return int(unique[idx])


__all__ = ["enforce_mirror"]
