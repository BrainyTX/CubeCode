"""Metrics and diagnostics for the coupled model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Union

import numpy as np

from .backend import to_numpy, to_scalar, xp
from .grid import GridState


@dataclass
class OrderParameterTracker:
    window: int = 256
    history: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        from collections import deque

        self.history = deque(maxlen=self.window)

    def update(self, state: GridState) -> float:
        unit_vectors = xp.exp(1j * (xp.pi / 4.0) * state.phase)
        R = xp.abs(xp.mean(unit_vectors))
        value = float(to_scalar(R))
        self.history.append(value)
        return value

    def spectrum(self, sample_rate: float = 1.0) -> Dict[str, np.ndarray]:
        if len(self.history) == 0:
            return {"freq": np.array([]), "power": np.array([])}
        values = np.array(self.history, dtype=np.float32)
        fft = np.fft.rfft(values - values.mean())
        freq = np.fft.rfftfreq(len(values), d=1.0 / sample_rate)
        power = np.abs(fft) ** 2
        return {"freq": freq, "power": power}


def shell_histograms(state: GridState) -> Dict[int, Dict[str, Union[List[int], float]]]:
    """Compute value/phase histograms and amplitude stats per shell."""

    stats: Dict[int, Dict[str, Union[List[int], float]]] = {}
    for r, coords in state.shell_lut.shells.items():
        idx = state.coords_index(coords)
        values = to_numpy(state.values[idx]).astype(np.int16)
        phases = to_numpy(state.phase[idx]).astype(np.int16)
        amps = to_numpy(state.amplitude[idx]).astype(np.float32)

        value_hist = np.bincount(values + 1, minlength=11)
        phase_hist = np.bincount(phases, minlength=8)

        stats[r] = {
            "value_hist": value_hist.astype(np.int64).tolist(),
            "phase_hist": phase_hist.astype(np.int64).tolist(),
            "amp_mean": float(np.mean(amps)) if amps.size else 0.0,
            "amp_max": float(np.max(amps)) if amps.size else 0.0,
        }

    return stats


def change_fraction(current: np.ndarray, previous: np.ndarray) -> float:
    """Return the fraction of entries that differ between two arrays."""

    if current.size == 0:
        return 0.0
    if previous.shape != current.shape:
        raise ValueError("Arrays must share the same shape for change tracking")
    diff = xp.count_nonzero(current != previous)
    return float(to_scalar(diff) / current.size)


__all__ = ["OrderParameterTracker", "change_fraction", "shell_histograms"]
