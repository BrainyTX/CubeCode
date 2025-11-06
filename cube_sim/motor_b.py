"""Implementation of Motor B (time-discrete eight-phase oscillation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .backend import xp
from .config import OscillationConfig, ShellConfig
from .grid import GridState


@dataclass
class MotorB:
    osc_cfg: OscillationConfig
    shell_cfg: ShellConfig
    unit_vectors: np.ndarray = field(init=False)
    shell_increments: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        angles = xp.arange(8, dtype=np.float32) * (xp.pi / 4.0)
        self.unit_vectors = xp.exp(1j * angles).astype(np.complex64)
        self.shell_increments = {}

    def _shell_increment(self, r: int) -> int:
        if self.osc_cfg.dphi_shell == "constant":
            return 0
        if r in self.shell_increments:
            return self.shell_increments[r]

        s_values = list(self.shell_cfg.S)
        if not s_values:
            increment = 0
        else:
            base = s_values[0]
            if r < len(s_values):
                ref = s_values[r]
            else:
                if len(s_values) < 2:
                    step = 1
                else:
                    step = s_values[-1] - s_values[-2]
                    if step <= 0:
                        step = 1
                ref = s_values[-1] + (r - len(s_values) + 1) * step
            diff = max(0, ref - base)
            step = 1 if len(s_values) < 2 else max(1, s_values[-1] - s_values[-2])
            increment = int(diff // step) % 8

        self.shell_increments[r] = increment
        return increment

    def step(self, state: GridState) -> None:
        centre = state.centre
        state.phase[centre] = 0
        prev_amplitude = state.amplitude.copy()

        phase_vectors = self.unit_vectors[state.phase]
        resultant = xp.zeros_like(phase_vectors, dtype=np.complex64)
        neighbour_counts = xp.zeros(state.phase.shape, dtype=np.int16)

        for dx, dy, dz in state.offsets:
            src_slice, dst_slice = self._shift_slices(state.values.shape, int(dx), int(dy), int(dz))
            resultant[dst_slice] += phase_vectors[src_slice]
            neighbour_counts[dst_slice] += 1

        neighbour_counts = xp.maximum(neighbour_counts, 1)
        coherence = (xp.abs(resultant) / neighbour_counts).astype(np.float32)
        phase_angle = xp.angle(resultant, deg=False)
        phase_angle = xp.nan_to_num(phase_angle, nan=0.0)
        coupling = xp.round(self.osc_cfg.kappa * phase_angle / (xp.pi / 4.0)).astype(np.int32)

        base_increment = xp.zeros_like(state.phase, dtype=np.int32)
        for r, coords in state.shell_lut.shells.items():
            inc = self._shell_increment(r)
            base_increment[state.coords_index(coords)] = inc

        new_phase = (state.phase.astype(np.int32) + base_increment + coupling) % 8
        new_phase[centre] = 0
        state.phase[...] = new_phase.astype(np.uint8)

        state.amplitude[...] = (
            (1.0 - self.osc_cfg.delta) * prev_amplitude
            + self.osc_cfg.alpha * coherence
            + self.osc_cfg.beta * (state.phase == 0).astype(np.float32)
        )
        state.amplitude[centre] = self.osc_cfg.pulse_amplitude
        xp.clip(state.amplitude, 0.0, None, out=state.amplitude)

    @staticmethod
    def _shift_slices(shape: tuple[int, int, int], dx: int, dy: int, dz: int):
        sx0 = max(0, -dx)
        sx1 = shape[0] - max(0, dx)
        sy0 = max(0, -dy)
        sy1 = shape[1] - max(0, dy)
        sz0 = max(0, -dz)
        sz1 = shape[2] - max(0, dz)

        dx0 = max(0, dx)
        dx1 = shape[0] - max(0, -dx)
        dy0 = max(0, dy)
        dy1 = shape[1] - max(0, -dy)
        dz0 = max(0, dz)
        dz1 = shape[2] - max(0, -dz)

        src = np.s_[sx0:sx1, sy0:sy1, sz0:sz1]
        dst = np.s_[dx0:dx1, dy0:dy1, dz0:dz1]
        return src, dst


__all__ = ["MotorB"]
