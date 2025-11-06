"""Implementation of Motor A (static resonance geometry)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import OscillationConfig, ShellConfig
from .grid import GridState


@dataclass
class MotorA:
    shell_cfg: ShellConfig
    osc_cfg: OscillationConfig
    sum_tolerance: int = 2
    amplitude_threshold: float = 0.05

    def iterate(self, state: GridState, max_iters: int = 1) -> bool:
        """Run up to ``max_iters`` iterations of the static solver."""

        changed = False
        for _ in range(max_iters):
            iter_changed = self._iterate_once(state)
            changed = changed or iter_changed
            if not iter_changed:
                break
        return changed

    def _iterate_once(self, state: GridState) -> bool:
        shell_indices = sorted(state.shell_lut.shells.keys())
        changed = False
        state.front_active.fill(False)

        for r in shell_indices:
            coords = state.shell_lut.shells[r]
            target_value = self._shell_value(r)
            for x, y, z in coords:
                idx = (int(x), int(y), int(z))
                if state.fixed_mask[idx]:
                    continue
                if state.amplitude[idx] < self.amplitude_threshold:
                    continue

                proposed = self._update_cell(state, idx, target_value)
                if proposed is None:
                    continue
                old_val = int(state.values[idx])
                if proposed != old_val and proposed >= old_val:
                    state.values[idx] = proposed
                    state.front_active[idx] = True
                    changed = True
        return changed

    def _update_cell(
        self, state: GridState, idx: Tuple[int, int, int], shell_value: int
    ) -> int | None:
        x, y, z = idx
        weighted_sum = 0.0
        total_weight = 0.0
        for nx, ny, nz in state.iter_neighbours(x, y, z):
            neighbour_val = int(state.values[nx, ny, nz])
            if neighbour_val < 0:
                continue
            if neighbour_val == 0:
                continue
            neighbour_amp = float(state.amplitude[nx, ny, nz])
            weight = 1.0 + self.osc_cfg.amplitude_weight * neighbour_amp
            weighted_sum += neighbour_val * weight
            total_weight += weight

        if total_weight == 0.0:
            return None

        avg = weighted_sum / total_weight
        if abs(avg - self.shell_cfg.T) > self.sum_tolerance:
            return None

        candidate = max(shell_value, int(state.values[idx]))
        return min(9, candidate)

    def _shell_value(self, r: int) -> int:
        if r < len(self.shell_cfg.S):
            value = self.shell_cfg.S[r]
        else:
            if len(self.shell_cfg.S) == 1:
                step = 1
            else:
                step = self.shell_cfg.S[-1] - self.shell_cfg.S[-2]
                if step <= 0:
                    step = 1
            extra = r - (len(self.shell_cfg.S) - 1)
            value = self.shell_cfg.S[-1] + step * extra
        value = max(2, value)
        return min(9, int(value))


__all__ = ["MotorA"]
