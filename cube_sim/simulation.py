"""High level orchestration for the coupled lattice model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .config import SimulationConfig
from .backend import to_numpy, to_scalar, xp
from .grid import GridState
from .metrics import OrderParameterTracker, change_fraction, shell_histograms
from .mirror import enforce_mirror
from .motor_a import MotorA
from .motor_b import MotorB


@dataclass
class Simulation:
    config: SimulationConfig
    state: GridState = field(init=False)
    motor_a: MotorA = field(init=False)
    motor_b: MotorB = field(init=False)
    order_tracker: OrderParameterTracker = field(init=False)
    tick: int = 0
    static_iterations: int = 0
    _prev_values: Any = field(init=False, repr=False)
    _prev_phase: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = GridState.initialise(
            self.config.grid,
            self.config.shells,
            self.config.mirror,
            self.config.barriers,
        )
        self.motor_a = MotorA(self.config.shells, self.config.oscillation)
        self.motor_b = MotorB(self.config.oscillation, self.config.shells)
        self.order_tracker = OrderParameterTracker()
        self._prev_values = self.state.values.copy()
        self._prev_phase = self.state.phase.copy()

    def run_static_bootstrap(self) -> None:
        for _ in range(self.config.run.max_iters_static):
            changed = self.motor_a.iterate(self.state, max_iters=1)
            enforce_mirror(self.state, self.config.mirror)
            self.static_iterations += 1
            if not changed:
                break
        self._prev_values = self.state.values.copy()
        self._prev_phase = self.state.phase.copy()

    def step(self) -> Dict[str, Any]:
        prev_values = self._prev_values
        prev_phase = self._prev_phase

        self.motor_b.step(self.state)
        enforce_mirror(self.state, self.config.mirror)
        self.tick += 1

        if self.tick % self.config.run.static_interval == 0:
            self.motor_a.iterate(self.state, max_iters=1)
            enforce_mirror(self.state, self.config.mirror)

        order = self.order_tracker.update(self.state)
        amp_mean = float(to_scalar(xp.mean(self.state.amplitude)))
        amp_max = float(to_scalar(xp.max(self.state.amplitude)))
        delta_v = change_fraction(self.state.values, prev_values)
        delta_p = change_fraction(self.state.phase, prev_phase)

        self._prev_values = self.state.values.copy()
        self._prev_phase = self.state.phase.copy()

        metrics: Dict[str, Any] = {
            "tick": float(self.tick),
            "order": order,
            "amp_mean": amp_mean,
            "amp_max": amp_max,
            "delta_v": delta_v,
            "delta_p": delta_p,
        }

        if self.tick % self.config.run.log_every == 0:
            metrics["shells"] = shell_histograms(self.state)

        return metrics

    def run(self) -> List[Dict[str, Any]]:
        self.run_static_bootstrap()
        logs: List[Dict[str, Any]] = []
        for _ in range(self.config.run.max_ticks_dynamic):
            metrics = self.step()
            if self.tick % self.config.run.log_every == 0:
                logs.append(metrics)
        return logs

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "values": to_numpy(self.state.values.copy()),
            "phase": to_numpy(self.state.phase.copy()),
            "amplitude": to_numpy(self.state.amplitude.copy()),
        }


__all__ = ["Simulation"]
