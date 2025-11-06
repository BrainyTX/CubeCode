"""Configuration dataclasses for the coupled resonance/oscillation model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class GridConfig:
    """Configuration for the lattice discretisation."""

    N: int = 65
    neighborhood: str = "faces"  # faces|edges|corners

    def __post_init__(self) -> None:
        if self.N % 2 == 0:
            raise ValueError("N must be odd so that the grid has a unique centre")
        if self.N < 3:
            raise ValueError("N must be at least 3")
        if self.neighborhood not in {"faces", "edges", "corners"}:
            raise ValueError("neighborhood must be one of faces|edges|corners")


@dataclass
class ShellConfig:
    """Configuration for static shell expansion in Motor A."""

    metric: str = "linf"  # linf|l1|l2
    S: Sequence[int] = field(default_factory=lambda: [1, 2, 6, 9, 18, 27, 39])
    T: int = 9

    def __post_init__(self) -> None:
        if self.metric not in {"linf", "l1", "l2"}:
            raise ValueError("metric must be one of linf|l1|l2")
        if len(self.S) == 0:
            raise ValueError("S must contain at least one shell seed value")
        if any(s <= 0 for s in self.S):
            raise ValueError("S entries must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")


@dataclass
class MirrorConfig:
    """Configuration for mirroring rules."""

    planes: Sequence[str] = field(default_factory=lambda: ["XY", "YZ", "ZX"])
    map: str = "complement"  # identical|complement
    phase_map: str = "identical"  # identical|antisym

    def __post_init__(self) -> None:
        allowed_planes = {"XY", "YZ", "ZX", "XYZ"}
        for plane in self.planes:
            if plane not in allowed_planes:
                raise ValueError(f"Unsupported mirror plane '{plane}'")
        if self.map not in {"identical", "complement"}:
            raise ValueError("map must be identical|complement")
        if self.phase_map not in {"identical", "antisym"}:
            raise ValueError("phase_map must be identical|antisym")


@dataclass
class OscillationConfig:
    """Parameters for the eight-phase oscillation dynamics (Motor B)."""

    phases: int = 8
    kappa: float = 0.35
    delta: float = 0.02
    alpha: float = 0.15
    beta: float = 0.05
    T_drive: float = 1.0
    dphi_shell: str = "from_S"
    amplitude_weight: float = 1.0
    pulse_amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.phases != 8:
            raise ValueError("Only 8-phase oscillations are currently supported")
        if not (0.0 <= self.kappa <= 2.0):
            raise ValueError("kappa should be between 0 and 2")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError("delta must be in [0, 1)")
        if self.dphi_shell not in {"from_S", "constant"}:
            raise ValueError("dphi_shell must be 'from_S' or 'constant'")


@dataclass
class BarrierConfig:
    """Barrier/seed configuration for Motor A."""

    border_is_zero: bool = True
    internal_zero_seeds: Sequence[Sequence[int]] = field(default_factory=list)


@dataclass
class RunConfig:
    """Execution control parameters."""

    max_iters_static: int = 200
    max_ticks_dynamic: int = 5_000
    log_every: int = 25
    save_slices: Sequence[str] = field(default_factory=lambda: ["z=0", "y=0", "x=0"])
    static_interval: int = 10


@dataclass
class SimulationConfig:
    """Aggregate configuration for the coupled model."""

    grid: GridConfig = field(default_factory=GridConfig)
    shells: ShellConfig = field(default_factory=ShellConfig)
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
    oscillation: OscillationConfig = field(default_factory=OscillationConfig)
    barriers: BarrierConfig = field(default_factory=BarrierConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Construct a configuration from a nested dictionary (e.g. YAML)."""

        return cls(
            grid=GridConfig(**data.get("grid", {})),
            shells=ShellConfig(**data.get("shells", {})),
            mirror=MirrorConfig(**data.get("mirror", {})),
            oscillation=OscillationConfig(**data.get("oscillation", {})),
            barriers=BarrierConfig(**data.get("barriers", {})),
            run=RunConfig(**data.get("run", {})),
        )


__all__ = [
    "BarrierConfig",
    "GridConfig",
    "MirrorConfig",
    "OscillationConfig",
    "RunConfig",
    "ShellConfig",
    "SimulationConfig",
]
