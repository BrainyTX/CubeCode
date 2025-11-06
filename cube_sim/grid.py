"""Grid state representation and helper utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .backend import HAS_CUDA, xp
from .config import BarrierConfig, GridConfig, MirrorConfig, ShellConfig


def _neighborhood_offsets(neighborhood: str) -> np.ndarray:
    """Return neighbour offsets for the selected connectivity."""

    offsets: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                weight = abs(dx) + abs(dy) + abs(dz)
                if neighborhood == "faces" and weight == 1:
                    offsets.append((dx, dy, dz))
                elif neighborhood == "edges" and weight in {1, 2}:
                    offsets.append((dx, dy, dz))
                elif neighborhood == "corners":
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int8)


@dataclass
class ShellLUT:
    """Lookup table mapping each cell to its shell index."""

    radii: np.ndarray
    shells: Dict[int, np.ndarray]

    @classmethod
    def build(cls, grid: GridConfig, shells: ShellConfig) -> "ShellLUT":
        centre = (grid.N - 1) // 2
        coords = np.indices((grid.N, grid.N, grid.N))
        dx = coords[0] - centre
        dy = coords[1] - centre
        dz = coords[2] - centre

        if shells.metric == "linf":
            radii = np.maximum.reduce([np.abs(dx), np.abs(dy), np.abs(dz)])
        elif shells.metric == "l1":
            radii = np.abs(dx) + np.abs(dy) + np.abs(dz)
        else:  # l2
            radii = np.sqrt(dx * dx + dy * dy + dz * dz)
            radii = np.rint(radii).astype(np.int32)

        shell_map: Dict[int, List[Tuple[int, int, int]]] = {}
        it = np.ndindex((grid.N, grid.N, grid.N))
        for x, y, z in it:
            r = int(radii[x, y, z])
            shell_map.setdefault(r, []).append((x, y, z))

        shell_indices = {r: np.array(v, dtype=np.int16) for r, v in shell_map.items()}
        return cls(radii=radii.astype(np.int32), shells=shell_indices)


@dataclass
class GridState:
    """Mutable simulation state residing on the lattice."""

    values: np.ndarray  # int8
    phase: np.ndarray  # uint8
    amplitude: np.ndarray  # float32
    fixed_mask: np.ndarray  # bool
    mirror_id: np.ndarray  # int32
    mirror_groups: Dict[int, np.ndarray]
    front_active: np.ndarray  # bool

    offsets: np.ndarray  # neighbour offsets
    shell_lut: ShellLUT
    centre: Tuple[int, int, int]

    @classmethod
    def initialise(
        cls,
        grid_cfg: GridConfig,
        shell_cfg: ShellConfig,
        mirror_cfg: MirrorConfig,
        barrier_cfg: BarrierConfig,
    ) -> "GridState":
        """Initialise grid arrays with seeds, barriers and mirror identifiers."""

        N = grid_cfg.N
        values = xp.full((N, N, N), -1, dtype=np.int8)
        phase = xp.zeros((N, N, N), dtype=np.uint8)
        amplitude = xp.zeros((N, N, N), dtype=np.float32)
        fixed_mask = xp.zeros((N, N, N), dtype=bool)
        mirror_id = xp.full((N, N, N), -1, dtype=np.int32)
        front_active = xp.zeros((N, N, N), dtype=bool)

        centre = ((N - 1) // 2,) * 3
        values[centre] = 1
        fixed_mask[centre] = True

        if barrier_cfg.border_is_zero:
            border_slice = np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :], np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]
            for sl in border_slice:
                values[sl] = 0
                fixed_mask[sl] = True

        for seed in barrier_cfg.internal_zero_seeds:
            if len(seed) != 3:
                raise ValueError("Internal zero seeds must be length-3 coordinates")
            x, y, z = map(int, seed)
            values[x, y, z] = 0
            fixed_mask[x, y, z] = True

        offsets = _neighborhood_offsets(grid_cfg.neighborhood)
        shell_lut = ShellLUT.build(grid_cfg, shell_cfg)

        # Mirror identifiers: assign each cell to canonical representative.
        mirror_map = _build_mirror_map(grid_cfg.N, mirror_cfg)
        mirror_groups = {}
        for idx, cells in mirror_map.items():
            arr = np.array(cells, dtype=np.int32)
            mirror_groups[idx] = arr
            for cell in cells:
                mirror_id[tuple(cell)] = idx

        return cls(
            values=values,
            phase=phase,
            amplitude=amplitude,
            fixed_mask=fixed_mask,
            mirror_id=mirror_id,
            mirror_groups=mirror_groups,
            front_active=front_active,
            offsets=offsets,
            shell_lut=shell_lut,
            centre=centre,
        )

    def coords_index(self, coords: np.ndarray) -> Tuple[Any, Any, Any]:
        """Return an index tuple suitable for advanced indexing on ``xp`` arrays."""

        if HAS_CUDA:
            coords_dev = xp.asarray(coords, dtype=np.int32)
            return tuple(coords_dev[:, i] for i in range(coords_dev.shape[1]))
        return tuple(coords.T)

    def iter_neighbours(self, x: int, y: int, z: int) -> Iterable[Tuple[int, int, int]]:
        for dx, dy, dz in self.offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.values.shape[0] and 0 <= ny < self.values.shape[1] and 0 <= nz < self.values.shape[2]:
                yield nx, ny, nz


def _build_mirror_map(N: int, mirror_cfg: MirrorConfig) -> Dict[int, List[Tuple[int, int, int]]]:
    """Group coordinates into mirror equivalence classes."""

    centre = (N - 1) // 2
    coord_to_id: Dict[Tuple[int, int, int], int] = {}
    groups: Dict[int, List[Tuple[int, int, int]]] = {}
    next_id = 0

    def canonical(pt: Tuple[int, int, int]) -> Tuple[int, int, int]:
        x, y, z = pt
        variants = [pt]
        if "XY" in mirror_cfg.planes:
            variants.append((x, y, 2 * centre - z))
        if "YZ" in mirror_cfg.planes:
            variants.append((2 * centre - x, y, z))
        if "ZX" in mirror_cfg.planes:
            variants.append((x, 2 * centre - y, z))
        if "XYZ" in mirror_cfg.planes:
            variants.append((2 * centre - x, 2 * centre - y, 2 * centre - z))
        # Canonical representative is min lexicographically
        return min(variants)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                key = canonical((x, y, z))
                if key not in coord_to_id:
                    coord_to_id[key] = next_id
                    groups[next_id] = []
                    next_id += 1
                gid = coord_to_id[key]
                groups[gid].append((x, y, z))
    return groups


__all__ = ["GridState", "ShellLUT"]
