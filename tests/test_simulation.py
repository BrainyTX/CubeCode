import importlib

import pytest

np = pytest.importorskip("numpy")

cube_sim = importlib.import_module("cube_sim")
Simulation = cube_sim.Simulation
SimulationConfig = cube_sim.SimulationConfig


def test_simulation_runs_short_horizon():
    cfg = SimulationConfig.from_dict(
        {
            "grid": {"N": 5, "neighborhood": "faces"},
            "run": {"max_iters_static": 3, "max_ticks_dynamic": 8, "log_every": 1, "static_interval": 2},
            "oscillation": {"delta": 0.1, "alpha": 0.2, "beta": 0.05, "kappa": 0.25, "pulse_amplitude": 0.5},
        }
    )
    sim = Simulation(cfg)
    logs = sim.run()

    assert len(logs) == cfg.run.max_ticks_dynamic
    first_entry = logs[0]
    assert {"tick", "order", "amp_mean", "amp_max", "delta_v", "delta_p", "shells"}.issubset(first_entry)
    assert isinstance(first_entry["shells"], dict)

    snapshot = sim.snapshot()
    assert snapshot["values"].shape == (cfg.grid.N, cfg.grid.N, cfg.grid.N)
    assert snapshot["phase"].dtype == np.uint8
    assert np.all(snapshot["phase"] < 8)
    assert np.all(snapshot["amplitude"] >= 0.0)
