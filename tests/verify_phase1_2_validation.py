"""Deterministic smoke checks for the Phase 1.2 experiment runner."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.phase1_2_validate_uniform_flow import TrajectoryCase, run_case


def main():
    case = TrajectoryCase("smoke", 0.05, (1.0, 0.6), 0.30, 0.20, 6.0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        still_a_path = root / "still_a.npz"
        still_b_path = root / "still_b.npz"
        current_path = root / "current.npz"

        still_a = run_case(case, "still", (0.0, 0.0, 0.0), still_a_path)
        still_b = run_case(case, "still", (0.0, 0.0, 0.0), still_b_path)
        current = run_case(case, "x_pos_030", (0.3, 0.0, 0.0), current_path)

        assert still_a["finite"] and still_b["finite"] and current["finite"]
        assert still_a["samples"] == 50
        with np.load(still_a_path) as first, np.load(still_b_path) as second:
            assert set(first.files) == set(second.files)
            for key in first.files:
                assert np.array_equal(first[key], second[key]), key
        with np.load(current_path) as trace:
            assert trace["q"].shape == (50, 2)
            assert trace["link_relative_velocity"].shape == (50, 2, 2)
            assert np.allclose(trace["flow_velocity"], [0.3, 0.0, 0.0])

        response = np.mean(np.linalg.norm(
            current["_fluid_torque"] - still_a["_fluid_torque"], axis=1
        ))
        assert response > 0.01

    print("PASS: Phase 1.2 runner is deterministic and flow-conditioned")


if __name__ == "__main__":
    main()
