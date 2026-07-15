"""Deterministic checks for leakage-safe trajectory-level splits."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.dataset_splits import (
    SPLIT_NAMES,
    assert_group_disjoint,
    export_grouped_splits,
    grouped_split_indices,
    load_npz_dataset,
    split_dataset,
)


def synthetic_dataset():
    # Unequal lengths catch implementations that accidentally balance rows by
    # cutting a trajectory in two.
    trajectory_id = np.repeat(np.arange(11), np.arange(2, 13))
    n = len(trajectory_id)
    return {
        "trajectory_id": trajectory_id,
        "episode_id": trajectory_id.copy(),
        "time_true": np.arange(n, dtype=float) * 0.001,
        "q_true": np.arange(n * 2, dtype=float).reshape(n, 2),
        "flow_velocity": np.tile([0.3, 0.0, 0.0], (n, 1)),
    }


def verify_grouping_and_determinism():
    dataset = synthetic_dataset()
    split_a, indices_a = split_dataset(dataset, seed=17)
    split_b, indices_b = split_dataset(dataset, seed=17)
    for name in SPLIT_NAMES:
        assert np.array_equal(indices_a[name], indices_b[name])
        assert np.all(np.diff(indices_a[name]) > 0)  # source order retained
        assert len(split_a[name]["q_true"]) == len(indices_a[name])
        assert np.array_equal(split_a[name]["q_true"], split_b[name]["q_true"])
    assert_group_disjoint(dataset["trajectory_id"], indices_a)
    ids = [set(split_a[name]["trajectory_id"].tolist()) for name in SPLIT_NAMES]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def verify_leakage_detection():
    groups = np.repeat(np.arange(3), 2)
    bad = {
        "train": np.array([0, 1, 2]),  # trajectory 1 is cut
        "validation": np.array([3]),
        "test": np.array([4, 5]),
    }
    try:
        assert_group_disjoint(groups, bad)
    except ValueError as error:
        assert "trajectory IDs leak" in str(error)
    else:
        raise AssertionError("trajectory leakage was not rejected")


def verify_export_and_manifest():
    dataset = synthetic_dataset()
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "raw.npz")
        out = os.path.join(directory, "splits")
        np.savez_compressed(source, **dataset)
        manifest = export_grouped_splits(source, out, seed=23)
        with open(os.path.join(out, "split_manifest.json")) as stream:
            saved_manifest = json.load(stream)
        assert manifest == saved_manifest
        assert saved_manifest["total_samples"] == len(dataset["trajectory_id"])
        all_ids = []
        for name in SPLIT_NAMES:
            split = load_npz_dataset(os.path.join(out, f"{name}.npz"))
            details = saved_manifest["splits"][name]
            assert len(split["trajectory_id"]) == details["samples"]
            assert len(np.unique(split["trajectory_id"])) == details["trajectories"]
            all_ids.extend(details["trajectory_ids"])
        assert sorted(all_ids) == list(range(11))


def verify_invalid_inputs():
    dataset = synthetic_dataset()
    for ratios in ((0.7, 0.3, 0.2), (1.0, 0.0, 0.0)):
        try:
            split_dataset(dataset, ratios=ratios)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ratios accepted: {ratios}")
    try:
        grouped_split_indices(np.array([0, 0, 1, 1]))
    except ValueError as error:
        assert "three trajectories" in str(error)
    else:
        raise AssertionError("too few trajectories were accepted")


def main():
    verify_grouping_and_determinism()
    verify_leakage_detection()
    verify_export_and_manifest()
    verify_invalid_inputs()
    print("PASS: deterministic trajectory splits have no row or group leakage")


if __name__ == "__main__":
    main()
