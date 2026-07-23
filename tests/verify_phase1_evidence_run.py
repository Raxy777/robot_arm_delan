"""Deterministic checks for the Phase-1 evidence assembly utilities."""
from pathlib import Path
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.phase1_evidence import merge_dataset_parts, write_sha256_manifest


def part(offset):
    return {
        "trajectory_id": np.array([0, 0, 1, 1]),
        "episode_id": np.array([0, 0, 1, 1]),
        "q_true": np.arange(8).reshape(4, 2) + offset,
        "flow_velocity": np.full((4, 3), offset),
    }


def main():
    merged = merge_dataset_parts([part(0), part(10)])
    assert merged["trajectory_id"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert merged["episode_id"].tolist() == merged["trajectory_id"].tolist()
    assert len(merged["q_true"]) == 8
    assert np.all(merged["q_true"][4:] >= 10)
    try:
        merge_dataset_parts([])
        raise AssertionError("empty merge was accepted")
    except ValueError:
        pass
    bad = part(0); bad["extra"] = np.zeros(4)
    try:
        merge_dataset_parts([part(0), bad])
        raise AssertionError("schema mismatch was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); file = root / "x.txt"; file.write_text("phase1")
        manifest = write_sha256_manifest(root, [file])
        text = manifest.read_text()
        assert text.endswith("  x.txt\n") and len(text.split()[0]) == 64
    print("PASS: unique trajectories, schema checks, and checksums")


if __name__ == "__main__":
    main()
