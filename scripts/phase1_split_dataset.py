"""Create leakage-safe trajectory-level Phase-1 dataset splits."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_splits import export_grouped_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="raw sample-aligned NPZ dataset")
    parser.add_argument("--out-dir", default="data/phase1_splits")
    parser.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        default=(0.7, 0.2, 0.1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-key", default="trajectory_id")
    args = parser.parse_args()
    try:
        manifest = export_grouped_splits(
            args.input, args.out_dir, args.ratios, args.seed, args.group_key)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"split {manifest['total_samples']} samples from "
          f"{manifest['total_trajectories']} trajectories")
    for name, details in manifest["splits"].items():
        print(f"{name:10s}: {details['samples']:7d} samples, "
              f"{details['trajectories']:3d} trajectories")
    print(f"manifest: {os.path.abspath(os.path.join(args.out_dir, 'split_manifest.json'))}")


if __name__ == "__main__":
    main()
