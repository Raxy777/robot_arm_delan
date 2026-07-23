"""Utilities for reproducible Phase-1 multi-condition evidence datasets."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np


def merge_dataset_parts(parts):
    """Merge sample-aligned datasets and assign globally unique trajectory IDs.

    Collectors restart episode numbering for each flow. Reindexing here prevents
    unrelated trajectories from being grouped together during leakage-safe
    splitting.
    """
    if not parts:
        raise ValueError("at least one dataset part is required")
    keys = tuple(parts[0])
    merged = {key: [] for key in keys}
    next_id = 0
    for index, part in enumerate(parts):
        if tuple(part) != keys:
            raise ValueError(f"dataset part {index} has a different schema")
        lengths = {len(np.asarray(part[key])) for key in keys}
        if len(lengths) != 1:
            raise ValueError(f"dataset part {index} is not sample aligned")
        ids = np.asarray(part["trajectory_id"])
        unique = np.unique(ids)
        mapping = {old: next_id + i for i, old in enumerate(unique.tolist())}
        global_ids = np.asarray([mapping[value] for value in ids.tolist()], dtype=np.int64)
        for key in keys:
            values = np.asarray(part[key])
            if key in ("trajectory_id", "episode_id"):
                values = global_ids.copy()
            merged[key].append(values)
        next_id += len(unique)
    return {key: np.concatenate(values, axis=0) for key, values in merged.items()}


def write_sha256_manifest(root, paths, filename="SHA256SUMS.txt"):
    root = Path(root).resolve()
    lines = []
    for path in sorted({Path(p).resolve() for p in paths}):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    output = root / filename
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_run_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
