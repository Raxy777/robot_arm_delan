"""Leakage-safe trajectory-grouped dataset splitting utilities."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SPLIT_NAMES = ("train", "validation", "test")


def load_npz_dataset(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load an NPZ without pickle and detach arrays from the archive handle."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_sample_dataset(dataset: Mapping[str, np.ndarray], group_key="trajectory_id") -> int:
    """Validate a sample-aligned dataset and return its sample count."""
    if not dataset:
        raise ValueError("dataset is empty")
    if group_key not in dataset:
        raise ValueError(f"dataset is missing required group field {group_key!r}")
    groups = np.asarray(dataset[group_key])
    if groups.ndim != 1:
        raise ValueError(f"{group_key} must be one-dimensional")
    n_samples = len(groups)
    if n_samples == 0:
        raise ValueError("dataset contains no samples")
    for name, values in dataset.items():
        array = np.asarray(values)
        if array.ndim == 0 or len(array) != n_samples:
            raise ValueError(
                f"field {name!r} is not sample-aligned: expected first dimension "
                f"{n_samples}, got {array.shape}"
            )
        if array.dtype.hasobject:
            raise ValueError(f"field {name!r} uses an unsafe object dtype")
    if np.issubdtype(groups.dtype, np.floating) and not np.all(np.isfinite(groups)):
        raise ValueError(f"{group_key} contains non-finite IDs")
    return n_samples


def _normalise_ratios(ratios) -> np.ndarray:
    values = np.asarray(ratios, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("ratios must contain three finite values")
    if np.any(values <= 0.0):
        raise ValueError("train, validation, and test ratios must all be positive")
    if not np.isclose(values.sum(), 1.0, atol=1e-9):
        raise ValueError("split ratios must sum to one")
    return values / values.sum()


def _group_counts(n_groups: int, ratios: np.ndarray) -> np.ndarray:
    if n_groups < len(SPLIT_NAMES):
        raise ValueError("at least three trajectories are required for non-empty splits")
    raw = ratios * n_groups
    counts = np.floor(raw).astype(int)
    remainder = n_groups - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    # Preserve all three gates even for highly skewed ratios.
    for empty in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError("cannot construct three non-empty splits")
        counts[donor] -= 1
        counts[empty] += 1
    return counts


def grouped_split_indices(group_ids, ratios=(0.7, 0.2, 0.1), seed=0):
    """Return sample indices split by whole trajectory, never individual rows."""
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("group IDs must be a non-empty one-dimensional array")
    unique_groups = np.unique(groups)
    counts = _group_counts(len(unique_groups), _normalise_ratios(ratios))
    shuffled = np.random.default_rng(seed).permutation(unique_groups)

    result = {}
    start = 0
    for name, count in zip(SPLIT_NAMES, counts):
        selected = shuffled[start:start + count]
        # Preserve chronological/source ordering inside each exported split.
        result[name] = np.flatnonzero(np.isin(groups, selected))
        start += count
    assert_group_disjoint(groups, result)
    return result


def assert_group_disjoint(group_ids, split_indices) -> None:
    """Fail if rows or trajectories leak, disappear, or occur in two splits."""
    groups = np.asarray(group_ids)
    expected_names = set(SPLIT_NAMES)
    if set(split_indices) != expected_names:
        raise ValueError(f"expected split names {SPLIT_NAMES}")
    seen_rows: set[int] = set()
    seen_groups: set = set()
    for name in SPLIT_NAMES:
        indices = np.asarray(split_indices[name])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if len(indices) == 0:
            raise ValueError(f"{name} split is empty")
        if np.any(indices < 0) or np.any(indices >= len(groups)):
            raise ValueError(f"{name} contains out-of-range indices")
        rows = set(map(int, indices))
        if seen_rows.intersection(rows):
            raise ValueError("sample rows leak between splits")
        split_groups = set(np.asarray(groups[indices]).tolist())
        if seen_groups.intersection(split_groups):
            raise ValueError("trajectory IDs leak between splits")
        seen_rows.update(rows)
        seen_groups.update(split_groups)
    if seen_rows != set(range(len(groups))):
        raise ValueError("split rows do not cover the complete dataset")
    if seen_groups != set(np.asarray(groups).tolist()):
        raise ValueError("split trajectories do not cover the complete dataset")


def split_dataset(dataset, ratios=(0.7, 0.2, 0.1), seed=0,
                  group_key="trajectory_id"):
    """Split every sample-aligned field with trajectory-level isolation."""
    validate_sample_dataset(dataset, group_key)
    indices = grouped_split_indices(dataset[group_key], ratios, seed)
    splits = {
        name: {field: np.asarray(values)[rows] for field, values in dataset.items()}
        for name, rows in indices.items()
    }
    return splits, indices


def _json_id(value):
    return value.item() if isinstance(value, np.generic) else value


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, dataset) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **dataset)
    os.replace(temporary, path)


def export_grouped_splits(source_path, output_dir, ratios=(0.7, 0.2, 0.1),
                          seed=0, group_key="trajectory_id"):
    """Create three NPZ files plus a reproducibility/integrity manifest."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    dataset = load_npz_dataset(source_path)
    splits, indices = split_dataset(dataset, ratios, seed, group_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "group_key": group_key,
        "seed": int(seed),
        "ratios": dict(zip(SPLIT_NAMES, map(float, _normalise_ratios(ratios)))),
        "total_samples": int(len(dataset[group_key])),
        "total_trajectories": int(len(np.unique(dataset[group_key]))),
        "splits": {},
    }
    for name in SPLIT_NAMES:
        path = output_dir / f"{name}.npz"
        _atomic_savez(path, splits[name])
        trajectory_ids = np.unique(splits[name][group_key])
        manifest["splits"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "samples": int(len(indices[name])),
            "trajectories": int(len(trajectory_ids)),
            "trajectory_ids": [_json_id(value) for value in trajectory_ids],
        }
    manifest_path = output_dir / "split_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest
