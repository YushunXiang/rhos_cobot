#!/usr/bin/env python3
# coding=utf-8

"""Overwrite source episode files from fixed_stage with stage-3 removal.

Default behavior is a dry run that only prints the planned operations.
Use --apply to actually overwrite files.
Before replacement, the source file is renamed in place to episode_*.hdf5.bkup.
If the corrected file contains stage == 3, all stage-3 frames are removed from
the copied HDF5.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np


@dataclass
class OverwritePlan:
    name: str
    fixed_path: Path
    source_path: Path
    backup_path: Path
    size_bytes: int
    fixed_frames: int | None
    drop_count: int


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use fixed_stage/episode_*.hdf5 to overwrite same-named source files. "
            "Original source files are renamed to .hdf5.bkup before replacement."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset root containing episode_*.hdf5 and fixed_stage/.",
    )
    parser.add_argument(
        "--fixed_dir_name",
        type=str,
        default="fixed_stage",
        help="Subdirectory containing corrected HDF5 files. Default: fixed_stage",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="episode_*.hdf5",
        help="Filename glob to select corrected files inside fixed_stage. Default: episode_*.hdf5",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform overwrite. Without this flag the script only prints a dry run plan.",
    )
    parser.add_argument(
        "--skip_missing_source",
        action="store_true",
        help="Skip corrected files that do not have a same-named source file in dataset_dir.",
    )
    return parser.parse_args(argv)


def iter_fixed_files(fixed_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted((path for path in fixed_dir.glob(pattern) if path.is_file()), key=lambda p: p.name)


def first_existing_frame_count(root: h5py.File) -> int | None:
    candidates = [
        "/observations/qpos",
        "/action",
        "/base_action",
        "/action_eef",
    ]
    for name in candidates:
        if name in root:
            dataset = root[name]
            if dataset.shape:
                return int(dataset.shape[0])

    image_group = root.get("/observations/images")
    if isinstance(image_group, h5py.Group):
        for camera_name in image_group.keys():
            dataset = image_group[camera_name]
            if dataset.shape:
                return int(dataset.shape[0])
    return None


def inspect_fixed_file(path: Path) -> tuple[int | None, int]:
    """Return (frame_count, number_of_stage3_frames) for a fixed HDF5 file."""
    with h5py.File(path, "r") as root:
        n_frames = first_existing_frame_count(root)
        if "/stage" not in root:
            return n_frames, 0

        stage = np.asarray(root["/stage"][()]).reshape(-1)
        return n_frames, int(np.sum(stage == 3))


def backup_path_for(source_path: Path) -> Path:
    return source_path.with_suffix(source_path.suffix + ".bkup")


def collect_plans(
    dataset_dir: Path,
    fixed_dir: Path,
    pattern: str,
    skip_missing_source: bool,
) -> list[OverwritePlan]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    if not fixed_dir.is_dir():
        raise FileNotFoundError(f"fixed_dir does not exist: {fixed_dir}")

    plans: list[OverwritePlan] = []
    missing_sources: list[str] = []

    for fixed_path in iter_fixed_files(fixed_dir, pattern):
        source_path = dataset_dir / fixed_path.name
        if not source_path.exists():
            if skip_missing_source:
                print(f"[SKIP] source file missing: {source_path}")
                continue
            missing_sources.append(fixed_path.name)
            continue

        fixed_frames, drop_count = inspect_fixed_file(fixed_path)
        plans.append(
            OverwritePlan(
                name=fixed_path.name,
                fixed_path=fixed_path,
                source_path=source_path,
                backup_path=backup_path_for(source_path),
                size_bytes=fixed_path.stat().st_size,
                fixed_frames=fixed_frames,
                drop_count=drop_count,
            )
        )

    if missing_sources:
        missing = ", ".join(missing_sources)
        raise FileNotFoundError(
            f"Missing same-named source files in dataset_dir: {missing}. "
            "Use --skip_missing_source to ignore them."
        )

    return plans


def print_plan(dataset_dir: Path, fixed_dir: Path, plans: list[OverwritePlan], apply: bool) -> None:
    mode = "APPLY" if apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Fixed dir: {fixed_dir}")
    print(f"Matched files: {len(plans)}")
    if not plans:
        print("No matching fixed files found.")
        return

    for plan in plans:
        print(
            f"- {plan.name}: fixed={plan.fixed_path} -> source={plan.source_path} "
            f"(backup_rename={plan.backup_path}, frames={plan.fixed_frames}, drop_stage3={plan.drop_count}, size={plan.size_bytes} bytes)"
        )


def copy_attrs(src_obj: h5py.AttributeManager | h5py.Group | h5py.Dataset, dst_obj: h5py.Group | h5py.Dataset) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def copy_node(
    src_node: h5py.Group | h5py.Dataset,
    dst_parent: h5py.Group | h5py.File,
    name: str,
    frame_count: int | None,
    keep_indices: np.ndarray | None,
) -> None:
    if isinstance(src_node, h5py.Group):
        dst_group = dst_parent.create_group(name)
        copy_attrs(src_node, dst_group)
        for child_name, child_node in src_node.items():
            copy_node(child_node, dst_group, child_name, frame_count, keep_indices)
        return

    if src_node.shape == ():
        dst_dataset = dst_parent.create_dataset(name, data=src_node[()], dtype=src_node.dtype)
        copy_attrs(src_node, dst_dataset)
        return

    if keep_indices is not None and frame_count is not None and src_node.shape and src_node.shape[0] == frame_count:
        data = src_node[list(keep_indices)]
    else:
        data = src_node[()]

    dst_dataset = dst_parent.create_dataset(name, data=data, dtype=src_node.dtype)
    copy_attrs(src_node, dst_dataset)


def compute_keep_indices(fixed_path: Path) -> np.ndarray | None:
    """Return indices of frames where stage != 3, or None if no filtering needed."""
    with h5py.File(fixed_path, "r") as root:
        if "/stage" not in root:
            return None
        stage = np.asarray(root["/stage"][()]).reshape(-1)
        if np.all(stage != 3):
            return None
        keep = np.flatnonzero(stage != 3)
        if keep.size == 0:
            raise ValueError(
                f"All frames in {fixed_path.name} are stage 3. "
                "This would produce an empty episode."
            )
        return keep


def write_corrected_copy(fixed_path: Path, target_path: Path, drop_count: int) -> None:
    if drop_count == 0:
        shutil.copy2(fixed_path, target_path)
        return

    keep_indices = compute_keep_indices(fixed_path)
    if keep_indices is None:
        shutil.copy2(fixed_path, target_path)
        return

    with h5py.File(fixed_path, "r") as src_root:
        frame_count = first_existing_frame_count(src_root)
        if frame_count is None:
            raise ValueError(f"Unable to infer frame count from fixed file: {fixed_path}")

        with h5py.File(target_path, "w") as dst_root:
            copy_attrs(src_root, dst_root)
            for name, node in src_root.items():
                copy_node(node, dst_root, name, frame_count, keep_indices)


def overwrite_one(plan: OverwritePlan) -> None:
    if plan.backup_path.exists():
        raise FileExistsError(f"Backup already exists, refusing to overwrite: {plan.backup_path}")

    tmp_path = plan.source_path.with_suffix(plan.source_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        write_corrected_copy(plan.fixed_path, tmp_path, plan.drop_count)
        os.replace(plan.source_path, plan.backup_path)
        try:
            os.replace(tmp_path, plan.source_path)
        except Exception:
            os.replace(plan.backup_path, plan.source_path)
            raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    fixed_dir = dataset_dir / args.fixed_dir_name

    try:
        plans = collect_plans(
            dataset_dir=dataset_dir,
            fixed_dir=fixed_dir,
            pattern=args.pattern,
            skip_missing_source=args.skip_missing_source,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_plan(dataset_dir, fixed_dir, plans, apply=args.apply)
    if not plans:
        return 0

    if not args.apply:
        print("\nDry run only. Re-run with --apply to overwrite source files.")
        print("Source backups will be renamed in place to episode_*.hdf5.bkup.")
        return 0

    try:
        for plan in plans:
            overwrite_one(plan)
            print(f"[DONE] {plan.name}")
    except Exception as exc:
        print(f"[ERROR] overwrite stopped: {exc}", file=sys.stderr)
        return 1

    print("\nOverwrite completed successfully.")
    print("Original files have been renamed in place to episode_*.hdf5.bkup.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
