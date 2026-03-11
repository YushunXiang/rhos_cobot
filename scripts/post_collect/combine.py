#!/usr/bin/env python3
"""
合并两个目录中的 hdf5 文件到目标目录：
- 仅处理源目录根下的 .hdf5 文件（不递归）；但若存在 fixed/ 目录，则使用其中与根目录同名的文件进行覆盖；
- 目标文件以 episode_{idx}.hdf5 命名，idx 连续并顺延（若目标已有文件，则从最大 idx+1 开始）。

用法：
  python scripts/post_collect/combine.py SRC1 SRC2 DST

示例：
  python scripts/post_collect/combine.py /data/a /data/b /data/merged
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple


EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$")


@dataclass
class SourcePlan:
	root: str
	files_ordered: List[Tuple[str, str]]  # (filename, abs_path)
	overrides: List[str]  # filenames that were taken from fixed/ instead of root


def list_hdf5_root_only(path: str) -> Dict[str, str]:
	"""List .hdf5 files directly under path (non-recursive). Return mapping name->abs_path."""
	result: Dict[str, str] = {}
	try:
		for name in os.listdir(path):
			p = os.path.join(path, name)
			if os.path.isfile(p) and name.lower().endswith(".hdf5"):
				result[name] = p
	except FileNotFoundError:
		pass
	return result


def collect_source_plan(src: str) -> SourcePlan:
	root_map = list_hdf5_root_only(src)

	fixed_dir = os.path.join(src, "fixed")
	fixed_map: Dict[str, str] = {}
	if os.path.isdir(fixed_dir):
		fixed_map = list_hdf5_root_only(fixed_dir)

	overrides: List[str] = []

	# Only take files present in root, but override them with fixed/ if same name exists
	chosen: Dict[str, str] = {}
	for name, root_path in root_map.items():
		if name in fixed_map:
			overrides.append(name)
			chosen[name] = fixed_map[name]
		else:
			chosen[name] = root_path

	# Ordering: numeric by episode idx when possible, else lexicographic
	def sort_key(item: Tuple[str, str]):
		fname, _ = item
		m = EPISODE_RE.match(fname)
		return (0, int(m.group(1))) if m else (1, fname)

	files_ordered = sorted(chosen.items(), key=sort_key)
	return SourcePlan(root=src, files_ordered=files_ordered, overrides=overrides)


def next_destination_index(dst: str) -> int:
	"""Determine next episode index in destination directory."""
	if not os.path.isdir(dst):
		return 0
	max_idx = -1
	try:
		for name in os.listdir(dst):
			m = EPISODE_RE.match(name)
			if m:
				max_idx = max(max_idx, int(m.group(1)))
	except FileNotFoundError:
		return 0
	return max_idx + 1


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def copy_with_new_names(plans: List[SourcePlan], dst: str) -> Tuple[int, List[str]]:
	"""Copy files according to plans into dst with sequential episode indices.
	Returns (count, logs).
	"""
	ensure_dir(dst)
	start_idx = next_destination_index(dst)
	cur_idx = start_idx
	logs: List[str] = []

	for plan in plans:
		# Log overrides for this source
		if plan.overrides:
			logs.append(f"[源:{plan.root}] 使用 fixed 覆盖同名文件: {', '.join(sorted(plan.overrides))}")

		for fname, src_path in plan.files_ordered:
			dst_name = f"episode_{cur_idx}.hdf5"
			dst_path = os.path.join(dst, dst_name)

			# Safety: avoid overwrite if somehow exists
			while os.path.exists(dst_path):
				cur_idx += 1
				dst_name = f"episode_{cur_idx}.hdf5"
				dst_path = os.path.join(dst, dst_name)

			shutil.copy2(src_path, dst_path)
			logs.append(f"复制: {plan.root}/{fname} -> {dst_name}")
			cur_idx += 1

	count = cur_idx - start_idx
	return count, logs


def parse_args(argv: List[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="合并两个目录中的 hdf5 文件到目标目录（支持 fixed 覆盖），目标文件顺延 episode 索引")
	p.add_argument("src1", help="源目录1")
	p.add_argument("src2", help="源目录2")
	p.add_argument("dst", help="目标目录（不存在会自动创建）")
	return p.parse_args(argv)


def write_log(dst: str, lines: List[str]) -> None:
	ensure_dir(dst)
	log_path = os.path.join(dst, "combine_log.txt")
	try:
		with open(log_path, "a", encoding="utf-8") as f:
			f.write("\n" + "-" * 60 + "\n")
			f.write("combine run\n")
			for line in lines:
				f.write(line + "\n")
	except Exception:
		# Logging failure shouldn't stop the main operation
		pass


def main(argv: List[str]) -> int:
	args = parse_args(argv)
	src1, src2, dst = args.src1, args.src2, args.dst

	# Validate sources
	errors: List[str] = []
	if not os.path.isdir(src1):
		errors.append(f"源目录不存在: {src1}")
	if not os.path.isdir(src2):
		errors.append(f"源目录不存在: {src2}")
	if errors:
		for e in errors:
			print(e, file=sys.stderr)
		return 1

	plan1 = collect_source_plan(src1)
	plan2 = collect_source_plan(src2)

	files_total = len(plan1.files_ordered) + len(plan2.files_ordered)
	if files_total == 0:
		print("两个源目录均未找到 .hdf5 文件（仅检查根目录，fixed 仅用于同名覆盖）")
		return 0

	# Perform copy
	count, logs = copy_with_new_names([plan1, plan2], dst)

	# Output logs
	print(f"目标目录: {dst}")
	for line in logs:
		print(line)
	print(f"合并完成，共复制 {count} 个文件。")

	# Persist logs
	write_log(dst, [f"目标目录: {dst}"] + logs + [f"合并完成，共复制 {count} 个文件。"])
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))

