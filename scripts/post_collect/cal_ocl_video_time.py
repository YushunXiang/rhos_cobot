import os
import argparse
import h5py
from pathlib import Path


def get_episode_frames(episode_path):
    with h5py.File(episode_path, 'r') as f:
        if '/observations/qpos' not in f:
            raise ValueError(f"{episode_path} does not contain '/observations/qpos'")
        return len(f['/observations/qpos'])


EXCLUDE_DIRS = {'failed_data', 'fixed_stage', 'uncompleted_data'}


def calc_task_duration(task_dir, fps):
    hdf5_files = sorted(
        f for f in Path(task_dir).rglob('*.hdf5')
        if not any(part in EXCLUDE_DIRS for part in f.parts)
    )
    total_frames = 0
    for f in hdf5_files:
        total_frames += get_episode_frames(str(f))
    return len(hdf5_files), total_frames / fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 ocl_data 下各 task 的视频总时长")
    parser.add_argument('--dataset_dir', type=str, default='./ocl_data', help='数据根目录')
    parser.add_argument('--task_names', nargs='*', default=None, help='指定 task 名称列表，不指定则统计全部')
    parser.add_argument('--fps', type=int, default=25, help='帧率，默认 25')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"目录不存在: {dataset_dir}")
        exit(1)

    if args.task_names:
        task_names = args.task_names
    else:
        task_names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())

    if not task_names:
        print("未找到任何 task 目录")
        exit(1)

    total_episodes = 0
    total_seconds = 0.0
    rows = []

    for task_name in task_names:
        task_dir = dataset_dir / task_name
        if not task_dir.exists():
            print(f"警告: 目录不存在，跳过 {task_dir}")
            continue
        n_eps, secs = calc_task_duration(task_dir, args.fps)
        rows.append((task_name, n_eps, secs))
        total_episodes += n_eps
        total_seconds += secs

    name_w = max(len(r[0]) for r in rows) if rows else 20
    sep = "─" * (name_w + 42)

    for task_name, n_eps, secs in rows:
        print(f"{task_name:<{name_w}}: {n_eps:>4} episodes, {secs:>8.1f}s ({secs/60:>5.1f} min)")

    print(sep)
    print(f"{'Total':<{name_w}}: {total_episodes:>4} episodes, {total_seconds:>8.1f}s ({total_seconds/60:>5.1f} min)")
