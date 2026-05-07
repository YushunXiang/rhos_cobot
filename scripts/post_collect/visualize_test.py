import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 配置
episode_dir = "/your/episode/dir"  # 替换为你的episode文件夹路径
dataset_key = "observations/qpos"  # 你要分析的数据key
joint_idx = 0                      # 你要分析的关节维度（如0~13）

def main() -> int:
    all_values = []

    # 遍历所有episode文件
    for fname in os.listdir(episode_dir):
        if not fname.endswith(".hdf5"):
            continue
        fpath = os.path.join(episode_dir, fname)
        with h5py.File(fpath, "r") as f:
            if dataset_key in f:
                data = f[dataset_key][:]
                if data.shape[1] > joint_idx:
                    all_values.append(data[:, joint_idx])
                else:
                    print(f"{fname} 数据维度不足，跳过。")
            else:
                print(f"{fname} 不包含 {dataset_key}，跳过。")

    if not all_values:
        print("没有找到任何数据。")
        return 1

    all_values = np.concatenate(all_values, axis=0)

    # 绘制散点图
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(len(all_values)), all_values, s=2, alpha=0.6)
    plt.title(f"Joint {joint_idx} Scatter")
    plt.xlabel("Step")
    plt.ylabel("Value")

    # 绘制箱形图
    plt.subplot(1, 2, 2)
    plt.boxplot(all_values, vert=True, showfliers=True)
    plt.title(f"Joint {joint_idx} Boxplot")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
