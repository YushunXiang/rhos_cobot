"""
Script to convert local Aloha hdf5 data to the LeRobot dataset v2.0 format.
Stores the converted dataset in the 'convert' subdirectory.

Usage: python convert_eef.py
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder", 
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    motors_eef = [
        "x_l",
        "y_l",
        "z_l",
        "qx_l",
        "qy_l",
        "qz_l",
        "qw_l",
        "gripper_l",
        "x_r",
        "y_r",
        "z_r",
        "qx_r",
        "qy_r",
        "qz_r",
        "qw_r",
        "gripper_r",
    ]
    
    # 根据实际数据调整相机列表
    cameras = [
        "cam_high",
        "cam_left_wrist", 
        "cam_right_wrist",
    ]

    # state&action use eef
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors_eef),),
            "names": [
                motors_eef,
            ],
        },
        "action": {
            "dtype": "float32", 
            "shape": (len(motors_eef),),
            "names": [
                motors_eef,
            ],
        },
    }
    # velocity&effort use joint space
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height", 
                "width",
            ],
        }

    # 创建本地数据集目录
    local_dataset_path = Path("convert") / repo_id
    if local_dataset_path.exists():
        shutil.rmtree(local_dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def swap_left_right_arms(data: torch.Tensor) -> torch.Tensor:
    """
    交换左右臂的运动数据
    
    Args:
        data: 形状为 (num_frames, 14) 的张量，包含左右臂的运动数据
              顺序为: [right_waist, right_shoulder, right_elbow, right_forearm_roll, 
                      right_wrist_angle, right_wrist_rotate, right_gripper,
                      left_waist, left_shoulder, left_elbow, left_forearm_roll,
                      left_wrist_angle, left_wrist_rotate, left_gripper]
    
    Returns:
        交换后的数据张量
    """
    swapped_data = data
    
    # 交换左右臂数据 (前7个是右臂，后7个是左臂)
    right_arm_data = data[:, :7]  # 原来的右臂数据
    left_arm_data = data[:, 7:14]  # 原来的左臂数据
    
    # 交换: 原来的左臂数据放到右臂位置，原来的右臂数据放到左臂位置
    swapped_data[:, :7] = left_arm_data
    swapped_data[:, 7:14] = right_arm_data
    
    return swapped_data


def extract_episode_number(ep_path: Path) -> int:
    """
    从文件名中提取 episode 编号
    
    Args:
        ep_path: episode 文件的路径，格式应为 episode_{num}.hdf5
    
    Returns:
        episode 编号
    """
    import re
    
    filename = ep_path.name
    match = re.match(r'episode_(\d+)\.hdf5', filename)
    if match:
        return int(match.group(1))
    else:
        # 如果无法提取编号，返回一个默认值，这样会进行交换
        print(f"Warning: Could not extract episode number from {filename}, defaulting to swap arms")
        return 0


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                # 确保数据是正确的格式
                if isinstance(data, bytes):
                    # 如果是字节数据，直接使用
                    img_data = np.frombuffer(data, dtype=np.uint8)
                elif hasattr(data, 'tobytes'):
                    # 如果是其他格式，转换为字节
                    img_data = np.frombuffer(data.tobytes(), dtype=np.uint8)
                else:
                    # 如果是numpy数组，直接使用
                    img_data = np.array(data, dtype=np.uint8)
                
                # 解码图像
                decoded_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if decoded_img is not None:
                    # imgs_array.append(cv2.cvtColor(decoded_img, cv2.IMREAD_COLOR))
                    imgs_array.append(cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB))
                else:
                    print(f"Warning: Failed to decode image data for camera {camera}")
                    # 创建一个空白图像作为替代
                    imgs_array.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    cameras: list[str],
    swap_arms: bool = True,  # 添加参数控制是否交换左右臂
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/action_eef"][:])
        print(f"Loaded state shape: {state.shape} from {ep_path.name}")
        
        action = state[1:]
        action = np.append(action, action[-1][np.newaxis, :], axis=0)

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # 根据文件名判断是否需要交换左右臂数据
        # episode_num = extract_episode_number(ep_path)
        # should_swap = swap_arms and not (90 <= episode_num <= 99)
        
        # if should_swap:
        #     state = swap_left_right_arms(state)
        #     action = swap_left_right_arms(action)
        #     if velocity is not None:
        #         velocity = swap_left_right_arms(velocity)
        #     if effort is not None:
        #         effort = swap_left_right_arms(effort)

        imgs_per_cam = load_raw_images_per_camera(ep, cameras)

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    cameras: list[str],
    task: str,
    episodes: list[int] | None = None,
    swap_arms: bool = False,  # 添加参数控制是否交换左右臂
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        episode_num = extract_episode_number(ep_path)
        should_swap = swap_arms and not (90 <= episode_num <= 99)
        
        print(f"Processing episode {ep_idx}: {ep_path.name}")

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, cameras, swap_arms)
        num_frames = state.shape[0]
        
        swap_status = " (arms swapped)" if should_swap else " (arms NOT swapped)" if swap_arms else ""
        print(f"Episode has {num_frames} frames{swap_status}")

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    return dataset


def convert_local_aloha_data(
    raw_dir: str = ".",
    repo_id: str = "task0063_dataset",
    task: str = "task0063",
    *,
    episodes: list[int] | None = None,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    swap_arms: bool = True,  # 添加参数控制是否交换左右臂
):
    """
    转换当前目录中的 Aloha HDF5 数据到 LeRobot 格式
    
    Args:
        raw_dir: 包含 HDF5 文件的目录
        repo_id: 数据集名称
        task: 任务名称
        episodes: 要转换的特定 episode 列表，None 表示转换所有
        is_mobile: 是否为移动 Aloha
        mode: 图像存储模式 ("video" 或 "image")
        dataset_config: 数据集配置
        swap_arms: 是否交换左右臂数据（修复录制时的连接错误）
    """
    raw_dir_path = Path(raw_dir)
    
    if not raw_dir_path.exists():
        raise ValueError(f"Raw directory {raw_dir} does not exist")

    # 只获取当前目录中的 HDF5 文件，不包括子目录
    hdf5_files = sorted(raw_dir_path.glob("episode_*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No episode_*.hdf5 files found in {raw_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files to convert")
    if swap_arms:
        print("左右臂数据将被交换以修复录制时的连接错误")
        print("注意: episode_90.hdf5 到 episode_99.hdf5 将不会交换左右臂数据")
    
    # 获取相机列表
    cameras = get_cameras(hdf5_files)
    print(f"Found cameras: {cameras}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        cameras,
        task=task,
        episodes=episodes,
        swap_arms=swap_arms,
    )
    
    print("Dataset conversion completed, preparing to save...")
    
    # 移动数据集到本地目录
    try:
        source_path = Path("/home/agilex/.cache/huggingface/lerobot") / repo_id
        target_path = Path("convert") / repo_id
        
        if source_path.exists():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.move(str(source_path), str(target_path))
            print(f"Dataset moved to: {target_path}")
        else:
            print(f"Dataset saved to default location: {source_path}")
    except Exception as e:
        print(f"Note: Could not move dataset to local directory: {e}")
        print(f"Dataset saved to default LeRobot location")
    
    print(f"Dataset conversion completed!")


if __name__ == "__main__":
    # 转换当前目录中的 HDF5 数据
    convert_local_aloha_data(
        raw_dir="data/tube_transfer_ac_view",  # 当前目录
        repo_id="tube_transfer_ac",
        task="tube_transfer",
        mode="video",  # 使用视频模式
        swap_arms=False,  # 启用左右臂数据交换来修复录制错误
    )