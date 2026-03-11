
# coding=utf-8
import os
import time
import argparse
import numpy as np
from examples.ocl_robot.pose_util import *
from scipy.spatial.transform import Rotation
from examples.ocl_robot.env_eef import PiperRealEnv
from tf.transformations import quaternion_from_euler, euler_from_quaternion


def compute_transform(raw_pose):
    
    pose_mat = pose_to_mat(quat_to_rot(raw_pose))
    x_rotation_clockwise_90 = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = x_rotation_clockwise_90
    
    pose_mat_out = transform_matrix @ pose_mat
    additional_transform_matrix = np.eye(4)
    additional_R_y = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    additional_R_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    transform_mat =  additional_R_y @ additional_R_z
    additional_transform_matrix[:3, :3] = transform_mat
    pose_mat_out = pose_mat @ additional_transform_matrix
    
    return mat_to_certain_pose_type(pose_mat_out, pose_type="rotvec")


def read_pose_from_raw_data(raw_root, eposide_index):
    eposide_pose_dir = os.path.join(raw_root, f"pose_{eposide_index}")
    pose_files = os.listdir(eposide_pose_dir)
    pose_files = [f for f in pose_files if f.endswith('.npz')]
    pose_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_pose = []

    for pose_f in pose_files:
        pose_path = os.path.join(eposide_pose_dir, pose_f)
        pose_data = np.load(pose_path)
        all_pose.extend(pose_data["pose"])
    return all_pose

def run_replay(args):
    
    env_eef = PiperRealEnv(
        init_node=True)
    
    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    
    actions_eef = read_pose_from_raw_data(dataset_dir, episode_idx)
    # rospy = env_eef.ros_operator
    print_flag = True
    i = 0
    while i < len(actions_eef):
        
        result = env_eef.get_observation()
        
        qpos = result["qpos"]
        left_arm_pose = qpos[0:7]
        left_arm_gripper = qpos[7]
        
        right_arm_pose = qpos[8:15]
        right_arm_gripper = qpos[15]
        
        print("right_arm_pose", right_arm_pose)
        
        left_abs_pose = []
        left_pose_rot = euler_from_quaternion(left_arm_pose[3:7])
        left_pose = np.array(list(left_arm_pose[0:3]) + list(left_pose_rot))
        left_abs_pose.append(left_pose)
        
        right_abs_pose = []
        right_pose_rot = euler_from_quaternion(right_arm_pose[3:7])
        right_pose = np.array(list(right_arm_pose[0:3]) + list(right_pose_rot))
        right_abs_pose.append(right_pose)
        
        refer_eepose = compute_transform(actions_eef[i])
        
        print("delta xyz:", actions_eef[i+1][:3] - actions_eef[i][:3])
        
        refer_obs_data = {
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_gripper_width': []
        }

        refer_mat = pose_to_mat(refer_eepose)
        refer_ee_vec = mat_to_certain_pose_type(refer_mat, pose_type="rotvec")
        refer_obs_data['robot0_eef_pos'].append(refer_ee_vec[:3])
        refer_obs_data['robot0_eef_rot_axis_angle'].append(refer_ee_vec[3:])
        refer_obs_data['robot0_gripper_width'].append(np.array(right_arm_gripper))
        
        episode_start_pose = np.concatenate([refer_obs_data['robot0_eef_pos'], refer_obs_data['robot0_eef_rot_axis_angle']], axis=-1)[-1]
        start_pose_mat = certain_pose_type_to_mat(episode_start_pose, pose_type="rotvec")
        
        s = time.time()
        obs_data = {
                'robot0_eef_pos': [],
                'robot0_eef_rot_axis_angle': [],
                'robot0_gripper_width': []
            }
        
        eepose = compute_transform(actions_eef[i+1])

        obs_data['robot0_eef_pos'].append(eepose[:3] )
        obs_data['robot0_eef_rot_axis_angle'].append( eepose[3:] )
        obs_data['robot0_gripper_width'].append(np.array(right_arm_gripper))
            
        for key in obs_data.keys():
            obs_data[key] = np.stack(obs_data[key], axis=0)
            
        pos_mat = certain_pose_type_to_mat(np.concatenate([obs_data['robot0_eef_pos'], obs_data['robot0_eef_rot_axis_angle']], axis=-1),pose_type="rotvec")

        real_bos_pose_mat = convert_pose_mat_rep(pose_mat=pos_mat,
                                                    base_pose_mat=start_pose_mat,
                                                    pose_rep="relative",
                                                    backward=False)
        
        rel_obs_pose = mat_to_certain_pose_type(real_bos_pose_mat, "10d")
        
        raw_action =to_torch(np.concatenate([rel_obs_pose, [obs_data['robot0_gripper_width']]], axis=-1))
        in_abs_pose = np.concatenate([right_abs_pose, [obs_data['robot0_gripper_width']]], axis=-1)
        
        this_target_poses = get_real_umi_inference_action(raw_action.cpu().numpy(), in_abs_pose, "relative")
        this_target_poses[:, :3] = this_target_poses[:, :3] * 1000
        to_action = np.concatenate([np.array(left_abs_pose), this_target_poses], axis=-1)
        print("to action:", to_action[0][7:])
        input()
        env_eef.step(to_action[0], STOP=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store',
                        type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int,
                        help='Episode index.', default=0, required=False)

    args = parser.parse_args()
    run_replay(args)
    
