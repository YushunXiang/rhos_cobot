# -- coding: UTF-8
import os
import argparse

from rhos_cobot.data_collection import RosOperator, save_data


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("1", "true", "t", "yes", "y"):
        return True
    if normalized in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)

    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)

    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)

    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)

    # topic name of eef
    parser.add_argument('--puppet_eef_left_topic', action='store', type=str, help='puppet_eef_left_topic',
                        default='/puppet/end_pose_left', required=False)
    parser.add_argument('--puppet_eef_right_topic', action='store', type=str, help='puppet_eef_right_topic',
                        default='/puppet/end_pose_right', required=False)

    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)

    parser.add_argument('--use_robot_base', nargs='?', const=True, type=_str_to_bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    # parser.add_argument('--use_depth_image', action='store_true', help='use_depth_image')
    parser.add_argument('--use_depth_image', nargs='?', const=True, type=_str_to_bool, help='use_depth_image',
                        default=False, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=25, required=False)
    parser.add_argument('--sync_debug_interval', action='store', type=float,
                        help='Seconds between repeated sync-failure diagnostics. Set 0 to print only once per failure run.',
                        default=2.0, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    # args.use_depth_image = False
    print(f"use_depth_img: {args.use_depth_image}")
    ros_operator = RosOperator(args)
    
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    failed_data_dir = os.path.join(
        args.dataset_dir, args.task_name, "failed_data")
    uncompleted_data_dir = os.path.join(
        args.dataset_dir, args.task_name, "uncompleted_data")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(failed_data_dir):
        os.makedirs(failed_data_dir)
    if not os.path.exists(uncompleted_data_dir):
        os.makedirs(uncompleted_data_dir)
        
    episode_idx = args.episode_idx

    timesteps, actions, actions_eef, exit_type = ros_operator.process()

    if exit_type == 'f':
        # 按f键退出，保存到失败数据目录
        print("Saving failed demonstration data...")
        while os.path.exists(os.path.join(failed_data_dir, "episode_" + str(episode_idx) + ".hdf5")):
            episode_idx += 1
        dataset_path = os.path.join(
            failed_data_dir, "episode_" + str(episode_idx))
    elif exit_type == 'normal':
        # 按n键退出，保存到未完成数据目录
        print("Saving uncompleted demonstration data...")
        while os.path.exists(os.path.join(uncompleted_data_dir, "episode_" + str(episode_idx) + ".hdf5")):
            episode_idx += 1
        dataset_path = os.path.join(
            uncompleted_data_dir, "episode_" + str(episode_idx))
    elif exit_type == 'r':
        print("Skip this demonstration data...")
    else:
        # 按s键退出，保存到正常数据目录
        print("Saving successful demonstration data...")
        # if(len(actions) < args.max_timesteps):
        #     print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
        #     exit(-1)
        while os.path.exists(os.path.join(dataset_dir, "episode_" + str(episode_idx) + ".hdf5")):
            episode_idx += 1
        dataset_path = os.path.join(
            dataset_dir, "episode_" + str(episode_idx))

    save_data(args, timesteps, actions, actions_eef,
              dataset_path, fps=args.frame_rate)
    


if __name__ == '__main__':
    main()
