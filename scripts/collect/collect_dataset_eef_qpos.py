# -- coding: UTF-8
import os
import argparse
import signal
import sys

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


def get_dataset_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_num', action='store', type=int, help='Episode Number.',
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
    parser.add_argument('--use_depth_image', nargs='?', const=True, type=_str_to_bool, help='use_depth_image',
                        default=False,
                        required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=25, required=False)
    parser.add_argument('--sync_debug_interval', action='store', type=float,
                        help='Seconds between repeated sync-failure diagnostics. Set 0 to print only once per failure run.',
                        default=2.0, required=False)

    args = parser.parse_args()
    return args


def main():
    signal.signal(signal.SIGINT, lambda: sys.exit(0))  # exit when press Ctrl C

    args = get_dataset_arguments()
    # args.use_depth_image = False
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

    print(f"use_depth_img: {args.use_depth_image}")
    num_episode = args.episode_num
    episode_idx = 0
    failed_idx = 0
    uncompleted_idx = 0
    while episode_idx < num_episode:
        while os.path.exists(os.path.join(dataset_dir, "episode_" + str(episode_idx) + ".hdf5")):
            episode_idx += 1
        while os.path.exists(os.path.join(failed_data_dir, "episode_" + str(failed_idx) + ".hdf5")):
            failed_idx += 1
        while os.path.exists(os.path.join(uncompleted_data_dir, "episode_" + str(uncompleted_idx) + ".hdf5")):
            uncompleted_idx += 1

        print("======= Collecting data for episode {} =======".format(episode_idx))
        timesteps, actions, actions_eef, exit_type = ros_operator.process()

        if exit_type == 'f':
            # 按f键退出，保存到失败数据目录
            print("Saving failed demonstration data...")
            dataset_path = os.path.join(
                failed_data_dir, "episode_" + str(failed_idx))
            failed_idx += 1
        elif exit_type == 'normal':
            # 按n键退出，保存到未完成数据目录
            print("Saving uncompleted demonstration data...")
            dataset_path = os.path.join(
                uncompleted_data_dir, "episode_" + str(uncompleted_idx))
            uncompleted_idx += 1
        elif exit_type == 'r':
            print("Skip this demonstration data...")
            continue
        else:
            # 按s键退出，保存到正常数据目录
            print("Saving successful demonstration data...")
            # if(len(actions) < args.max_timesteps):
            #     print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
            #     exit(-1)
            dataset_path = os.path.join(
                dataset_dir, "episode_" + str(episode_idx))
            episode_idx += 1

        save_data(args, timesteps, actions, actions_eef,
                  dataset_path, fps=args.frame_rate)

    print("All episodes collected.")


if __name__ == "__main__":
    main()

# cd collect_data
# python collect_dataset_eef.py --dataset_dir ./data --task_name stir --max_timesteps 500 --episode_num 10
