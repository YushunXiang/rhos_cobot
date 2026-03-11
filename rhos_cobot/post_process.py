# coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from piper_msgs.msg import PosCmd
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from rhos_cobot.utils import load_hdf5

DT = 0.02
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]
EEF_NAMES = [
    "pos_x", "pos_y", "pos_z",                      # 末端执行器的位置
    "orient_x", "orient_y", "orient_z", "orient_w",  # 末端执行器的方向
    "joint_last",                                   # 最后一个关节位置
]


def save_videos(video, actions, dt, video_path=None, display=True):
    cam_names = list(video.keys())
    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(video[cam_name])
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]
        image = image[:, :, [2, 1, 0]]  # swap B and R channel
        if display:
            cv2.putText(image, f"Frame: {t+1}/{n_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Left: {np.round(actions[t][:7], 3)}", (
                10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Right: {np.round(actions[t][7:], 3)}", (
                10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("images", image)
            cv2.waitKey(30)
        # print("frame_id: ", t, "left: ", np.round(actions[t][:7], 3), "right: ", np.round(actions[t][7:], 3))
        out.write(image)
    out.release()
    print(f'Saved video to: {video_path}')


def save_depth_videos(video, actions, dt, video_path='depth_video.mp4'):
    # 指定你想保存的深度图相机名称
    cam_names = list(video.keys())

    all_cam_videos = []

    for cam_name in cam_names:
        decoded_video = []
        for encoded in video[cam_name]:
            # 解码为单通道图像
            img = cv2.imdecode(np.frombuffer(
                encoded, np.uint8), cv2.IMREAD_UNCHANGED)

            # 归一化并转换为uint8（如果不是）
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)

            # 伪彩色可视化（可选），否则注释掉这行显示灰度图
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            decoded_video.append(img)
        all_cam_videos.append(decoded_video)

    # 拼接成 (N, H, W_total, 3)
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)

    n_frames, h, w, _ = np.array(all_cam_videos).shape
    fps = int(1 / dt)

    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for t in range(n_frames):
        image = all_cam_videos[t]
        cv2.imshow("depth_video", image)
        cv2.waitKey(int(1000 * dt))
        print("episode_id: ", t,
              "left: ", np.round(actions[t][:7], 3),
              "right: ", np.round(actions[t][7:], 3), "\n")
        out.write(image)

    out.release()
    cv2.destroyAllWindows()
    print(f"Saved depth video to: {video_path}")


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)

    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + \
        [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    # for dim_idx in range(num_dim):
    #     ax = axs[dim_idx]
    #     ax.plot(command[:, dim_idx], label=label2)
    #     ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_base(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(
            20)/20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(
            10)/10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(
            5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_eef(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot eef
    all_names = ['left_' + name for name in EEF_NAMES] + \
        ['right_' + name for name in EEF_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw_data')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'eef {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved eef plot to: {plot_path}')
    plt.close()


def replay_eef_data(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(
        args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(
        args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(
        args.img_front_topic, Image, queue_size=10)

    puppet_arm_left_publisher = rospy.Publisher(
        args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(
        args.puppet_arm_right_topic, JointState, queue_size=10)

    master_arm_left_publisher = rospy.Publisher(
        args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(
        args.master_arm_right_topic, JointState, queue_size=10)

    puppet_eef_left_publisher = rospy.Publisher(
        args.puppet_eef_left_topic, PosCmd, queue_size=10)
    puppet_eef_right_publisher = rospy.Publisher(
        args.puppet_eef_right_topic, PosCmd, queue_size=10)

    robot_base_publisher = rospy.Publisher(
        args.robot_base_topic, Twist, queue_size=10)

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name = args.task_name
    dataset_name = f'episode_{episode_idx}'

    origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
    origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    endleft0 = [0.111376, -0.09219, 0.352633, 0.41211237354116054,
                0.7627680187460423, -0.14576932099436934, 0.47654973109776155]
    endright0 = [0.305041, 0.18106, 0.339395, 0.23205404033907173, -
                 0.9431655689825339, -0.11071169388959519, -0.21055296883269586]

    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    eef_msg = PosCmd()
    # eef_msg.header = Header()

    rate = rospy.Rate(args.frame_rate)
    qposs, qvels, efforts, actions, base_actions, image_dicts, actions_eef = load_hdf5(
        os.path.join(dataset_dir, task_name), dataset_name)

    # if args.only_pub_master:
    #     last_action = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179,
    #                    0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    #     rate = rospy.Rate(100)
    #     for action in actions:
    #         if (rospy.is_shutdown()):
    #             break

    #         new_actions = np.linspace(last_action, action, 20)  # 插值
    #         last_action = action
    #         for act in new_actions:
    #             print(np.round(act[:7], 4))
    #             cur_timestamp = rospy.Time.now()  # 设置时间戳
    #             joint_state_msg.header.stamp = cur_timestamp

    #             joint_state_msg.position = act[:7]
    #             master_arm_left_publisher.publish(joint_state_msg)

    #             joint_state_msg.position = act[7:]
    #             master_arm_right_publisher.publish(joint_state_msg)

    #             if (rospy.is_shutdown()):
    #                 break
    #             rate.sleep()

    # else:
    i = 0

    
    while (not rospy.is_shutdown() and i < len(actions)):
        # , " right: ", np.round(qposs[i][7:], 4))
        print("left: ", np.round(qposs[i][:7], 4))

        cam_names = [k for k in image_dicts.keys()]
        image0 = image_dicts[cam_names[0]][i]
        image0 = image0[:, :, [2, 1, 0]]  # swap B and R channel

        image1 = image_dicts[cam_names[1]][i]
        image1 = image1[:, :, [2, 1, 0]]  # swap B and R channel

        image2 = image_dicts[cam_names[2]][i]
        image2 = image2[:, :, [2, 1, 0]]  # swap B and R channel

        cur_timestamp = rospy.Time.now()  # 设置时间戳

        eef_msg.x, eef_msg.y, eef_msg.z = actions_eef[i][8:11]
        eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(
            actions_eef[i][11:15])
        eef_msg.gripper = actions_eef[i][15:16]
        puppet_eef_right_publisher.publish(eef_msg)

        eef_msg.x, eef_msg.y, eef_msg.z = actions_eef[i][:3]
        eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(
            actions_eef[i][3:7])
        eef_msg.gripper = actions_eef[i][7:8]
        puppet_eef_left_publisher.publish(eef_msg)

        img_front_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
        img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
        img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))

        twist_msg.linear.x = base_actions[i][0]
        twist_msg.angular.z = base_actions[i][1]
        robot_base_publisher.publish(twist_msg)

        i += 1
        rate.sleep()


def replay_joint_data(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(
        args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(
        args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(
        args.img_front_topic, Image, queue_size=10)

    puppet_arm_left_publisher = rospy.Publisher(
        args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(
        args.puppet_arm_right_topic, JointState, queue_size=10)

    master_arm_left_publisher = rospy.Publisher(
        args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(
        args.master_arm_right_topic, JointState, queue_size=10)

    robot_base_publisher = rospy.Publisher(
        args.robot_base_topic, Twist, queue_size=10)

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name = args.task_name
    dataset_name = f'episode_{episode_idx}'

    origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
    origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]

    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    rate = rospy.Rate(args.frame_rate)

    qposs, qvels, efforts, actions, base_actions, image_dicts, _ = load_hdf5(
        os.path.join(dataset_dir, task_name), dataset_name)

    last_action = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179,
                    0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    rate = rospy.Rate(100)
    i=0
    while(not rospy.is_shutdown() and i < len(actions)):
        cam_names = [k for k in image_dicts.keys()]
        image0 = image_dicts[cam_names[0]][i]
        image0 = image0[:, :, [2, 1, 0]]  # swap B and R channel
        
        # show images use OpenCV
        cv2.imshow("image0", image0)
        cv2.waitKey(1)

        action = actions[i]
        # action = np.concatenate([action[7:],action[:7]]) # for old data
        new_actions = np.linspace(last_action, action, 20)  # 插值
        last_action = action
        for act in new_actions:
            print(np.round(act, 4))
            cur_timestamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.header.stamp = cur_timestamp

            joint_state_msg.position = act[:7]
            master_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = act[7:]
            master_arm_right_publisher.publish(joint_state_msg)

            if (rospy.is_shutdown()):
                break
            
            rate.sleep()
        i += 1
        
    cv2.destroyAllWindows()
            


def get_episode_seconds(episode_path):
    """
    Count the number of timesteps in a single episode.
    """
    with h5py.File(episode_path, 'r') as f:
        fps = f.attrs.get('fps', 25)
        if '/observations/qpos' in f:
            num_timesteps = len(f['/observations/qpos'])
        else:
            raise ValueError(
                f"Episode {episode_path} does not contain '/observations/qpos' dataset.")
    return num_timesteps / fps


def get_dataset_seconds(episode_root):
    """
    Count the total number of timesteps in a dataset.
    """
    total_seconds = 0
    for episode_name in os.listdir(episode_root):
        episode_path = os.path.join(episode_root, episode_name)
        if os.path.isfile(episode_path) and episode_path.endswith('.hdf5'):
            total_seconds += get_episode_seconds(episode_path)
    return total_seconds
