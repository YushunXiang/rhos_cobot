# coding=utf-8
import argparse
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from rhos_cobot.utils import load_hdf5

DEFAULT_CAMERA_NAMES = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
JOINT_NAMES = ['joint0', 'joint1', 'joint2',
               'joint3', 'joint4', 'joint5', 'joint6']
MASTER_HOME = np.array(
    [-0.0057, -0.0310, -0.0122, -0.0320, 0.0099, 0.0179, 0.2279,
     0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279],
    dtype=np.float64,
)
MASTER_ONLY_INTERPOLATION_STEPS = 20
MASTER_ONLY_FRAME_RATE = 100


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store',
                        type=str, help='Dataset dir.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default='aloha_mobile_dummy', required=False)
    parser.add_argument('--episode_idx', action='store', type=int,
                        help='Episode index.', default=0, required=False)
    parser.add_argument(
        '--camera_names',
        nargs='*',
        default=DEFAULT_CAMERA_NAMES,
        help='Camera names to replay. Defaults to the standard front/left/right cameras.',
    )
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=25, required=False)
    parser.add_argument('--only_pub_master', action='store_true',
                        help='Only publish master joint commands with interpolation.', required=False)
    return parser


def _publish_joint_state(publisher, positions, stamp):
    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.header.stamp = stamp
    joint_state_msg.name = JOINT_NAMES
    joint_state_msg.position = np.asarray(positions, dtype=np.float64).tolist()
    publisher.publish(joint_state_msg)


def _decode_image_frame(frame):
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        image = frame
    else:
        if isinstance(frame, np.ndarray):
            encoded = frame.tobytes()
        else:
            encoded = bytes(frame)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Failed to decode image frame from dataset.')
    return image[:, :, [2, 1, 0]]


def _resolve_camera_names(requested_names, image_dicts):
    if requested_names:
        missing = [name for name in requested_names if name not in image_dicts]
        if not missing:
            return requested_names
        rospy.logwarn(
            'Missing cameras in dataset: %s. Fall back to dataset camera order.',
            ', '.join(missing),
        )
    return list(image_dicts.keys())


def _publish_images(image_dicts, image_publishers, bridge, step_idx, camera_names):
    for publisher, cam_name in zip(image_publishers, camera_names):
        image = _decode_image_frame(image_dicts[cam_name][step_idx])
        publisher.publish(bridge.cv2_to_imgmsg(image, 'bgr8'))


def _publish_master_only(actions, publishers):
    left_master_publisher, right_master_publisher = publishers
    rate = rospy.Rate(MASTER_ONLY_FRAME_RATE)
    last_action = MASTER_HOME.copy()

    for action in actions:
        if rospy.is_shutdown():
            break

        action = np.asarray(action, dtype=np.float64)
        interpolated_actions = np.linspace(
            last_action, action, MASTER_ONLY_INTERPOLATION_STEPS)
        last_action = action

        for interpolated_action in interpolated_actions:
            if rospy.is_shutdown():
                break

            stamp = rospy.Time.now()
            _publish_joint_state(
                left_master_publisher, interpolated_action[:7], stamp)
            _publish_joint_state(
                right_master_publisher, interpolated_action[7:], stamp)
            print(np.round(interpolated_action, 4))
            rate.sleep()


def _publish_full_replay(args, qposs, actions, base_actions, image_dicts, publishers):
    bridge = CvBridge()
    img_publishers, master_publishers, puppet_publishers, robot_base_publisher = publishers
    rate = rospy.Rate(args.frame_rate)
    camera_names = _resolve_camera_names(args.camera_names, image_dicts)

    for step_idx in range(len(actions)):
        if rospy.is_shutdown():
            break

        stamp = rospy.Time.now()
        action = np.asarray(actions[step_idx], dtype=np.float64)
        qpos = np.asarray(qposs[step_idx], dtype=np.float64)

        _publish_joint_state(master_publishers[0], action[:7], stamp)
        _publish_joint_state(master_publishers[1], action[7:], stamp)
        _publish_joint_state(puppet_publishers[0], qpos[:7], stamp)
        _publish_joint_state(puppet_publishers[1], qpos[7:], stamp)
        _publish_images(image_dicts, img_publishers, bridge, step_idx, camera_names)

        if base_actions is not None and step_idx < len(base_actions):
            twist_msg = Twist()
            twist_msg.linear.x = float(base_actions[step_idx][0])
            twist_msg.angular.z = float(base_actions[step_idx][1])
            robot_base_publisher.publish(twist_msg)

        print('left: ', np.round(qpos[:7], 4), ' right: ', np.round(qpos[7:], 4))
        rate.sleep()


def main(args):
    rospy.init_node('replay_node')

    img_front_publisher = rospy.Publisher(
        args.img_front_topic, Image, queue_size=10)
    img_left_publisher = rospy.Publisher(
        args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(
        args.img_right_topic, Image, queue_size=10)

    master_arm_left_publisher = rospy.Publisher(
        args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(
        args.master_arm_right_topic, JointState, queue_size=10)
    puppet_arm_left_publisher = rospy.Publisher(
        args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(
        args.puppet_arm_right_topic, JointState, queue_size=10)
    robot_base_publisher = rospy.Publisher(
        args.robot_base_topic, Twist, queue_size=10)

    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    dataset_name = f'episode_{args.episode_idx}'
    qposs, _, _, actions, base_actions, image_dicts, _ = load_hdf5(
        dataset_dir, dataset_name)

    if args.only_pub_master:
        _publish_master_only(
            actions,
            (master_arm_left_publisher, master_arm_right_publisher),
        )
        return

    _publish_full_replay(
        args,
        qposs,
        actions,
        base_actions,
        image_dicts,
        (
            (img_front_publisher, img_left_publisher, img_right_publisher),
            (master_arm_left_publisher, master_arm_right_publisher),
            (puppet_arm_left_publisher, puppet_arm_right_publisher),
            robot_base_publisher,
        ),
    )


if __name__ == '__main__':
    main(build_parser().parse_args())
