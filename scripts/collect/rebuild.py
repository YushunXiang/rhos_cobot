#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import argparse
import sys

bridge = CvBridge()
background = None
alpha = 0.5  # 默认混合比例
background_path = ""

def image_callback(msg):
    global background

    # 转换 ROS 图像为 OpenCV 格式
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # 仅第一次加载背景图
    if background is None:
        if not os.path.exists(background_path):
            rospy.logerr(f"背景图片不存在：{background_path}")
            rospy.signal_shutdown("无效背景路径")
            return
        bg = cv2.imread(background_path)
        if bg is None:
            rospy.logerr("无法读取背景图片")
            rospy.signal_shutdown("背景图片加载失败")
            return
        # 调整背景尺寸
        background = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

    # 图像混合
    blended = cv2.addWeighted(frame, alpha, background, 1 - alpha, 0)

    # 显示
    cv2.imshow("Camera + Background Blend", blended)
    key = cv2.waitKey(1)
    if key == 27:  # ESC退出
        rospy.signal_shutdown("用户中断")

def main():
    global alpha, background_path

    # ---------- 参数解析 ----------
    parser = argparse.ArgumentParser(description="Blend ROS camera image with background image.")
    parser.add_argument("--alpha", type=float, default=0.5, help="混合比例 (0~1)，数值越大前景越清晰")
    parser.add_argument("--image", type=str, required=True, help="背景图片路径")
    args, unknown = parser.parse_known_args()  # 兼容ROS参数系统
    alpha = args.alpha
    background_path = args.image

    # ---------- 启动ROS节点 ----------
    rospy.init_node("camera_image_blender", anonymous=True)
    rospy.Subscriber("/camera_f/color/image_raw", Image, image_callback)
    rospy.loginfo(f"开始接收 /camera_f/color/image_raw 并混合显示，alpha={alpha}, background='{background_path}'")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
