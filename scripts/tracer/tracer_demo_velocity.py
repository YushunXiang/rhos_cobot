import argparse

import rospy
from geometry_msgs.msg import Twist


DEFAULT_MAX_LINEAR_ABS = 0.3
DEFAULT_MAX_ANGULAR_ABS = 0.5

def move_base(pub, linear_x=0.0, angular_z=0.0, duration=1.0):
    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z

    rate = rospy.Rate(10) # 10Hz 控制频率
    start_time = rospy.Time.now()

    rospy.loginfo(f"Sending command: v_x={linear_x} m/s, w_z={angular_z} rad/s for {duration}s")

    # 持续发送控制指令
    while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
        pub.publish(twist)
        rate.sleep()
        
    # 动作结束后，发送全零速度让底盘刹车
    pub.publish(Twist())
    rospy.loginfo("Movement complete, stopping.")


def validate_args(args):
    """Validate CLI arguments for one-shot base motion command."""

    if args.duration <= 0:
        raise ValueError("duration 必须大于 0")
    if abs(args.linear_x) > args.max_linear_abs:
        raise ValueError(f"|linear_x| 不能超过 {args.max_linear_abs}")
    if abs(args.angular_z) > args.max_angular_abs:
        raise ValueError(f"|angular_z| 不能超过 {args.max_angular_abs}")

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="One-shot TRACER cmd_vel control")
        parser.add_argument("--linear-x", type=float, required=True, help="linear velocity in m/s")
        parser.add_argument("--angular-z", type=float, required=True, help="angular velocity in rad/s")
        parser.add_argument("--duration", type=float, required=True, help="command duration in seconds")
        parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="velocity command topic")
        parser.add_argument("--max-linear-abs", type=float, default=DEFAULT_MAX_LINEAR_ABS)
        parser.add_argument("--max-angular-abs", type=float, default=DEFAULT_MAX_ANGULAR_ABS)
        args = parser.parse_args()
        validate_args(args)

        #  初始化节点
        rospy.init_node('tracer_demo_node', anonymous=True)
        
        # 创建发布器
        pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        
        #确保发布器和 ROS Master 建立连接
  
        rospy.sleep(1.0) 

        move_base(
            pub,
            linear_x=args.linear_x,
            angular_z=args.angular_z,
            duration=args.duration,
        )
        pub.publish(Twist())

    except ValueError as exc:
        raise SystemExit(f"参数错误: {exc}") from exc
    except rospy.ROSInterruptException:
        pass