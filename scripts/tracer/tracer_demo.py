import rospy
from geometry_msgs.msg import Twist

def move_base(pub, linear_x=0.0, angular_z=0.0, duration=1.0):
    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z

    rate = rospy.Rate(10) # 10Hz 控制频率
    start_time = rospy.Time.now()

    rospy.loginfo(f"Sending command: v_x={linear_x} m/s, w_z={angular_z} rad/s for {duration}s")

    # 持续发送控制指令：线速度和角速度
    while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
        pub.publish(twist) # 将twist消息发送到/cmd_vel话题
        rate.sleep()
        
    # 动作结束后，发送全零速度让底盘刹车
    pub.publish(Twist())
    rospy.loginfo("Movement complete, stopping.")

if __name__ == '__main__':
    try:
        #  初始化节点
        rospy.init_node('tracer_demo_node', anonymous=True)
        
        # 创建发布器
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        #确保发布器和 ROS Master 建立连接
  
        rospy.sleep(1.0) 

        #  执行动作序列
        

        move_base(pub, linear_x=-0.2, duration=1.0) # 后退
        rospy.sleep(1.0)                          
        move_base(pub, angular_z=0.2, duration=9.0) # 原地左转
        rospy.sleep(1.0) 
        move_base(pub, linear_x=0.2, duration=1.5)  # 前进，这个还可以大一点点
        rospy.sleep(1.0) 
        move_base(pub, angular_z=-0.2, duration=9.0) # 原地右转
        rospy.sleep(1.0) 
        move_base(pub, linear_x=0.1, duration=2.0) # 前进
        rospy.sleep(1.0) 

        #move_base(pub, linear_x=0.2, duration=2.0)  # 前进
        #rospy.sleep(0.5)                            # 动作之间停顿一下，保护电机
        # move_base(pub, angular_z=0.2, duration=10.0) 
        # rospy.sleep(1.0)
        # move_base(pub, linear_x=0.1, duration=2.0)  # 前进

        #move_base(pub, angular_z=0.5, duration=2.0) # 原地左转
        #rospy.sleep(0.5)
        
        #move_base(pub, angular_z=-0.5, duration=2.0)# 原地右转

    except rospy.ROSInterruptException:
        pass



