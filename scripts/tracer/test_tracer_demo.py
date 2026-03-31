import sys

# 搭建“虚拟 ROS 环境”（Mocking）

from unittest.mock import MagicMock
mock_rospy = MagicMock()
mock_rospy.is_shutdown.return_value = False

# 创造一个“虚拟时钟”
class VirtualTimeEnv:
    def __init__(self):
        self.current_time = 0.0
        
    def now(self):
        class TimeObj:
            def __init__(self, t): self.t = t
            def to_sec(self): return self.t
            def __sub__(self, other): return TimeObj(self.t - other.t)
        return TimeObj(self.current_time)
        
    def sleep(self, duration=0.1):
        # 每次调用 sleep 时，虚拟时间瞬间推进
        self.current_time += duration

v_env = VirtualTimeEnv()
mock_rospy.Time.now = v_env.now

class MockRate:
    def __init__(self, hz): self.hz = hz
    def sleep(self): v_env.sleep(1.0 / self.hz)

mock_rospy.Rate = MockRate

# 将虚拟的 rospy 强行注入到系统中
sys.modules['rospy'] = mock_rospy

# 导入我们要测试的真实代码
from geometry_msgs.msg import Twist
from tracer_demo import move_base


# 创建虚拟底盘接收器

class VirtualPublisher:
    def __init__(self):
        self.distance_x = 0.0
        self.dt = 0.1  # 你的代码里 rate 是 10Hz，每次循环耗时 0.1 秒
        
    def publish(self, msg):
        # 核心积分计算：位移 = 速度 * 时间
        self.distance_x += msg.linear.x * self.dt

# 执行测试用例
if __name__ == "__main__":
    print(" 脱机安全测试")
    
    # 实例化虚拟底盘
    pub = VirtualPublisher()
    
    print("正在执行")
    
    linear_x = -0.2
    duration = 4.0
    # 调用原始函数
    move_base(pub, linear_x=linear_x, duration=duration)
    
    # 提取总位移结果
    result = pub.distance_x
    exp_result = linear_x * duration
    # 格式化输出
    print(f"预期输出: {exp_result:.1f}")
    
    if result > 0:
        print(f"实际输出: 前进{result:.1f}")
    elif result < 0:
        print(f"实际输出: 后退{-result:.1f}")
    else:
        print(f"实际输出: 原地不动")
        
    print("-" * 35)
    if round(result, 1) == round(exp_result, 1):
        print("测试通过！你的底盘速度与时间积分逻辑完全正确。")
    else:
        print(" 测试失败！请检查 tracer_demo.py 中的逻辑。")