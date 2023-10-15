#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped


class TalkerPY(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        timer_period = 0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.declare_parameter('v', 0.0)
        self.declare_parameter('d', 0.0)

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(self.get_parameter('v').get_parameter_value().double_value)
        msg.drive.steering_angle = float(self.get_parameter('d').get_parameter_value().double_value)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: speed = %f, steering angle = %f' % (msg.drive.speed,
                                                                                msg.drive.steering_angle))


def main(args=None):
    rclpy.init(args=args)
    talker_node = TalkerPY()
    rclpy.spin(talker_node)
    talker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
