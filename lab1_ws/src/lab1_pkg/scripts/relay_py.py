#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped


class RelayPY(Node):
    def __init__(self):
        super().__init__('relay')
        self.subscription = self.create_subscription(AckermannDriveStamped, 'drive', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive_relay', 10)

    def listener_callback(self, msg):
        self.get_logger().info('I heard: speed = %f, steering angle = %f' % (msg.drive.speed,
                                                                             msg.drive.steering_angle))

        relay_msg = AckermannDriveStamped()
        relay_msg.drive.speed = msg.drive.speed * 3
        relay_msg.drive.steering_angle = msg.drive.steering_angle * 3
        self.get_logger().info('Relay: speed = %f, steering angle = %f' % (relay_msg.drive.speed,
                                                                           relay_msg.drive.steering_angle))
        self.publisher_.publish(relay_msg)


def main(args=None):
    rclpy.init(args=args)
    relay_node = RelayPY()
    rclpy.spin(relay_node)
    relay_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
