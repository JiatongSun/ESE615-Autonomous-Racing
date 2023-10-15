#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """

    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.0

        self.declare_parameter('ttc_tol', 1.5)
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def odom_callback(self, odom_msg: Odometry):
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg: LaserScan):
        n_ranges = len(scan_msg.ranges)
        ranges = np.array(scan_msg.ranges)
        in_bound = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        ranges = np.where(in_bound, ranges, np.inf)
        angles = np.arange(n_ranges) * scan_msg.angle_increment + scan_msg.angle_min
        vels = self.speed * np.cos(angles)
        vels = np.where(vels > 0, vels, 1e-8)
        ittc = min(ranges / vels)
        self.get_logger().info("iTTC: %f" % ittc)
        if ittc < self.get_parameter('ttc_tol').get_parameter_value().double_value:
            message = AckermannDriveStamped()
            message.drive.speed = 0.0
            self.publisher_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
