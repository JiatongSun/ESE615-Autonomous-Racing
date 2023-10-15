#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

from nav_msgs.msg import Odometry

import numpy as np

import os
import csv
import atexit
from time import gmtime, strftime

DATA_FOLDER = './data/'

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)


class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger_node')

        # Files
        self.folder = DATA_FOLDER
        self.file = open(strftime(self.folder + 'wp-%Y-%m-%d-%H-%M-%S', gmtime()) + '.csv', 'w')

        fieldnames = ['x', 'y', 'theta', 'v']
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

        # Topics & Subs, Pubs
        odom_topic = '/ego_racecar/odom'
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

    def odom_callback(self, data: Odometry):
        quaternion = np.array([data.pose.pose.orientation.x,
                               data.pose.pose.orientation.y,
                               data.pose.pose.orientation.z,
                               data.pose.pose.orientation.w])

        euler = euler_from_quaternion(quaternion)
        speed = np.linalg.norm(np.array([data.twist.twist.linear.x,
                                         data.twist.twist.linear.y,
                                         data.twist.twist.linear.z]), 2)

        self.writer.writerow({'x': data.pose.pose.position.x,
                              'y': data.pose.pose.position.y,
                              'theta': euler[2],
                              'v': speed})

    def shutdown(self):
        self.file.close()
        print('File saved!')


def main(args=None):
    rclpy.init(args=args)
    print('Saving waypoints...')
    waypoint_logger_node = WaypointLogger()
    atexit.register(waypoint_logger_node.shutdown)
    rclpy.spin(waypoint_logger_node)

    waypoint_logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
