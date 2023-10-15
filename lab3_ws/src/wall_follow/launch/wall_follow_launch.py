#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    wall_follow_node = Node(
        package="wall_follow",
        executable="wall_follow_node",
        name="wall_follow_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"wall": "left"},  # which wall to follow, ["left", "right"]
            {"theta_min": 10},
            {"theta_max": 70},
            {"theta_step": 5},
            {"steer_min": -45},  # min control / min steering angle
            {"steer_max": 45},  # max control / max steering angle
            {"filter": "std"},  # "std" or "iqr"
            {"L": 1.0},
            {"dist": 1.3},  # desired distance
            {"err_low_tol": 0.05},  # step-like error low tolerance
            {"err_high_tol": 0.10},  # step-like error high tolerance
            {"low_vel": 1.0},  # step-like low velocity
            {"mid_vel": 2.0},  # step-like mid velocity
            {"high_vel": 5.0},  # step-like high velocity
            {"kp": 2.0},
            {"kd": 1.0},
            {"ki": 0.002},
        ]
    )
    ld.add_action(wall_follow_node)

    return ld
