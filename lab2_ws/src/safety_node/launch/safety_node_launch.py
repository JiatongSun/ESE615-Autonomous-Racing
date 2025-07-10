#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    talker_node = Node(
        package="safety_node",
        executable="safety_node",
        name="safety_node",
        output="screen",
        parameters=[
            {"ttc_tol": 1.5},
        ]
    )
    ld.add_action(talker_node)

    return ld
