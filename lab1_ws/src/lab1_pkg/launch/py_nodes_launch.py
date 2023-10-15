#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    talker_node = Node(
        package="lab1_pkg",
        executable="talker_py.py",
        name="talker",
        output="screen",
        parameters=[
            {"v": 2.2},
            {"d": 0.8}
        ]
    )
    relay_node = Node(
        package="lab1_pkg",
        executable="relay_py.py",
        name="relay",
        output="screen",
    )
    ld.add_action(talker_node)
    ld.add_action(relay_node)

    return ld
