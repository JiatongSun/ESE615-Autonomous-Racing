#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    talker_node = Node(
        package="lab1_pkg",
        executable="talker_cpp",
        name="talker",
        output="screen",
        parameters=[
            {"v": 1.0},
            {"d": 0.5}
        ]
    )
    relay_node = Node(
        package="lab1_pkg",
        executable="relay_cpp",
        name="relay",
        output="screen",
    )
    ld.add_action(talker_node)
    ld.add_action(relay_node)

    return ld
