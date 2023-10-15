#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


def generate_launch_description():
    ld = LaunchDescription()
    rrt_node = Node(
        package="lab7_pkg",
        executable="rrt_node",
        name="rrt_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            # RVIZ Params
            {"visualize": True},

            # Map Params
            {"real_test": False},  # True: real car; False: simulation
            {"car_map": "levine_sim"},  # ["levine_real", "levine_sim"]

            # Pure Pursuit Params
            {"lookahead_distance": 2.5},
            {"lookahead_attenuation": 0.6},
            {"lookahead_idx": 16},
            {"lookbehind_idx": 0},

            # PID Control Params
            {"kp": 1.0},
            {"ki": 0.0},
            {"kd": 0.0},
            {"max_control": MAX_STEER},
            {"steer_alpha": 1.0},

            # Occupancy Grid Params
            {"grid_xmin": 0.0},
            {"grid_xmax": 4.0},
            {"grid_ymin": -1.25},
            {"grid_ymax": 1.25},
            {"grid_resolution": 0.05},
            {"plot_resolution": 0.25},
            {"grid_safe_dist": 0.2},
            {"goal_safe_dist": 0.5},

            # RRT Params
            {"use_rrt": True},
            {"collision_tol": 0.2},
            {"expand_dis": 0.6},
            {"path_resolution": 0.05},
            {"goal_sample_rate": 7.0},
            {"max_iter": 100},
            {"circle_dist": 0.3},
            {"early_stop": True},
            {"smooth": True},
            {"smoother_type": "dp"},  # ["rand", "dp"]
            {"smoother_iter": 100},

            # Opponent Car Params
            {"opp_racecar_speed": 1.5},
        ]
    )
    ld.add_action(rrt_node)

    return ld
