#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
import math
import random
import os
from time import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Int16

"""
Constant Definition
"""

WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class TreeNode:
    """
    Class for tree node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.cost = 0.0


class RRT:
    """
    Class for RRT / RRT*
    """

    def __init__(self,
                 start,
                 goal,
                 occupancy_grid,
                 collision_tol=0.5,
                 expand_dis=0.5,
                 path_resolution=0.05,
                 goal_sample_rate=5,
                 max_iter=500,
                 circle_dist=0.5,
                 early_stop=True,
                 smooth=False,
                 smoother_type="dp",
                 smoother_iter=100
                 ):
        """
        Setting Parameter

        Args:
            start (numpy.ndarray or (x, y)): Start Position [x,y]
            goal (numpy.ndarray or (x, y)): Goal Position [x,y]
            occupancy_grid (numpy.ndarray): occupancy grid
        """
        self.start = TreeNode(start[0], start[1])
        self.end = TreeNode(goal[0], goal[1])

        self.occupancy_grid = occupancy_grid
        self.grid_v = occupancy_grid[:, :, 0].flatten()
        self.grid_x = occupancy_grid[:, :, 1].flatten()
        self.grid_y = occupancy_grid[:, :, 2].flatten()
        occupied_x = self.grid_x[self.grid_v == 1.0]
        occupied_y = self.grid_y[self.grid_v == 1.0]
        self.occupied_pos = np.vstack((occupied_x, occupied_y))

        self.xmin = np.min(occupancy_grid[:, :, 1])
        self.xmax = np.max(occupancy_grid[:, :, 1])
        self.ymin = np.min(occupancy_grid[:, :, 2])
        self.ymax = np.max(occupancy_grid[:, :, 2])

        self.collision_tol = collision_tol
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.circle_dist = circle_dist
        self.early_stop = early_stop

        self.smooth = smooth
        self.smoother_type = smoother_type
        self.smoother_iter = smoother_iter

        self.tree = []

    def planning(self):
        """
        This method is the main RRT loop

        Args:
        Returns:
            path (list of [x, y]): final course

        """
        self.tree = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.sample()
            nearest_node = self.nearest(self.tree, rnd_node)

            new_node = self.steer(nearest_node, rnd_node, expand_dis=self.expand_dis)
            new_node.cost = nearest_node.cost + self.line_cost(nearest_node, new_node)

            if self.check_collision(new_node):
                continue

            neighborhood = self.near(self.tree, new_node)
            updated_new_node = self.choose_parent(new_node, neighborhood)

            if updated_new_node:
                self.rewire(updated_new_node, neighborhood)
                self.tree.append(updated_new_node)
            else:
                self.tree.append(new_node)

            if self.early_stop:  # if reaches goal
                last_idx = self.search_best_goal_node()
                if last_idx is not None:
                    return self.find_path(self.tree[last_idx], smooth=self.smooth)

        last_idx = self.search_best_goal_node()
        if last_idx is not None:
            return self.find_path(self.tree[last_idx], smooth=self.smooth)

        return None, None  # cannot find path

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            node (TreeNode): a tuple representing the sampled point

        """
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
        else:  # goal point sampling
            x = self.end.x
            y = self.end.y

        return TreeNode(x, y)

    @staticmethod
    def nearest(tree, sampled_node):
        """
        This method should return the nearest node on the tree to the sampled node

        Args:
            tree (list of TreeNode): the current RRT tree
            sampled_node (TreeNode): point sampled in free space
        Returns:
            nearest_node (TreeNode): the nearest node on the tree
        """
        dlist = [(node.x - sampled_node.x) ** 2 + (node.y - sampled_node.y) ** 2
                 for node in tree]
        min_idx = dlist.index(min(dlist))
        return tree[min_idx]

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def steer(self, nearest_node, sampled_node, expand_dis=float('inf')):
        """
        This method should return a point in the viable set such that it is closer
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (TreeNode): nearest node on the tree to the sampled point
            sampled_node (TreeNode): sampled point
            expand_dis (float): expand distance
        Returns:
            new_node (TreeNode): new node created from steering
        """
        new_node = TreeNode(nearest_node.x, nearest_node.y)
        d, theta = self.calc_distance_and_angle(new_node, sampled_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        expand_dis = min(expand_dis, d)

        n_expand = math.floor(expand_dis / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, sampled_node)
        if d <= self.path_resolution:
            new_node.path_x.append(sampled_node.x)
            new_node.path_y.append(sampled_node.y)
            new_node.x = sampled_node.x
            new_node.y = sampled_node.y

        new_node.parent = nearest_node

        return new_node

    def check_collision(self, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            new_node (TreeNode): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        if new_node is None:
            return False

        for pos in zip(new_node.path_x, new_node.path_y):
            if self.dist_to_grid(pos) < self.collision_tol:
                return True  # collision

        return False  # safe

    def dist_to_grid(self, pos):
        """
        Calculate distance to occupancy grid

        Args:
            pos (numpy.ndarray or (x, y)): current position
        Returns:
            dist (float): distance to occupancy grid

        """
        return np.min(np.linalg.norm(self.occupied_pos.T - np.array(pos), axis=-1))

    def is_goal(self, latest_added_node, goal_node):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (TreeNode): latest added node on the tree
            goal_node (TreeNode): goal node
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        dx = latest_added_node.x - goal_node.x
        dy = latest_added_node.y - goal_node.y
        if math.hypot(dx, dy) > self.expand_dis:
            return False  # not close to goal
        final_node = self.steer(latest_added_node, goal_node, expand_dis=self.expand_dis)
        if self.check_collision(final_node):
            return False  # close enough but has collision

        return True

    def find_path(self, latest_added_node, smooth=False):
        """
        This method returns a path as a list of TreeNodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            latest_added_node (TreeNode): latest added node in the tree
            smooth (bool): whether to use path smoother
        Returns:
            path ([]): valid path as a list of TreeNodes
            smooth_path ([]): smoothed path as a list of TreeNodes
        """
        path = []
        if latest_added_node.x != self.end.x or latest_added_node.y != self.end.y:
            path.append([self.end.x, self.end.y])

        node = latest_added_node
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path.reverse()

        if not smooth:
            smooth_path = None
        elif self.smoother_type == "rand":
            smooth_path = self.path_smoother_rand(path, max_iter=self.smoother_iter)
        elif self.smoother_type == "dp":
            smooth_path = self.path_smoother_dp(path)
        else:
            raise ValueError("Invalid smoother type")

        return path, smooth_path

    # The following methods are needed for RRT* and not RRT
    def choose_parent(self, new_node, neighborhood):
        """
        This method should compute the cheapest point to new_node contained in the list
        and set such a node as the parent of new_node.

        Args:
            new_node (TreeNode): current node
            neighborhood (list of int):  indices of neighborhood of nodes
        Returns:
            new_node (TreeNode): updated new_node
        """
        if not neighborhood:
            return None

        # Search the nearest cost in neighborhood
        costs = []
        for i in neighborhood:
            neighbor = self.tree[i]
            t_node = self.steer(neighbor, new_node, expand_dis=self.expand_dis)
            if t_node and not self.check_collision(t_node):
                costs.append(self.cost(neighbor, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node

        min_cost = min(costs)
        if min_cost == float("inf"):
            return None

        min_idx = neighborhood[costs.index(min_cost)]
        new_node = self.steer(self.tree[min_idx], new_node, expand_dis=self.expand_dis)
        new_node.cost = min_cost

        return new_node

    def rewire(self, new_node, neighborhood):
        """
        This method should check for each node in neighborhood, if it is cheaper to
        arrive to them from new_node. If so, re-assign the parent of the nodes in
        neighborhood to new_node

        Args:
            new_node (TreeNode): current node
            neighborhood (list of int):  indices of neighborhood of nodes
        Returns:

        """
        for i in neighborhood:
            neighbor = self.tree[i]
            edge_node = self.steer(new_node, neighbor, expand_dis=self.expand_dis)
            if not edge_node:
                continue
            edge_node.cost = self.cost(new_node, neighbor)

            collision = self.check_collision(edge_node)
            improved_cost = neighbor.cost > edge_node.cost

            if not collision and improved_cost:
                self.tree[i].x = edge_node.x
                self.tree[i].y = edge_node.y
                self.tree[i].cost = edge_node.cost
                self.tree[i].path_x = edge_node.path_x
                self.tree[i].path_y = edge_node.path_y
                self.tree[i].parent = edge_node.parent
        self.propagate_cost_to_leaves(new_node)

    def search_best_goal_node(self):
        """
        This method should search the cheapest point to reach goal

        Args:
        Returns:
            best_goal_node (int): index of the cheapest node that is close enough to goal

        """
        goal_indices = []
        for idx, node in enumerate(self.tree):
            if not self.is_goal(node, self.end):
                continue
            goal_indices.append(idx)

        if not goal_indices:
            return None

        min_cost = min([self.tree[i].cost for i in goal_indices])
        for i in goal_indices:
            if self.tree[i].cost == min_cost:
                return i

        return None

    def cost(self, from_node, to_node):
        """
        This method should return the cost of a node

        Args:
            from_node (TreeNode): node at one end of the straight line
            to_node (TreeNode): node at the other end of the straight line
        Returns:
            cost (float): the cost value of the node
        """

        return from_node.cost + self.line_cost(from_node, to_node)

    @staticmethod
    def line_cost(n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (TreeNode): node at one end of the straight line
            n2 (TreeNode): node at the other end of the straight line
        Returns:
            cost (float): the cost value of the line
        """

        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def propagate_cost_to_leaves(self, parent_node):
        """
        This method should recursively propagate the cost from parent node to leaf node

        Args:
            parent_node (TreeNode): dfs root
        Returns:

        """
        for node in self.tree:
            if node.parent == parent_node:
                node.cost = self.cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def near(self, tree, new_node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree (list of TreeNode): current tree as a list of TreeNodes
            new_node (TreeNode): current node we're finding neighbors for
        Returns:
            neighborhood (list of int): indices of neighborhood of nodes
        """
        n_node = len(tree) + 1
        r = self.circle_dist * math.sqrt((math.log(n_node) / n_node))
        r = min(r, self.expand_dis)

        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in tree]
        neighborhood = [dlist.index(i) for i in dlist if i <= r ** 2]

        return neighborhood

    @staticmethod
    def get_path_length(path):
        """
        This method should return the length of the whole path

        Args:
            path (list of [x, y]): path as 2d list
        Returns:
            length (float): length of the whole path
        """
        length = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.hypot(dx, dy)
            length += d

        return length

    @staticmethod
    def get_target_point(path, target_length):
        """
        This method should return target point based on its distance to path's start point

        Args:
            path (list of [x, y]): path as 2d list
            target_length (float): sampled point's distance to start point
        Returns:
            target_point ([x, y, idx]): target point
        """
        length = 0
        idx = 0
        curr_length = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            length += d
            if length >= target_length:
                idx = i - 1
                curr_length = d
                break

        partRatio = (length - target_length) / curr_length

        x = path[idx][0] + (path[idx + 1][0] - path[idx][0]) * partRatio
        y = path[idx][1] + (path[idx + 1][1] - path[idx][1]) * partRatio

        return [x, y, idx]

    def path_smoother_rand(self, path, max_iter=100):
        """
        Remove redundant waypoints to smooth path using random sampling

        Args:
            path (list of [x, y]): rrt path as a list of TreeNodes
            max_iter (int): maximum iterations
        Returns:
            smooth_path (list of [x, y]): smoothed path as a list of TreeNodes

        """
        path_length = self.get_path_length(path)

        for i in range(max_iter):
            # Sample two points
            pickPoints = [random.uniform(0, path_length), random.uniform(0, path_length)]
            pickPoints.sort()
            first = self.get_target_point(path, pickPoints[0])
            second = self.get_target_point(path, pickPoints[1])

            # Check valid index
            if first[2] <= 0 or second[2] <= 0:
                continue
            if (second[2] + 1) > len(path):
                continue
            if second[2] == first[2]:
                continue

            # Check no collision
            node_1 = TreeNode(first[0], first[1])
            node_2 = TreeNode(second[0], second[1])
            new_node = self.steer(node_1, node_2)
            if self.check_collision(new_node):
                continue

            # Create New path
            newPath = []
            newPath.extend(path[:first[2] + 1])
            newPath.append([first[0], first[1]])
            newPath.append([second[0], second[1]])
            newPath.extend(path[second[2] + 1:])
            path = newPath
            path_length = self.get_path_length(path)

        return path

    def path_smoother_dp(self, path):
        """
        Remove redundant waypoints to smooth path using dynamic programming

        Args:
            path (list of [x, y]): rrt path as a list of TreeNodes
        Returns:
            smooth_path (list of [x, y]): smoothed path as a list of TreeNodes

        """
        n = len(path)

        # Construct dynamic programming table with size (n,)
        # dp[i] represents minimum cost from node 0 to node i

        # Initialization:
        #     dp[i] = 0   for i == 0
        #     dp[i] = +∞  for i != 0
        dp = np.full((n,), np.inf)
        dp[0] = 0

        # Also keep track of each node's parent
        # Initialize parent[i + 1] = i
        parents = {}
        for i in range(n - 1):
            parents[i + 1] = i

        # Iterate node index to fill dp table
        node_0 = TreeNode(path[0][0], path[0][1])
        for i in range(1, n):
            node_i = TreeNode(path[i][0], path[i][1])

            # If two nodes can be connected directly
            new_node = self.steer(node_0, node_i, expand_dis=float('inf'))
            if not self.check_collision(new_node):
                dp[i] = self.line_cost(node_0, node_i)  # cost equal to nodes distance
                parents[i] = 0  # update node i parent
                continue

            # If two nodes cannot be connected directly,
            # then dp[i] = min{dp[j] + d(j, i)} for 0 < j < i
            for j in range(1, i):
                node_j = TreeNode(path[j][0], path[j][1])
                new_node = self.steer(node_j, node_i, expand_dis=float('inf'))
                if self.check_collision(new_node):
                    continue  # cannot connect j and i directly
                cost = dp[j] + self.line_cost(node_j, node_i)
                if cost >= dp[i]:
                    continue
                dp[i] = cost
                parents[i] = j

        # Back track final path
        smooth_path = []
        node_idx = n - 1
        while node_idx != 0:
            smooth_path.append(path[node_idx])
            node_idx = parents[node_idx]
        smooth_path.append(path[0])
        smooth_path.reverse()

        return smooth_path


class DynamicPlanner(Node):
    """
    Class for planner node
    """

    def __init__(self):
        super().__init__('dynamic_planner_node')

        # ROS Params
        self.declare_parameter('visualize')
        
        self.declare_parameter('real_test')
        self.declare_parameter('car_map')

        self.declare_parameter('lookahead_distance')
        self.declare_parameter('lookahead_attenuation')
        self.declare_parameter('lookahead_idx')
        self.declare_parameter('lookbehind_idx')

        self.declare_parameter('kp')
        self.declare_parameter('ki')
        self.declare_parameter('kd')
        self.declare_parameter("max_control")
        self.declare_parameter("steer_alpha")

        self.declare_parameter('grid_xmin')
        self.declare_parameter('grid_xmax')
        self.declare_parameter('grid_ymin')
        self.declare_parameter('grid_ymax')
        self.declare_parameter('grid_resolution')
        self.declare_parameter('plot_resolution')
        self.declare_parameter('grid_safe_dist')
        self.declare_parameter('goal_safe_dist')

        self.declare_parameter('use_rrt')
        self.declare_parameter('collision_tol')
        self.declare_parameter('expand_dis')
        self.declare_parameter('path_resolution')
        self.declare_parameter('goal_sample_rate')
        self.declare_parameter('max_iter')
        self.declare_parameter('circle_dist')
        self.declare_parameter('early_stop')
        self.declare_parameter('smooth')
        self.declare_parameter('smoother_type')
        self.declare_parameter('smoother_iter')

        self.declare_parameter('opp_racecar_speed')

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0

        # Global Map Params
        self.real_test = self.get_parameter('real_test').get_parameter_value().bool_value
        car_map = self.get_parameter('car_map').get_parameter_value().string_value
        csv_loc = os.path.join('src', 'lab7_pkg', 'csv', car_map + '.csv')

        waypoints = np.loadtxt(csv_loc, delimiter=',')
        self.num_pts = len(waypoints)
        self.waypoint_x = waypoints[:, 0]
        self.waypoint_y = waypoints[:, 1]
        self.waypoint_v = waypoints[:, 2]
        self.waypoint_yaw = waypoints[:, 3]
        self.waypoint_pos = waypoints[:, 0:2]
        self.v_max = np.max(self.waypoint_v)
        self.v_min = np.min(self.waypoint_v)

        # Local Map Params
        self.grid = None
        self.rrt_tree = None
        self.rrt_path = None
        self.smooth_path = None

        # Car State Params
        self.curr_global_pos, self.curr_global_yaw = None, None
        self.goal_local_pos = None
        self.goal_global_pos = None

        # Other Params
        self.frame_cnt = 0

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        scan_topic = "/scan"
        drive_topic = '/drive'

        opp_pose_topic = "/opp_racecar/odom"
        opp_drive_topic = '/opp_drive'

        grid_topic = '/grid'
        rrt_topic = '/rrt'
        smooth_topic = '/smooth'
        waypoint_topic = '/waypoint'
        path_topic = '/global_path'
        fps_topic = '/fps'

        self.timer = self.create_timer(1.0, self.timer_callback)

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
            self.opp_pose_sub_ = self.create_subscription(PoseStamped, opp_pose_topic, self.opp_pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
            self.opp_pose_sub_ = self.create_subscription(Odometry, opp_pose_topic, self.opp_pose_callback, 1)

        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.opp_drive_pub_ = self.create_publisher(AckermannDriveStamped, opp_drive_topic, 10)
        self.grid_pub_ = self.create_publisher(MarkerArray, grid_topic, 10)
        self.rrt_pub_ = self.create_publisher(Marker, rrt_topic, 10)
        self.smooth_pub_ = self.create_publisher(Marker, smooth_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.path_pub_ = self.create_publisher(Marker, path_topic, 10)
        self.fps_pub_ = self.create_publisher(Int16, fps_topic, 10)

    def timer_callback(self):
        fps = Int16()
        fps.data = self.frame_cnt
        self.frame_cnt = 0
        self.fps_pub_.publish(fps)
        self.get_logger().info('fps: %d' % fps.data)

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args:
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        ranges = np.array(scan_msg.ranges)
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)

        xmin = self.get_parameter('grid_xmin').get_parameter_value().double_value
        xmax = self.get_parameter('grid_xmax').get_parameter_value().double_value
        ymin = self.get_parameter('grid_ymin').get_parameter_value().double_value
        ymax = self.get_parameter('grid_ymax').get_parameter_value().double_value
        resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        grid_safe_dist = self.get_parameter('grid_safe_dist').get_parameter_value().double_value

        nx = int((xmax - xmin) / resolution) + 1
        ny = int((ymax - ymin) / resolution) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        y, x = np.meshgrid(y, x)
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        ray_idx = ((phi - scan_msg.angle_min) / scan_msg.angle_increment).astype(int)
        obs_rho = ranges[ray_idx]

        self.grid = np.where(np.abs(rho - obs_rho) < grid_safe_dist, 1.0, 0.0)
        self.grid = np.dstack((self.grid, x, y))  # (h, w, 3)

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args:
            pose_msg (PoseStamped or Odometry): incoming message from subscribed topic
        Returns:

        """
        # Get speed, steer, car position and yaw
        use_rrt = self.get_parameter('use_rrt').get_parameter_value().bool_value

        res = self.get_control(pose_msg, use_rrt=use_rrt)

        if not res:
            return

        speed = res[0]
        steer = res[1]
        self.curr_global_pos = res[2]
        self.curr_global_yaw = res[3]
        self.goal_global_pos = res[4]
        self.goal_local_pos = res[5]

        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)

        # Visualize waypoint
        visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        if visualize:
            self.visualize_occupancy_grid()
            self.visualize_rrt()
            self.visualize_smooth_path()
            self.visualize_waypoints()

        # Increase frame count
        self.frame_cnt += 1

        return None

    def opp_pose_callback(self, pose_msg):
        """
        The opponent pose callback

        Args:
            pose_msg (PoseStamped or Odometry): incoming message from subscribed topic
        Returns:

        """
        # Get speed, steer, car position and yaw
        res = self.get_control(pose_msg, use_rrt=False)

        if not res:
            return

        speed = self.get_parameter('opp_racecar_speed').get_parameter_value().double_value
        steer = res[1]

        # Publish drive message
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.opp_drive_pub_.publish(message)

    def get_control(self, pose_msg, use_rrt=False):
        """
        This method should calculate the desired speed and steering angle and other status

        Args:
            pose_msg (PoseStamped or Odometry): incoming message from subscribed topic
            use_rrt (bool): whether to use RRT local planner
        Returns:
            speed (float): car target speed
            steer (float): car target steering angle
            curr_global_pos (numpy.ndarray): car current position in map frame
            curr_global_yaw (float): car current yaw angle in map frame
            goal_global_pos (numpy.ndarray): car target position in map frame
            goal_local_pos (numpy.ndarray): car target position in car frame

        """
        # Read pose data
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_quat = pose_msg.pose.pose.orientation

        curr_global_pos = np.array([curr_x, curr_y])
        curr_global_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                                     1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        # Wait until laser scan available
        if self.grid is None:
            return

        # Find index of the current point
        distances = np.linalg.norm(self.waypoint_pos - curr_global_pos, axis=1)
        curr_idx = np.argmin(distances)

        # Get lookahead distance
        L = self.get_lookahead_dist(curr_idx)

        # Search safe global target waypoint
        goal_safe_dist = self.get_parameter('goal_safe_dist').get_parameter_value().double_value
        while True:
            # Binary search goal waypoint to track
            goal_idx = curr_idx
            while distances[goal_idx] <= L:
                goal_idx = (goal_idx + 1) % self.num_pts

            left = self.waypoint_pos[(goal_idx - 1) % self.num_pts, :]
            right = self.waypoint_pos[goal_idx % self.num_pts, :]

            while True:
                mid = (left + right) / 2
                dist = np.linalg.norm(mid - curr_global_pos)
                if abs(dist - L) < 1e-2:
                    goal_global_pos = mid
                    break
                elif dist > L:
                    right = mid
                else:
                    left = mid

            # Transform goal point to vehicle frame of reference
            R = np.array([[np.cos(curr_global_yaw), np.sin(curr_global_yaw)],
                          [-np.sin(curr_global_yaw), np.cos(curr_global_yaw)]])
            goal_local_pos = R @ np.array([goal_global_pos[0] - curr_global_pos[0],
                                           goal_global_pos[1] - curr_global_pos[1]])

            # Check if target point collision free
            if self.dist_to_grid(goal_local_pos) > goal_safe_dist:
                break
            L *= 1.1

        # Use RRT for local planning
        if use_rrt:
            self.rrt_tree, self.rrt_path, self.smooth_path = self.local_planning(goal_local_pos)

            if not self.rrt_path or len(self.rrt_path) == 2:
                y_error = goal_local_pos[1]
            else:
                y_error = self.smooth_path[1][1] if self.smooth_path else self.rrt_path[1][1]

        else:
            y_error = goal_local_pos[1]

        # Get desired speed and steering angle
        speed = self.waypoint_v[curr_idx % self.num_pts]
        gamma = 2 / L ** 2
        error = gamma * y_error
        steer = self.get_steer(error)

        return speed, steer, curr_global_pos, curr_global_yaw, goal_global_pos, goal_local_pos

    def get_lookahead_dist(self, curr_idx):
        """
        This method should calculate the lookahead distance based on past and future waypoints

        Args:
            curr_idx (ndarray[int]): closest waypoint index
        Returns:
            lookahead_dist (float): lookahead distance

        """
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_idx = self.get_parameter('lookahead_idx').get_parameter_value().integer_value
        lookbehind_idx = self.get_parameter('lookbehind_idx').get_parameter_value().integer_value
        slope = self.get_parameter('lookahead_attenuation').get_parameter_value().double_value

        yaw_before = self.waypoint_yaw[(curr_idx - lookbehind_idx) % self.num_pts]
        yaw_after = self.waypoint_yaw[(curr_idx + lookahead_idx) % self.num_pts]
        yaw_diff = abs(yaw_after - yaw_before)
        if yaw_diff > np.pi:
            yaw_diff = yaw_diff - 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff = yaw_diff + 2 * np.pi
        yaw_diff = abs(yaw_diff)
        if yaw_diff > np.pi / 2:
            yaw_diff = np.pi / 2
        L = max(0.5, L * (np.pi / 2 - yaw_diff * slope) / (np.pi / 2))

        return L

    def dist_to_grid(self, pos):
        """
        Calculate distance to occupancy grid

        Args:
            pos (numpy.ndarray or (x, y)): current position
        Returns:
            dist (float): distance to occupancy grid

        """
        grid_v = self.grid[:, :, 0].flatten()
        grid_x = self.grid[:, :, 1].flatten()
        grid_y = self.grid[:, :, 2].flatten()

        grid_x = grid_x[grid_v == 1.0]
        grid_y = grid_y[grid_v == 1.0]
        grid_pos = np.vstack((grid_x, grid_y))

        dist = np.min(np.linalg.norm(grid_pos.T - pos, axis=-1))

        return dist

    def get_steer(self, error):
        """
        Get desired steering angle by PID

        Args:
            error (float): current error
        Returns:
            new_steer (float): desired steering angle

        """
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error
        steer = kp * error + ki * self.integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer

    def local_planning(self, goal_pos):
        """
        Get desired steering angle by PID

        Args:
            goal_pos (numpy.ndarray or (x, y)): goal position in car local frame
        Returns:
            path (list of [x, y]): local path found by rrt
            tree (list of TreeNode): searched tree in rrt

        """
        collision_tol = self.get_parameter('collision_tol').get_parameter_value().double_value
        expand_dis = self.get_parameter('expand_dis').get_parameter_value().double_value
        path_resolution = self.get_parameter('path_resolution').get_parameter_value().double_value
        goal_sample_rate = self.get_parameter('goal_sample_rate').get_parameter_value().double_value
        max_iter = self.get_parameter('max_iter').get_parameter_value().integer_value
        circle_dist = self.get_parameter('circle_dist').get_parameter_value().integer_value
        early_stop = self.get_parameter('early_stop').get_parameter_value().bool_value
        smooth = self.get_parameter('smooth').get_parameter_value().bool_value
        smoother_type = self.get_parameter('smoother_type').get_parameter_value().string_value
        smoother_iter = self.get_parameter('smoother_iter').get_parameter_value().integer_value

        rrt = RRT(start=(0, 0),
                  goal=goal_pos,
                  occupancy_grid=self.grid,
                  collision_tol=collision_tol,
                  expand_dis=expand_dis,
                  path_resolution=path_resolution,
                  goal_sample_rate=goal_sample_rate,
                  max_iter=max_iter,
                  circle_dist=circle_dist,
                  early_stop=early_stop,
                  smooth=smooth,
                  smoother_type=smoother_type,
                  smoother_iter=smoother_iter)
        path, smooth_path = rrt.planning()

        return rrt.tree, path, smooth_path

    def visualize_occupancy_grid(self):
        if self.grid is None:
            return

        grid_resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        plot_resolution = self.get_parameter('plot_resolution').get_parameter_value().double_value
        down_sample = max(1, int(plot_resolution / grid_resolution))

        grid = self.grid.copy()
        grid = grid[::down_sample, ::down_sample, :]  # down sample for faster plotting

        grid_v = grid[:, :, 0].flatten()
        grid_x = grid[:, :, 1].flatten()
        grid_y = grid[:, :, 2].flatten()

        # Transform occupancy grid into map frame
        pos = np.vstack((grid_x.flatten(), grid_y.flatten()))
        R = np.array([[np.cos(self.curr_global_yaw), -np.sin(self.curr_global_yaw)],
                      [np.sin(self.curr_global_yaw), np.cos(self.curr_global_yaw)]])
        grid_x, grid_y = R @ pos + self.curr_global_pos.reshape(-1, 1)

        # Publish occupancy grid
        marker_arr = MarkerArray()

        for i in range(len(grid_v)):
            if grid_v[i] == 0:
                continue

            marker = Marker()
            marker.header.frame_id = '/map'
            marker.id = i
            marker.ns = 'occupancy_grid_%u' % i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = grid_x[i]
            marker.pose.position.y = grid_y[i]

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            marker.lifetime.nanosec = int(1e8)

            marker_arr.markers.append(marker)

        self.grid_pub_.publish(marker_arr)

    def visualize_rrt(self):
        if not self.rrt_tree:
            return

        # Rotation matrix from car frame to map frame
        R = np.array([[np.cos(self.curr_global_yaw), -np.sin(self.curr_global_yaw)],
                      [np.sin(self.curr_global_yaw), np.cos(self.curr_global_yaw)]])

        # Publish rrt tree and path
        line_list = Marker()
        line_list.header.frame_id = '/map'
        line_list.id = 0
        line_list.ns = 'rrt'
        line_list.type = Marker.LINE_LIST
        line_list.action = Marker.ADD

        line_list.scale.x = 0.1
        line_list.scale.y = 0.1
        line_list.scale.z = 0.1

        line_list.points = []

        for node in self.rrt_tree:
            if node.parent is None:
                continue

            if not self.rrt_path:
                color = (0.0, 0.0, 1.0)
            else:
                dist = np.linalg.norm(np.array(self.rrt_path) - np.array([node.x, node.y]), axis=-1)
                on_path = np.any(dist == 0.0)
                color = (0.0, 1.0, 0.0) if on_path else (0.0, 0.0, 1.0)

            # Add first point
            this_point = Point()

            local_pos = np.array([node.parent.x, node.parent.y], dtype=float)
            global_pos = R @ local_pos + self.curr_global_pos
            this_point.x = global_pos[0]
            this_point.y = global_pos[1]
            line_list.points.append(this_point)

            this_color = ColorRGBA()
            this_color.r = color[0]
            this_color.g = color[1]
            this_color.b = color[2]
            this_color.a = 1.0
            line_list.colors.append(this_color)

            # Add second point
            this_point = Point()

            local_pos = np.array([node.x, node.y], dtype=float)
            global_pos = R @ local_pos + self.curr_global_pos
            this_point.x = global_pos[0]
            this_point.y = global_pos[1]
            line_list.points.append(this_point)

            this_color = ColorRGBA()
            this_color.r = color[0]
            this_color.g = color[1]
            this_color.b = color[2]
            this_color.a = 1.0
            line_list.colors.append(this_color)

        self.rrt_pub_.publish(line_list)

    def visualize_smooth_path(self):
        if not self.smooth_path:
            return

        # Rotation matrix from car frame to map frame
        R = np.array([[np.cos(self.curr_global_yaw), -np.sin(self.curr_global_yaw)],
                      [np.sin(self.curr_global_yaw), np.cos(self.curr_global_yaw)]])

        # Publish rrt tree and path
        line_list = Marker()
        line_list.header.frame_id = '/map'
        line_list.id = 0
        line_list.ns = 'smooth_path'
        line_list.type = Marker.LINE_LIST
        line_list.action = Marker.ADD

        line_list.scale.x = 0.1
        line_list.scale.y = 0.1
        line_list.scale.z = 0.1

        line_list.points = []

        color = (255.0, 105.0, 180.0)

        for idx in range(len(self.smooth_path) - 1):
            # Add first point
            this_point = Point()

            local_pos = np.array(self.smooth_path[idx], dtype=float)
            global_pos = R @ local_pos + self.curr_global_pos
            this_point.x = global_pos[0]
            this_point.y = global_pos[1]
            line_list.points.append(this_point)

            this_color = ColorRGBA()
            this_color.r = color[0] / 255.0
            this_color.g = color[1] / 255.0
            this_color.b = color[2] / 255.0
            this_color.a = 1.0
            line_list.colors.append(this_color)

            # Add second point
            this_point = Point()

            local_pos = np.array(self.smooth_path[idx + 1], dtype=float)
            global_pos = R @ local_pos + self.curr_global_pos
            this_point.x = global_pos[0]
            this_point.y = global_pos[1]
            line_list.points.append(this_point)

            this_color = ColorRGBA()
            this_color.r = color[0] / 255.0
            this_color.g = color[1] / 255.0
            this_color.b = color[2] / 255.0
            this_color.a = 1.0
            line_list.colors.append(this_color)

        self.smooth_pub_.publish(line_list)

    def visualize_waypoints(self):
        # Publish all waypoints
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'global_planner'
        marker.type = 4
        marker.action = 0
        marker.points = []
        marker.colors = []
        for i in range(self.num_pts + 1):
            this_point = Point()
            this_point.x = self.waypoint_x[i % self.num_pts]
            this_point.y = self.waypoint_y[i % self.num_pts]
            marker.points.append(this_point)

            this_color = ColorRGBA()
            speed_ratio = (self.waypoint_v[i % self.num_pts] - self.v_min) / (self.v_max - self.v_min)
            this_color.a = 1.0
            this_color.r = (1 - speed_ratio)
            this_color.g = speed_ratio
            marker.colors.append(this_color)

        this_scale = 0.1
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.path_pub_.publish(marker)

        # Publish target waypoint
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'target_waypoint'
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.goal_global_pos[0]
        marker.pose.position.y = self.goal_global_pos[1]

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.waypoint_pub_.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("Motion Planner Initialized")
    planner_node = DynamicPlanner()
    rclpy.spin(planner_node)

    planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
