#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

#  Constants from xacro
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


class ReactiveFollowGap(Node):
    """
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('reactive_node')

        # Params
        self.declare_parameter('window_size', 5)
        self.declare_parameter('max_horizon', 2.5)

        self.declare_parameter('disparity_thresh', 0.15)
        self.declare_parameter('bubble_radius', 0.9*(WIDTH + 2 * WHEEL_LENGTH))
        self.declare_parameter('corner_dist_thresh', 1.0)
        self.declare_parameter('turning_angle_thresh', 20.0 * np.pi / 180.0)

        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter("max_control", MAX_STEER)

        self.declare_parameter("low_vel", 0.5)
        self.declare_parameter("high_vel", 3.0)
        self.declare_parameter("velocity_attenuation", 1.5)

        self.declare_parameter("steer_alpha", 0.3)

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0

        self.prev_steer = 0.0

        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub_ = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def preprocess_lidar(self, ranges, range_min=0.0, range_max=np.inf):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (e.g. > 3m)
        """
        # Remove invalid readings
        proc_ranges = np.clip(ranges, range_min, range_max)

        # Clip high values
        max_horizon = self.get_parameter('max_horizon').get_parameter_value().double_value
        proc_ranges = np.clip(proc_ranges, 0, max_horizon)

        # Average over window
        window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        window = np.ones(window_size) / window_size
        proc_ranges = ndimage.convolve1d(proc_ranges, window, mode='nearest')

        return proc_ranges

    def find_disparities(self, ranges):
        """ Find disparity indices in the LiDAR readings
        """
        disparity_thresh = self.get_parameter('disparity_thresh').get_parameter_value().double_value

        # Calculate difference between index [i] and [i-1]
        # Prepend zero to align indices
        disparities = np.hstack((0, np.diff(ranges)))

        # Two situations:
        #     1. ranges[i-1] >> ranges[i]  => choose left_indices = ranges[i-1]
        #     2. ranges[i-1] << ranges[i]  => choose right_indices = ranges[i]
        # Concatenate two indices array as output
        left_indices = np.where(disparities > disparity_thresh)[0] - 1
        right_indices = np.where(disparities < -disparity_thresh)[0]
        indices = np.hstack((left_indices, right_indices))

        self.get_logger().info("Num disparities: %0.2f" % len(indices))

        return disparities, indices

    def find_bubbles(self, ranges):
        """ Find bubble indices
        """
        closest_indices = np.argmin(ranges)
        disparities, disparity_indices = self.find_disparities(ranges)
        indices = np.hstack((closest_indices, disparity_indices))

        # Filter result: continuous indices come from same obstacle,
        # only keep the index with the largest disparity
        # e.g. indices = [2, 3, 4] and abs(disparities[2]) > abs(disparities[3]) > abs(disparities[4])
        #      then filtered indices = [2]
        indices = np.sort(indices)
        #bubble_indices = []
        #curr = -1
        #for idx in indices:
        #    if curr == -1 or idx - curr > 1:
        #        bubble_indices.append(idx)
        #    elif abs(disparities[idx]) > abs(disparities[bubble_indices[-1]]):
        #        bubble_indices[-1] = idx
        #    curr = idx
        bubble_indices = indices.copy()

        #self.get_logger().info("Num bubbles: %0.2f" % len(bubble_indices))

        return bubble_indices

    @staticmethod
    def find_max_gap(free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        # Assume ranges array: [1, 1, 1, 0, 0, 3, 3, 0, 0, 2, 2]

        # 1. Prepend and append zero, so all gaps are bounded by zeros
        #    The extended ranges array: [0, 1, 1, 1, 0, 0, 3, 3, 0, 0, 2, 2, 0]
        #    Notice the index is shifted by 1
        extend_ranges = np.hstack((0, free_space_ranges, 0))

        # 2. Find all indices that have zero: [0, 4, 5, 8, 9, 12]
        zero_indices = np.where(abs(extend_ranges<  0.0001))[0]
        #    Calculate the difference between two consecutive indices: [4, 1, 3, 1, 3]
        #    Minus 1 to get the gap angles: [3, 0, 2, 0, 2]
        #    Prepend 0 to align indices: [0, 3, 0, 2, 0, 2]
        #    Now, all positive values represents valid gap angles
        gap_angles = np.hstack((0, np.diff(zero_indices) - 1))

        # 3. Retrieve the start index and end index for each gap in original ranges array
        #    Only consider gap angles that are positive: [F, T, F, T, F, T]
        #    True indices as right bound: [1, 3, 5]
        #    Minus 1 as left bound: [0, 2, 4]
        #    Note here left bound and right bound are indices of indices
        right_bound = np.where(gap_angles > 0)[0]
        left_bound = right_bound - 1
        #    Get gap bounds (two zeros that bind the gap) by taking [[0, 1], [2, 3], [4, 5]]
        #    indices from zero indices array [0, 4, 5, 8, 9, 12]
        #    So the bound zeros are [[0, 4], [5, 8], [9, 12]]
        gap_bounds = zero_indices[np.vstack((left_bound, right_bound)).T]
        #    Then we get real gap bounds by adding [1, -1]
        #    So the real gap bound array: [[1, 3], [6, 7], [10, 11]]
        #    Note that those indices are still shifted by 1, so we need to reduce 1
        gap_bounds = gap_bounds + np.array([1, -1])
        gap_bounds = gap_bounds - 1

        # 4. Calculate area for each gap
        #    Candidate heuristics:
        #        (I)   gap angle
        #        (II)  gap min distance
        #        (III) gap max distance
        #        (IV)  arc length (min_dist * angle || max_dist * angle || mean(dist) * angle)
        #        (V)   integral area (∝ sum{gap ranges ^ 2})
        #    Find max gap of all gaps
        num_gaps = len(gap_bounds)
        gap_angles = gap_angles[gap_angles > 0]

        gap_min_dist = np.zeros(num_gaps, dtype=float)
        for i in range(num_gaps):
            gap_min_dist[i] = np.min(free_space_ranges[gap_bounds[i, 0]:gap_bounds[i, 1] + 1])

        gap_max_dist = np.zeros(num_gaps, dtype=float)
        for i in range(num_gaps):
            gap_max_dist[i] = np.max(free_space_ranges[gap_bounds[i, 0]:gap_bounds[i, 1] + 1])

        gap_min_arcs = gap_angles * gap_min_dist
        gap_max_arcs = gap_angles * gap_max_dist

        gap_areas = np.zeros(num_gaps, dtype=float)
        gap_sect_areas = np.zeros(num_gaps, dtype=float)
        gap_max_areas = np.zeros(num_gaps, dtype=float)
        for i in range(num_gaps):
            gap_areas[i] = np.sum(free_space_ranges[gap_bounds[i, 0]:gap_bounds[i, 1] + 1] ** 2)
            gap_sect_areas[i] = gap_angles[i]*gap_max_dist[i]**2 - gap_angles[i]*gap_min_dist[i]**2
            gap_max_areas[i] = gap_angles[i]*gap_max_dist[i]**2

        # gap_idx = np.argmax(gap_max_arcs)
        gap_idx = np.argmax(gap_max_areas)

        # Retrieve start and end index of resulting gap
        start = gap_bounds[gap_idx, 0]
        end = gap_bounds[gap_idx, 1]

        return start, end

    @staticmethod
    def find_best_point(start_i, end_i, ranges, angles):
        """ Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        If more than one furthest points exist, choose the one with the least steering angle
        """
        gap_ranges = ranges[start_i:end_i + 1]
        gap_angles = angles[start_i:end_i + 1]
        max_dist = np.max(gap_ranges)
        indices = np.where(gap_ranges == max_dist)[0]
        best_angle_idx = np.argmin(abs(gap_angles[indices]))
        best_angle_center = round((start_i + end_i)/2)
        best_angle_dist = start_i + indices[best_angle_idx]
        return best_angle_dist

    def get_steer(self, error):
        """ Get desired steering angle by PID
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

    def get_velocity(self, error):
        """ Get desired velocity based on current error
        """
        # Speed is exponential w.r.t error
        low_vel = self.get_parameter('low_vel').get_parameter_value().double_value
        high_vel = self.get_parameter('high_vel').get_parameter_value().double_value
        atten = self.get_parameter('velocity_attenuation').get_parameter_value().double_value

        return (high_vel - low_vel) * np.exp(-abs(error) * atten) + low_vel

    def lidar_callback(self, data: LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        # 1. Read lidar data
        n = len(data.ranges)
        ranges = np.array(data.ranges)
        angles = np.arange(n) * data.angle_increment + data.angle_min

        # 2. Preprocess lidar data
        ranges = self.preprocess_lidar(ranges, data.range_min, data.range_max)

        # 3. Extract front lidar data (-90° ~ 90°)
        front_indices = np.where((angles > -np.pi / 3) & (angles < np.pi / 3))[0]
        front_ranges = ranges[front_indices]
        front_angles = angles[front_indices]

        # 4. Find bubble points (the closest point / disparities)
        bubble_indices = self.find_bubbles(front_ranges)

        # 5. Eliminate all points inside 'bubble' (set them to zero)
        bubble_radius = self.get_parameter('bubble_radius').get_parameter_value().double_value
        zero_mask = np.zeros_like(front_ranges, dtype=bool)
        for bubble_idx in bubble_indices:
            if bubble_radius < front_ranges[bubble_idx]:
                theta = abs(np.arcsin(bubble_radius / front_ranges[bubble_idx]))
            else:
                self.get_logger().info("Too close to wall")
                theta = np.pi / 2
            mask_idx = int(theta / data.angle_increment)
            min_idx = max(bubble_idx - mask_idx, 0)
            max_idx = min(bubble_idx + mask_idx, len(front_ranges) - 1)
            zero_mask[min_idx:max_idx] = True
        front_ranges[zero_mask] = 0.0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(front_ranges)
        self.get_logger().info("Gap: (%d, %d)" % (gap_start, gap_end))

        # Find the best point in the gap
        best_idx = self.find_best_point(gap_start, gap_end, front_ranges, front_angles)

        # self.get_logger().info("best_idx: %d" % best_idx)

        # Get speed and steer
        curr_error = front_angles[best_idx]
        speed = self.get_velocity(curr_error)
        steer = self.get_steer(curr_error)

        # Publish Drive message
        self.get_logger().info("Error: %0.2f,\t Steer: %0.2f,\t Vel: %0.2f" % (np.rad2deg(curr_error),
                                                                               np.rad2deg(steer),
                                                                               speed))
        message = AckermannDriveStamped()
        message.drive.speed = speed
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
