#ifndef MOTION_PLANNER_H
#define MOTION_PLANNER_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int16.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "rrt.h"

using namespace std;

typedef nav_msgs::msg::Odometry Odometry;
typedef sensor_msgs::msg::LaserScan LaserScan;
typedef ackermann_msgs::msg::AckermannDriveStamped AckermannDriveStamped;
typedef visualization_msgs::msg::MarkerArray MarkerArray;
typedef visualization_msgs::msg::Marker Marker;
typedef std_msgs::msg::Int16 Int16;

class MotionPlanner : public rclcpp::Node {
public:
    MotionPlanner();

private:
    // PID Control Variables
    double prev_error = 0.0;
    double integral = 0.0;
    double prev_steer = 0.0;

    // Global Map Variables
    int num_waypoints = 0;
    vector<double> waypoint_x;
    vector<double> waypoint_y;
    vector<double> waypoint_v;
    vector<double> waypoint_yaw;
    double v_max = 0.0;
    double v_min = 0.0;

    // Local Map Variables
    RRT rrt;
    vector<vector<vector<double>>> grid;

    // Car State Variables
    vector<double> curr_global_pos;
    double curr_global_yaw = 0.0;
    vector<double> goal_global_pos;
    vector<double> goal_local_pos;

    // Other Variables
    short frame_cnt = 0;

    // Topics
    string pose_topic = "/ego_racecar/odom";
    string scan_topic = "/scan";
    string drive_topic = "/drive";

    string opp_pose_topic = "/opp_racecar/odom";
    string opp_drive_topic = "/opp_drive";

    string grid_topic = "/grid";
    string rrt_topic = "/rrt";
    string smooth_topic = "/smooth";
    string waypoint_topic = "/waypoint";
    string path_topic = "/global_path";
    string fps_topic = "/fps";

    // Timers
    rclcpp::TimerBase::SharedPtr timer_;

    // Subscribers
    rclcpp::Subscription<Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<Odometry>::SharedPtr opp_pose_sub_;
    rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<AckermannDriveStamped>::SharedPtr opp_drive_pub_;
    rclcpp::Publisher<MarkerArray>::SharedPtr grid_pub_;
    rclcpp::Publisher<Marker>::SharedPtr rrt_pub_;
    rclcpp::Publisher<Marker>::SharedPtr smooth_pub_;
    rclcpp::Publisher<Marker>::SharedPtr waypoint_pub_;
    rclcpp::Publisher<Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<Int16>::SharedPtr fps_pub_;

    // Member Functions
    void read_map();

    void timer_callback();

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);

    void opp_pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);

    vector<double> get_control(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg,
                               bool use_rrt = false, bool is_ego = true);

    double get_lookahead_dist(int curr_idx);

    double dist_to_grid(const vector<double> &pos);

    double get_steer(double error);

    void local_planning(const vector<double> &goal_pos);

    void visualize_occupancy_grid();

    void visualize_rrt();

    void visualize_smooth_path();

    void visualize_waypoints();
};

#endif //MOTION_PLANNER_H
