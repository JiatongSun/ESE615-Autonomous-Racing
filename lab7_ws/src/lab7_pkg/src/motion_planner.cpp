#include "constant.h"
#include "motion_planner.h"

MotionPlanner::MotionPlanner() : Node("motion_planner_node") {
    // ROS Params
    this->declare_parameter("visualize");

    this->declare_parameter("car_map");

    this->declare_parameter("lookahead_distance");
    this->declare_parameter("lookahead_attenuation");
    this->declare_parameter("lookahead_idx");
    this->declare_parameter("lookbehind_idx");

    this->declare_parameter("kp");
    this->declare_parameter("ki");
    this->declare_parameter("kd");
    this->declare_parameter("max_control");
    this->declare_parameter("steer_alpha");

    this->declare_parameter("grid_xmin");
    this->declare_parameter("grid_xmax");
    this->declare_parameter("grid_ymin");
    this->declare_parameter("grid_ymax");
    this->declare_parameter("grid_resolution");
    this->declare_parameter("plot_resolution");
    this->declare_parameter("grid_safe_dist");
    this->declare_parameter("goal_safe_dist");

    this->declare_parameter("use_rrt");
    this->declare_parameter("collision_tol");
    this->declare_parameter("expand_dis");
    this->declare_parameter("path_resolution");
    this->declare_parameter("goal_sample_rate");
    this->declare_parameter("max_iter");
    this->declare_parameter("circle_dist");
    this->declare_parameter("early_stop");
    this->declare_parameter("smooth");
    this->declare_parameter("smoother_type");
    this->declare_parameter("smoother_iter");

    this->declare_parameter("opp_racecar_speed");

    // Timers
    timer_ = this->create_wall_timer(
            1s, std::bind(&MotionPlanner::timer_callback, this));

    // Subscribers
    pose_sub_ = this->create_subscription<Odometry>(
            pose_topic, 1, std::bind(&MotionPlanner::pose_callback, this, std::placeholders::_1));
    opp_pose_sub_ = this->create_subscription<Odometry>(
            opp_pose_topic, 1, std::bind(&MotionPlanner::opp_pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<LaserScan>(
            scan_topic, 1, std::bind(&MotionPlanner::scan_callback, this, std::placeholders::_1));

    // Publishers
    drive_pub_ = this->create_publisher<AckermannDriveStamped>(drive_topic, 10);
    opp_drive_pub_ = this->create_publisher<AckermannDriveStamped>(opp_drive_topic, 10);
    grid_pub_ = this->create_publisher<MarkerArray>(grid_topic, 10);
    rrt_pub_ = this->create_publisher<Marker>(rrt_topic, 10);
    smooth_pub_ = this->create_publisher<Marker>(smooth_topic, 10);
    waypoint_pub_ = this->create_publisher<Marker>(waypoint_topic, 10);
    path_pub_ = this->create_publisher<Marker>(path_topic, 10);
    fps_pub_ = this->create_publisher<Int16>(fps_topic, 10);

    // Map Initialization
    read_map();
}

void MotionPlanner::read_map() {
    string car_map = this->get_parameter("car_map").as_string();
    string csv_loc = "src/lab7_pkg/csv/" + car_map + ".csv";

    fstream fin;
    fin.open(csv_loc, ios::in);
    string line, word;

    while (getline(fin, line)) {
        stringstream s(line);
        vector<string> row;
        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        waypoint_x.push_back(stod(row[0]));
        waypoint_y.push_back(stod(row[1]));
        waypoint_v.push_back(stod(row[2]));
        waypoint_yaw.push_back(stod(row[3]));
        num_waypoints++;
    }
    v_min = *min_element(begin(waypoint_v), end(waypoint_v));
    v_max = *max_element(begin(waypoint_v), end(waypoint_v));
}

void MotionPlanner::timer_callback() {
    Int16 fps;
    fps.data = frame_cnt;
    frame_cnt = 0;
    fps_pub_->publish(fps);
    RCLCPP_INFO(this->get_logger(), "fps: %d", fps.data);
}

void MotionPlanner::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    vector<float> ranges(scan_msg->ranges);
    for (auto &range: ranges) {
        if (range < scan_msg->range_min) {
            range = scan_msg->range_min;
        } else if (range > scan_msg->range_max) {
            range = scan_msg->range_max;
        }
    }

    double xmin = this->get_parameter("grid_xmin").as_double();
    double xmax = this->get_parameter("grid_xmax").as_double();
    double ymin = this->get_parameter("grid_ymin").as_double();
    double ymax = this->get_parameter("grid_ymax").as_double();
    double resolution = this->get_parameter("grid_resolution").as_double();
    double grid_safe_dist = this->get_parameter("grid_safe_dist").as_double();

    int nx = int((xmax - xmin) / resolution) + 1;
    int ny = int((ymax - ymin) / resolution) + 1;

    double x_resolution = (xmax - xmin) / (nx - 1);
    double y_resolution = (ymax - ymin) / (ny - 1);

    // Discretize x and y
    vector<double> xs(nx), ys(ny);
    vector<double>::iterator ptr;
    double val;
    for (ptr = xs.begin(), val = xmin; ptr != xs.end(); ++ptr) {
        *ptr = val;
        val += x_resolution;
    }
    for (ptr = ys.begin(), val = ymin; ptr != ys.end(); ++ptr) {
        *ptr = val;
        val += y_resolution;
    }

    if (grid.empty()) {
        vector<vector<double>> grid_v(nx, vector<double>(ny, -1e8));
        vector<vector<double>> grid_x(nx, vector<double>(ny, -1e8));
        vector<vector<double>> grid_y(nx, vector<double>(ny, -1e8));

        grid.push_back(grid_v);
        grid.push_back(grid_x);
        grid.push_back(grid_y);
    }

    for (int i = 0; i < nx; ++i) {
        double x = xs[i];
        for (int j = 0; j < ny; ++j) {
            double y = ys[j];
            double rho = sqrt(x * x + y * y);
            double phi = atan2(y, x);
            int ray_idx = int((phi - scan_msg->angle_min) / scan_msg->angle_increment);

            grid[0][i][j] = (abs(rho - ranges[ray_idx]) < grid_safe_dist);
            grid[1][i][j] = x;
            grid[2][i][j] = y;
        }
    }
}

void MotionPlanner::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    bool use_rrt = this->get_parameter("use_rrt").as_bool();

    vector<double> res = get_control(pose_msg, use_rrt, true);
    double speed = res[0];
    double steer = res[1];

    // Publish drive message
    AckermannDriveStamped message;
    message.drive.speed = (float) speed;
    message.drive.steering_angle = (float) steer;
    drive_pub_->publish(message);

    // Visualization
    bool visualize = this->get_parameter("visualize").as_bool();
    if (visualize) {
        visualize_occupancy_grid();
        visualize_rrt();
        visualize_smooth_path();
        visualize_waypoints();
    }

    // Increase frame count
    frame_cnt++;
}

void MotionPlanner::opp_pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    vector<double> res = get_control(pose_msg, false, false);
    double speed = this->get_parameter("opp_racecar_speed").as_double();
    double steer = res[1];

    // Publish drive message
    AckermannDriveStamped message;
    message.drive.speed = (float) speed;
    message.drive.steering_angle = (float) steer;
    opp_drive_pub_->publish(message);
}

vector<double> MotionPlanner::get_control(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg,
                                          bool use_rrt, bool is_ego) {
    // Read pose data
    double curr_x = pose_msg->pose.pose.position.x;
    double curr_y = pose_msg->pose.pose.position.y;
    double quat_x = pose_msg->pose.pose.orientation.x;
    double quat_y = pose_msg->pose.pose.orientation.y;
    double quat_z = pose_msg->pose.pose.orientation.z;
    double quat_w = pose_msg->pose.pose.orientation.w;

    vector<double> current_global_pos = {curr_x, curr_y};
    double current_global_yaw = atan2(2 * (quat_w * quat_z + quat_x * quat_y),
                                      1 - 2 * (quat_y * quat_y + quat_z * quat_z));

    if (grid.empty()) {
        // Laser scan not available yet
        return {};
    }

    // Find waypoint index of current position
    vector<double> distances(num_waypoints);
    for (int i = 0; i < num_waypoints; ++i) {
        double dx = waypoint_x[i] - curr_x;
        double dy = waypoint_y[i] - curr_y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
    int curr_idx = (int) (min_element(begin(distances), end(distances)) - distances.begin());

    // Get lookahead distance
    double L = get_lookahead_dist(curr_idx);

    // Search safe global target waypoint
    double goal_safe_dist = this->get_parameter("goal_safe_dist").as_double();
    vector<double> target_global_pos, target_local_pos;
    while (true) {
        // Binary search goal waypoint to track
        int goal_idx = curr_idx;
        while (distances[goal_idx] <= L) {
            goal_idx = (goal_idx + 1) % num_waypoints;
        }

        int left_idx = (goal_idx - 1) % num_waypoints;
        int right_idx = goal_idx;

        double left_x = waypoint_x[left_idx];
        double left_y = waypoint_y[left_idx];
        double right_x = waypoint_x[right_idx];
        double right_y = waypoint_y[right_idx];

        while (true) {
            double mid_x = (left_x + right_x) / 2;
            double mid_y = (left_y + right_y) / 2;
            double dx = mid_x - curr_x;
            double dy = mid_y - curr_y;
            double dist = sqrt(dx * dx + dy * dy);
            if (abs(dist - L) < 1e-2) {
                target_global_pos = {mid_x, mid_y};
                break;
            } else if (dist > L) {
                right_x = mid_x;
                right_y = mid_y;
            } else {
                left_x = mid_x;
                left_y = mid_y;
            }
        }

        // Transform goal point to vehicle frame of reference
        double dx = target_global_pos[0] - curr_x;
        double dy = target_global_pos[1] - curr_y;
        double local_x = cos(current_global_yaw) * dx + sin(current_global_yaw) * dy;
        double local_y = -sin(current_global_yaw) * dx + cos(current_global_yaw) * dy;

        // Check if target point collision free
        target_local_pos = {local_x, local_y};
        if (dist_to_grid(target_local_pos) > goal_safe_dist) {
            break;
        }

        // Target point too close to obstacles, increase L
        L *= 1.1;
    }

    // Use RRT for local planning
    double y_error;
    if (use_rrt) {
        local_planning(target_local_pos);

        if (rrt.path.empty() || rrt.path.size() == 2) {
            y_error = target_local_pos[1];
        } else if (rrt.smooth_path.empty()) {
            y_error = rrt.path[1].y;
        } else {
            y_error = rrt.smooth_path[1].y;
        }
    } else {
        y_error = target_local_pos[1];
    }

    // Get desired speed and steering angle
    double speed = waypoint_v[curr_idx];
    double gamma = 2 / (L * L);
    double error = gamma * y_error;
    double steer = get_steer(error);

    // Update member variables if is ego racecar
    if (is_ego) {
        curr_global_pos = current_global_pos;
        curr_global_yaw = current_global_yaw;
        goal_global_pos = target_global_pos;
        goal_local_pos = target_local_pos;
    }

    return {speed, steer};
}

double MotionPlanner::get_lookahead_dist(int curr_idx) {
    double L = this->get_parameter("lookahead_distance").as_double();
    double slope = this->get_parameter("lookahead_attenuation").as_double();
    int lookahead_idx = (int) this->get_parameter("lookahead_idx").as_int();
    int lookbehind_idx = (int) this->get_parameter("lookbehind_idx").as_int();

    double yaw_before = waypoint_yaw[(curr_idx - lookbehind_idx) % num_waypoints];
    double yaw_after = waypoint_yaw[(curr_idx + lookahead_idx) % num_waypoints];
    double yaw_diff = abs(yaw_after - yaw_before);
    if (yaw_diff > M_PI) {
        yaw_diff -= 2 * M_PI;
    } else if (yaw_diff < -M_PI) {
        yaw_diff += 2 * M_PI;
    }
    yaw_diff = abs(yaw_diff);
    if (yaw_diff > M_PI_2) {
        yaw_diff = M_PI_2;
    }

    L = max(0.5, L * (M_PI_2 - yaw_diff * slope) / M_PI_2);

    return L;
}

double MotionPlanner::dist_to_grid(const vector<double> &pos) {
    int nx = (int) grid[0].size();
    int ny = (int) grid[0][0].size();
    double min_dist = 1e8;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (grid[0][i][j] == 0.0) {
                continue;
            }
            double dx = grid[1][i][j] - pos[0];
            double dy = grid[2][i][j] - pos[1];
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }

    return min_dist;
}

double MotionPlanner::get_steer(double error) {
    double kp = this->get_parameter("kp").as_double();
    double ki = this->get_parameter("ki").as_double();
    double kd = this->get_parameter("kd").as_double();
    double max_control = this->get_parameter("max_control").as_double();
    double alpha = this->get_parameter("steer_alpha").as_double();

    double d_error = error - prev_error;
    prev_error = error;
    integral += error;
    double steer = kp * error + ki * integral + kd * d_error;
    if (steer < -max_control) {
        steer = -max_control;
    } else if (steer > max_control) {
        steer = max_control;
    }
    steer = alpha * steer + (1 - alpha) * prev_steer;
    prev_steer = steer;

    return steer;
}

void MotionPlanner::local_planning(const vector<double> &goal_pos) {
    double collision_tol = this->get_parameter("collision_tol").as_double();
    double expand_dis = this->get_parameter("expand_dis").as_double();
    double path_resolution = this->get_parameter("path_resolution").as_double();
    double goal_sample_rate = this->get_parameter("goal_sample_rate").as_double();
    int max_iter = (int) this->get_parameter("max_iter").as_int();
    double circle_dist = this->get_parameter("circle_dist").as_double();
    bool early_stop = this->get_parameter("early_stop").as_bool();
    bool smooth = this->get_parameter("smooth").as_bool();

    rrt = RRT({0, 0},
              goal_pos,
              grid,
              collision_tol,
              expand_dis,
              path_resolution,
              goal_sample_rate,
              max_iter,
              circle_dist,
              early_stop,
              smooth);
    rrt.planning();
}
