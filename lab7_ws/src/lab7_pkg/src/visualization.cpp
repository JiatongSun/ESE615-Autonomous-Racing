#include "motion_planner.h"

void MotionPlanner::visualize_occupancy_grid() {
    if (grid.empty()) {
        return;
    }

    double grid_resolution = this->get_parameter("grid_resolution").as_double();
    double plot_resolution = this->get_parameter("plot_resolution").as_double();
    int down_sample = max(1, int(plot_resolution / grid_resolution));

    int nx = (int) grid[0].size();
    int ny = (int) grid[0][0].size();

    MarkerArray marker_arr;

    int id = 0;

    for (int i = 0; i < nx; ++i) {
        if (i % down_sample) continue;
        for (int j = 0; j < ny; ++j) {
            if (j % down_sample) continue;
            if (grid[0][i][j] == 0.0) continue;

            // Transform to map frame
            // Rotation
            double x = grid[1][i][j] * cos(curr_global_yaw) - grid[2][i][j] * sin(curr_global_yaw);
            double y = grid[1][i][j] * sin(curr_global_yaw) + grid[2][i][j] * cos(curr_global_yaw);

            // Translation
            x += curr_global_pos[0];
            y += curr_global_pos[1];

            // Add marker
            Marker marker;
            marker.header.frame_id = "/map";
            marker.id = i;
            marker.ns = "occupancy_grid_" + to_string(id++);
            marker.type = Marker::CUBE;
            marker.action = Marker::ADD;

            marker.pose.position.x = x;
            marker.pose.position.y = y;

            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            marker.lifetime.nanosec = int(1e8);

            marker_arr.markers.push_back(marker);
        }
    }

    grid_pub_->publish(marker_arr);
}

void MotionPlanner::visualize_rrt() {
    if (rrt.tree.empty()) {
        return;
    }

    Marker line_list;
    line_list.header.frame_id = "/map";
    line_list.id = 0;
    line_list.ns = "rrt";
    line_list.type = Marker::LINE_LIST;
    line_list.action = Marker::ADD;

    line_list.scale.x = 0.1;
    line_list.scale.y = 0.1;
    line_list.scale.z = 0.1;

    for (auto & node : rrt.tree) {
        if (node.is_root) {
            continue;
        }

        vector<float> tmp_color = {0.0, 0.0, 1.0};
        for (const auto &p: rrt.path) {
            double dx = p.x - node.x;
            double dy = p.y - node.y;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist == 0.0) {
                tmp_color = {0.0, 1.0, 0.0};
                break;
            }
        }

        double x, y;

        // Add first point
        TreeNode parent_node = rrt.tree[node.parent];

        x = parent_node.x * cos(curr_global_yaw) - parent_node.y * sin(curr_global_yaw);
        y = parent_node.x * sin(curr_global_yaw) + parent_node.y * cos(curr_global_yaw);
        x += curr_global_pos[0];
        y += curr_global_pos[1];

        geometry_msgs::msg::Point p1;
        p1.x = x;
        p1.y = y;
        line_list.points.push_back(p1);

        std_msgs::msg::ColorRGBA c1;
        c1.r = tmp_color[0];
        c1.g = tmp_color[1];
        c1.b = tmp_color[2];
        c1.a = 1.0;
        line_list.colors.push_back(c1);

        // Add second point
        TreeNode child_node = node;

        x = child_node.x * cos(curr_global_yaw) - child_node.y * sin(curr_global_yaw);
        y = child_node.x * sin(curr_global_yaw) + child_node.y * cos(curr_global_yaw);
        x += curr_global_pos[0];
        y += curr_global_pos[1];

        geometry_msgs::msg::Point p2;
        p2.x = x;
        p2.y = y;
        line_list.points.push_back(p2);

        std_msgs::msg::ColorRGBA c2;
        c2.r = tmp_color[0];
        c2.g = tmp_color[1];
        c2.b = tmp_color[2];
        c2.a = 1.0;
        line_list.colors.push_back(c2);
    }

    rrt_pub_->publish(line_list);
}

void MotionPlanner::visualize_smooth_path() {
    if (rrt.smooth_path.empty()) {
        return;
    }

    Marker line_list;
    line_list.header.frame_id = "/map";
    line_list.id = 0;
    line_list.ns = "smooth_path";
    line_list.type = Marker::LINE_LIST;
    line_list.action = Marker::ADD;

    line_list.scale.x = 0.1;
    line_list.scale.y = 0.1;
    line_list.scale.z = 0.1;

    vector<float> tmp_color = {1.0, 0.4, 0.7};

    for (int i = 0; i < (int) rrt.smooth_path.size() - 1; ++i) {
        double x, y;

        // Add first point
        TreeNode parent_node = rrt.smooth_path[i];

        x = parent_node.x * cos(curr_global_yaw) - parent_node.y * sin(curr_global_yaw);
        y = parent_node.x * sin(curr_global_yaw) + parent_node.y * cos(curr_global_yaw);
        x += curr_global_pos[0];
        y += curr_global_pos[1];

        geometry_msgs::msg::Point p1;
        p1.x = x;
        p1.y = y;
        line_list.points.push_back(p1);

        std_msgs::msg::ColorRGBA c1;
        c1.r = tmp_color[0];
        c1.g = tmp_color[1];
        c1.b = tmp_color[2];
        c1.a = 1.0;
        line_list.colors.push_back(c1);

        // Add second point
        TreeNode child_node = rrt.smooth_path[i + 1];

        x = child_node.x * cos(curr_global_yaw) - child_node.y * sin(curr_global_yaw);
        y = child_node.x * sin(curr_global_yaw) + child_node.y * cos(curr_global_yaw);
        x += curr_global_pos[0];
        y += curr_global_pos[1];

        geometry_msgs::msg::Point p2;
        p2.x = x;
        p2.y = y;
        line_list.points.push_back(p2);

        std_msgs::msg::ColorRGBA c2;
        c2.r = tmp_color[0];
        c2.g = tmp_color[1];
        c2.b = tmp_color[2];
        c2.a = 1.0;
        line_list.colors.push_back(c2);
    }

    smooth_pub_->publish(line_list);
}

void MotionPlanner::visualize_waypoints() {
    Marker line_strip;
    line_strip.header.frame_id = "/map";
    line_strip.id = 0;
    line_strip.ns = "global_planner";
    line_strip.type = Marker::LINE_STRIP;
    line_strip.action = Marker::ADD;
    for (int i = 0; i <= num_waypoints; ++i) {
        geometry_msgs::msg::Point point;
        point.x = waypoint_x[i % num_waypoints];
        point.y = waypoint_y[i % num_waypoints];
        line_strip.points.push_back(point);

        std_msgs::msg::ColorRGBA color;
        double speed_ratio = (waypoint_v[i % num_waypoints] - v_min) / (v_max - v_min);
        color.r = (float) (1.0 - speed_ratio);
        color.g = (float) speed_ratio;
        color.b = 0.0;
        color.a = 1.0;
        line_strip.colors.push_back(color);
    }
    line_strip.scale.x = 0.1;
    line_strip.scale.y = 0.1;
    line_strip.scale.z = 0.1;

    path_pub_->publish(line_strip);

    Marker marker;
    marker.header.frame_id = "/map";
    marker.id = 0;
    marker.ns = "target_waypoint";
    marker.type = Marker::SPHERE;
    marker.action = Marker::ADD;

    marker.pose.position.x = goal_global_pos[0];
    marker.pose.position.y = goal_global_pos[1];

    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;

    waypoint_pub_->publish(marker);
}
