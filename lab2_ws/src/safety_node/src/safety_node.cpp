#include "rclcpp/rclcpp.hpp"
/// CHECK: include needed ROS msg type headers and libraries
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

using std::placeholders::_1;

class Safety : public rclcpp::Node {
// The class that handles emergency braking

public:
    Safety() : Node("safety_node") {
        /*
        You should also subscribe to the /scan topic to get the
        sensor_msgs/LaserScan messages and the /ego_racecar/odom topic to get
        the nav_msgs/Odometry messages

        The subscribers should use the provided odom_callback and 
        scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        */

        this->declare_parameter("ttc_tol", 1.5);

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                "/scan", 10, std::bind(&Safety::scan_callback, this, _1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "/ego_racecar/odom", 10, std::bind(&Safety::drive_callback, this, _1));
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                "/drive", 10);
    }

private:
    double speed = 0.0;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;

    void drive_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        speed = msg->twist.twist.linear.x;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        float iTTC = 1e8;
        for (int i = 0; i < (int) scan_msg->ranges.size(); ++i) {
            float curr_range = scan_msg->ranges[i];
            if (curr_range > scan_msg->range_max || curr_range < scan_msg->range_min) {
                continue;
            }
            float curr_angle = scan_msg->angle_min + scan_msg->angle_increment * float(i);
            float curr_vel = (float) speed * std::cos(curr_angle);
            if (curr_vel <= 0) {
                curr_vel = 1e-8;
            }
            float curr_ittc = curr_range / curr_vel;
            iTTC = std::min(iTTC, curr_ittc);
        }
        RCLCPP_INFO(this->get_logger(), "iTTC: %f", iTTC);
        if (iTTC < this->get_parameter("ttc_tol").as_double()) {
            auto message = ackermann_msgs::msg::AckermannDriveStamped();
            message.drive.speed = 0.0;
            publisher_->publish(message);
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}