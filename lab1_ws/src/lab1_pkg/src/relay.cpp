#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

using std::placeholders::_1;

class RelayCPP : public rclcpp::Node {
public:
    RelayCPP() : Node("relay") {
        subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
                "drive", 10, std::bind(&RelayCPP::topic_callback, this, _1));
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("drive_relay", 10);
    }

private:
    void topic_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(), "I heard: speed = %f, steering angle = %f",
                    msg->drive.speed, msg->drive.steering_angle);

        auto relay_msg = ackermann_msgs::msg::AckermannDriveStamped();
        relay_msg.drive.speed = msg->drive.speed * 3;
        relay_msg.drive.steering_angle = msg->drive.steering_angle * 3;
        RCLCPP_INFO(this->get_logger(), "Relay: speed = %f, steering angle = %f",
                    relay_msg.drive.speed, relay_msg.drive.steering_angle);
        publisher_->publish(relay_msg);
    }

    rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr subscription_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RelayCPP>());
    rclcpp::shutdown();
    return 0;
}
