#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

using namespace std::chrono_literals;

class TalkerCPP : public rclcpp::Node {
public:
    TalkerCPP() : Node("talker") {
        this->declare_parameter("v", 0.0);
        this->declare_parameter("d", 0.0);

        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10);
        timer_ = this->create_wall_timer(0ms, std::bind(&TalkerCPP::timer_callback, this));
    }

private:
    void timer_callback() {
        auto message = ackermann_msgs::msg::AckermannDriveStamped();
        message.drive.speed = (float) this->get_parameter("v").as_double();;
        message.drive.steering_angle = (float) this->get_parameter("d").as_double();
        RCLCPP_INFO(this->get_logger(), "Publishing: speed = %f, steering angle = %f",
                    message.drive.speed, message.drive.steering_angle);
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TalkerCPP>());
    rclcpp::shutdown();
    return 0;
}