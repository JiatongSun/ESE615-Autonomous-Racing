#include "motion_planner.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionPlanner>());
    rclcpp::shutdown();
    return 0;
}