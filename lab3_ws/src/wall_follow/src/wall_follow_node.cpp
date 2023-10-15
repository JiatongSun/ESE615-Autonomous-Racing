#include "rclcpp/rclcpp.hpp"
#include <string>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

using std::placeholders::_1;

class WallFollow : public rclcpp::Node {

public:
    WallFollow() : Node("wall_follow_node") {
        this->declare_parameter("wall", "left");
        this->declare_parameter("theta_min", 30);
        this->declare_parameter("theta_max", 60);
        this->declare_parameter("theta_step", 5);
        this->declare_parameter("steer_min", -35);
        this->declare_parameter("steer_max", 35);
        this->declare_parameter("filter", "iqr");
        this->declare_parameter("L", 2.5);
        this->declare_parameter("dist", 1.35);
        this->declare_parameter("err_low_tol", 0.15);
        this->declare_parameter("err_high_tol", 0.30);
	this->declare_parameter("steer_low_tol",5.0*M_PI/180.0);
	this->declare_parameter("steer_high_tol",15.0*M_PI/180.0);
        this->declare_parameter("low_vel", 1.0);
        this->declare_parameter("mid_vel", 3.0);
        this->declare_parameter("high_vel", 4.0);
        this->declare_parameter("kp", 0.27);
        this->declare_parameter("kd", 0.10);
        this->declare_parameter("ki", 0.0002);

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                lidarscan_topic, 10, std::bind(&WallFollow::scan_callback, this, _1));
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
                drive_topic, 10);
    }

private:
    // PID CONTROL PARAMS
    double kp = 0.0;
    double kd = 0.0;
    double ki = 0.0;
    double prev_error = 0.0;
    double integral = 0.0;

    // Topics
    std::string lidarscan_topic = "/scan";
    std::string drive_topic = "/drive";

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

    // Laser Scan Params
    float angle_min = 0.0, angle_max = 0.0;
    float angle_increment = 0.0;
    float range_min = 0.0, range_max = 0.0;

    double get_range(const float *range_data, double angle) const {
        /*
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle
        */
        if (angle_increment == 0.0 && angle_min == 0.0) return -1;  // constants uninitialized
        int idx = int((angle - angle_min) / angle_increment);
        float range = range_data[idx];
        if (range < range_min || range > range_max) return -1;  // out of bounds
        return range;
    }

    double get_error(float *range_data, double dist) {
        /*
        Calculates the error to the wall. Follow the wall to the left (going counterclockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        */

        // Calculate alpha for different theta
        int theta_min = (int) this->get_parameter("theta_min").as_int();
        int theta_max = (int) this->get_parameter("theta_max").as_int();
        int theta_step = (int) this->get_parameter("theta_step").as_int();

        std::string wall = this->get_parameter("wall").as_string();

        double alpha = 0.0, theta = 0.0;
        double a = 0.0, b = 0.0;
        if (wall == "left") {
            b = get_range(range_data, M_PI_2);
        } else if (wall == "right") {
            b = get_range(range_data, -M_PI_2);
        }
        std::vector<double> alphas;
        for (int i = theta_min; i <= theta_max; i += theta_step) {
            theta = i * M_PI / 180.0;
            if (wall == "left") {
                a = get_range(range_data, M_PI_2 - theta);
            } else if (wall == "right") {
                a = get_range(range_data, -M_PI_2 + theta);
            }

            // Trigonometry
            if (wall == "left") {
                alpha = atan2(b - a * cos(theta), a * sin(theta));
            } else if (wall == "right") {
                alpha = atan2(a * cos(theta) - b, a * sin(theta));
            }
            alphas.push_back(alpha);
        }

        // Filter outliers
        std::string filter = this->get_parameter("filter").as_string();
        std::vector<bool> valids;
        if (filter == "iqr") {
            valids = getInliersIQR(alphas);
        } else if (filter == "std") {
            valids = getInliersStd(alphas);
        }
        double total = 0.0;
        int num_valid = 0;
        for (int i = 0; i < (int) alphas.size(); ++i) {
            if (!valids[i]) continue;
            total += alphas[i];
            num_valid++;
        }
        alpha = total / num_valid;
        RCLCPP_INFO(this->get_logger(), "Alpha:  %.2f", alpha * 180.0 / M_PI);

        // Calculate error
        double L = this->get_parameter("L").as_double();
        double d_curr = b * cos(alpha);
        double d_next = 0.0;
        double error = 0.0;
        if (wall == "left") {
            d_next = d_curr - L * sin(alpha);
            error = d_next - dist;
        } else if (wall == "right") {
            d_next = d_curr + L * sin(alpha);
            error = dist - d_next;
        }

        return error;
    }

    void pid_control(double error) { //, double velocity) {
        /*
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        */
        kp = this->get_parameter("kp").as_double();
        kd = this->get_parameter("kd").as_double();
        ki = this->get_parameter("ki").as_double();
        RCLCPP_INFO(this->get_logger(), "Kp: %.2f, Kd: %.2f, Ki: %.2f, Err: %.4f", kp, kd, ki, error);
        double d_err = error - prev_error;
        prev_error = error;
        double err_high_tol = this->get_parameter("err_high_tol").as_double();
	if (error <= err_high_tol) {
            integral += error;
	}
        double angle = kp * error + ki * integral + kd * d_err;
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();

        // Add lower bound and upper bound constraint to steering angle
        double steer_min = (int) this->get_parameter("steer_min").as_int() * M_PI / 180.0;
        double steer_max = (int) this->get_parameter("steer_max").as_int() * M_PI / 180.0;
        if (angle > steer_max) {
            angle = steer_max;
        } else if (angle < steer_min) {
            angle = steer_min;
        }
        RCLCPP_INFO(this->get_logger(), "Control: Angle: %.4f, Velocity: %.4f", angle * 180.0 / M_PI, get_velocity(angle));//velocity);
        drive_msg.drive.speed = get_velocity(angle);//float(velocity);
        drive_msg.drive.steering_angle = float(angle);
        drive_pub_->publish(drive_msg);
    }

    double get_velocity(double error) {
        /*
        Based on the calculated error, calculate desired velocity

        Args:
            error: calculated error

        Returns:
            velocity: desired velocity
        */
        double err_low_tol = this->get_parameter("err_low_tol").as_double();
        double err_high_tol = this->get_parameter("err_high_tol").as_double();
        double low_vel = this->get_parameter("low_vel").as_double();
        double mid_vel = this->get_parameter("mid_vel").as_double();
        double high_vel = this->get_parameter("high_vel").as_double();
        if (abs(error) <= err_low_tol) {
            return high_vel;
        } else if (abs(error) <= err_high_tol) {
            return mid_vel;
        } else {
            return low_vel;
        }
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        /*
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        */
        angle_min = scan_msg->angle_min;
        angle_max = scan_msg->angle_max;
        angle_increment = scan_msg->angle_increment;
        range_min = scan_msg->range_min;
        range_max = scan_msg->range_max;

        std::vector<float> ranges = scan_msg->ranges;
        double dist = this->get_parameter("dist").as_double();

        double error = get_error(&ranges[0], dist);
        //double velocity = get_velocity(error);
        pid_control(error);//, velocity);
    }

    static int median(int l, int r) {
        int n = r - l + 1;
        n = (n + 1) / 2 - 1;
        return n + l;
    }

    static std::vector<bool> getInliersIQR(const std::vector<double> &nums) {
        /*
        Helper to remove outliers in nums based on IQR

        Args:
            nums: array of numbers

        Returns:
            valids: array of flags indicating inlier or outlier
        */
        std::vector<double> temp(nums);
        std::sort(temp.begin(), temp.end());
        int n = (int) nums.size();
        int mid_index = median(0, n);
        double Q1 = temp[median(0, mid_index)];
        double Q3 = temp[median(mid_index + 1, n)];
        double IQR = Q3 - Q1;
        double lb = Q1 - 1.5 * IQR;
        double ub = Q3 + 1.5 * IQR;
        std::vector<bool> res(n, true);
        for (int i = 0; i < n; ++i) {
            if (nums[i] < lb || nums[i] > ub) res[i] = false;  // outlier
        }
        return res;
    }

    static std::vector<bool> getInliersStd(const std::vector<double> &nums) {
        /*
        Helper to remove outliers in nums based on mean and standard deviation

        Args:
            nums: array of numbers

        Returns:
            valids: array of flags indicating inlier or outlier
        */
        int n = (int) nums.size();

        double sum = std::accumulate(nums.begin(), nums.end(), 0.0);
        double mean = sum / n;

        double sq_sum = 0.0;
        std::for_each(nums.begin(), nums.end(), [&](const double d) {
            sq_sum += (d - mean) * (d - mean);
        });

        double stdev = sqrt(sq_sum / (n - 1));

        std::vector<bool> res(n, true);
        for (int i = 0; i < n; ++i) {
            if (nums[i] < mean - stdev || nums[i] > mean + stdev) res[i] = false;  // outlier
        }
        return res;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<WallFollow>());
    rclcpp::shutdown();
    return 0;
}
