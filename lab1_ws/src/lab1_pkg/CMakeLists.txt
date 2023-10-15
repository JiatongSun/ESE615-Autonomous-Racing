cmake_minimum_required(VERSION 3.5)
project(lab1_pkg)

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ackermann_msgs REQUIRED)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(talker_cpp src/talker.cpp)
ament_target_dependencies(talker_cpp rclcpp ackermann_msgs)

add_executable(relay_cpp src/relay.cpp)
ament_target_dependencies(relay_cpp rclcpp ackermann_msgs)

# Install Cpp executables
install(TARGETS
        talker_cpp relay_cpp
        DESTINATION lib/${PROJECT_NAME}
        )
# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
        scripts/talker_py.py
        scripts/relay_py.py
        DESTINATION lib/${PROJECT_NAME}
        )

# Install launch executables
install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}
        )

ament_package()
