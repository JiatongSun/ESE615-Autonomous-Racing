# Lab 0: Docker and ROS 2

## Docker Hub ID
jtsun

## Written Questions

### Q1: During this assignment, you've probably ran these two following commands at some point: ```source /opt/ros/foxy/setup.bash``` and ```source install/local_setup.bash```. Functionally what is the difference between the two?

Answer: 

```source /opt/ros/foxy/setup.bash``` is to source the main ROS 2 installation, which is the underlay; ```source install/local_setup.bash``` will add the packages available in the overlay to the environment. The overlay gets prepended to the path, and takes precedence over the underlay.

### Q2: What does the ```queue_size``` argument control when creating a subscriber or a publisher? How does different ```queue_size``` affect how messages are handled?

Answer: 

```queue_size``` controls the size of the outgoing message queue for a publisher, or the size of incoming message queue for a subscriber. If the publisher is publishing faster than the message can be sent, the old messages will be dropped; If the messages arrive too fast on a subscriber node, the old messages will be thrown away. 

### Q3: Do you have to call ```colcon build``` again after you've changed a launch file in your package? (Hint: consider two cases: calling ```ros2 launch``` in the directory where the launch file is, and calling it when the launch file is installed with the package.)

Answer: 
1. If ```ros2 launch``` is called in the ```launch``` directory, there's no need to call ```colcon build``` again; 
2. If ```ros2 launch``` is called when the launch file is installed with ```--symlink-install``` specified, there's also no need to call ```colcon build``` again; 
3. If ```ros2 launch``` is called when the launch file is installed but ```--symlink-install``` is not specified, the ```colcon build``` needs to be called again.

### Q4 (optional): While completing this lab, are there any parts of the tutorial and lab instruction that could be improved?

Answer: 
No, instruction is perfect :)