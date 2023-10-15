FROM f1tenth_gym_ros:latest

SHELL ["/bin/bash", "-c"]

COPY . /sim_ws/src
RUN source /opt/ros/foxy/setup.bash && \
    cd /sim_ws/ && \
    colcon build --symlink-install

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]
