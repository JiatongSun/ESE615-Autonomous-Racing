FROM ros:foxy

SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       vim \
                       python3-pip \
                       tmux

RUN pip3 install --upgrade pip

RUN mkdir -p lab1_ws/src
COPY . /lab1_ws/src
RUN source /opt/ros/foxy/setup.bash && \
    cd lab1_ws/ && \
    rosdep install -i --from-path src --rosdistro foxy -y && \
    colcon build --symlink-install

WORKDIR '/lab1_ws'
ENTRYPOINT ["/bin/bash"]   
