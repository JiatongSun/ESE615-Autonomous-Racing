FROM f1tenth_gym_ros:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y zsh curl git wget locales && \
    locale-gen en_US.UTF-8 && \
    chsh -s /bin/zsh

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

COPY . /sim_ws/src
RUN source /opt/ros/foxy/setup.bash && \
    cd /sim_ws/ && \
    rosdep install -i --from-path src --rosdistro foxy -y && \
    colcon build --symlink-install

RUN echo 'export ZSH="$HOME/.oh-my-zsh"' >> ~/.zshrc && \
    echo 'ZSH_THEME="robbyrussell"' >> ~/.zshrc && \
    echo 'plugins=(git zsh-autosuggestions zsh-completions zsh-syntax-highlighting)' >> ~/.zshrc && \
    echo 'source $ZSH/oh-my-zsh.sh' >> ~/.zshrc && \
    echo 'source /opt/ros/foxy/setup.zsh' >> ~/.zshrc && \
    echo 'source /sim_ws/install/setup.zsh' >> ~/.zshrc
    
WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/zsh"]
