FROM ros:jazzy


ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt update && apt upgrade -y

RUN apt install ros-jazzy-desktop -y 

RUN apt install -y \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rosidl-generator-dds-idl \
    ros-jazzy-rosbag2-cpp \
    ros-jazzy-vision-msgs \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-joint-state-publisher-gui \
    ros2-jazzy-navigation2 \ 
    ros-jazzy-nav2-bringup \
    python3-pip \
    libyaml-cpp-dev \
    python3-pip \
    libspdlog-dev \
    libfmt-dev \
    tmux

RUN python3 -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130

RUN mkdir -p /workspace/ros2_ws

WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]



