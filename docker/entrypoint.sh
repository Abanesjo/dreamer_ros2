#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

cd /workspace/ros2_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

source /workspace/ros2_ws/install/setup.bash

grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc || \
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

grep -qxF "source /workspace/ros2_ws/install/setup.bash" ~/.bashrc || \
    echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

grep -qxF 'export RMW_IMPLEMENTATION="rmw_zenoh_cpp"' ~/.bashrc || \
    echo 'export RMW_IMPLEMENTATION="rmw_zenoh_cpp"' >> ~/.bashrc

grep -qxF 'export ZENOH_ROUTER_CONFIG_URI="/workspace/ros2_ws/src/RMW_ZENOH_ROUTER_CONFIG.json5"' ~/.bashrc || \
    echo 'export ZENOH_ROUTER_CONFIG_URI="/workspace/ros2_ws/src/RMW_ZENOH_ROUTER_CONFIG.json5"' >> ~/.bashrc

exec bash