#!/bin/bash

source /opt/ros/jazzy/setup.bash

cd /workspace/ros2_ws

rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install

echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///workspace/ros2_ws/src/cyclonedds.xml" >> ~/.bashrc

source /workspace/ros2_ws/install/setup.bash

grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc || \
    echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

grep -qxF "source /workspace/ros2_ws/install/setup.bash" ~/.bashrc || \
    echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

grep -qxF "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" ~/.bashrc || \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc

grep -qxF "export CYCLONEDDS_URI=file:///workspace/ros2_ws/src/cyclonedds.xml" ~/.bashrc || \
    echo "export CYCLONEDDS_URI=file:///workspace/ros2_ws/src/cyclonedds.xml" >> ~/.bashrc

cd /workspace/ros2_ws

exec bash