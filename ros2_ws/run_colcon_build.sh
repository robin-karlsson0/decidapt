#!/bin/bash
source ../decidapt_env/bin/activate
source /opt/ros/jazzy/setup.bash

# Update external ROS 2 package dependencies
vcs pull src

# Upgrade external pip packages
cd ../
uv pip install --upgrade -r requirements.txt

# Upgrade all local pip packages
git submodule update --remote
uv pip install --upgrade \
    ros2_ws/src/exodapt_robot_pt \
    ros2_ws/src/actions

cd ros2_ws

rm -rf build install log

# NOTE: Need setuptools for building, but conflict with ROS 2 when running
uv pip install setuptools
colcon build --symlink-install
uv pip uninstall setuptools

source install/setup.bash

# Ensure ROS 2 packages uses the virtual environment's Python
export PYTHONPATH="${VIRTUAL_ENV}/lib/python3.12/site-packages:${PYTHONPATH}"