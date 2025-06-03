#!/bin/bash
source ../decidapt_env/bin/activate
source /opt/ros/jazzy/setup.bash

# NOTE: Need setuptools for building, but conflict with ROS 2 when running
uv pip install setuptools
colcon build --symlink-install
uv pip uninstall setuptools

source install/setup.bash

# Ensure ROS 2 packages uses the virtual environment's Python
export PYTHONPATH="${VIRTUAL_ENV}/lib/python3.12/site-packages:${PYTHONPATH}"