#!/bin/bash
source ../decidapt_env/bin/activate
source /opt/ros/jazzy/setup.bash

# Ensure colcon uses the virtual environment's Python
export PYTHONPATH="${VIRTUAL_ENV}/lib/python3.12/site-packages:${PYTHONPATH}"

colcon build --symlink-install

source install/setup.bash 