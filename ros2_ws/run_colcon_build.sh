#!/bin/bash
source /home/$USER/projects/robot_llm/robot_llm_action_server/ros2_ws/install/setup.bash
source /home/$USER/projects/robot_state_manager/ros2_ws/install/setup.bash
source /home/$USER/projects/robot_action_reply/ros2_ws/install/setup.bash

# For install dependencies
source /home/$USER/.pyenv/versions/robot_action_coordinator/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/$USER/.pyenv/versions/robot_action_coordinator/lib/python3.10/site-packages

# Local Robot prompt templates repository
export PYTHONPATH=/home/$USER/projects/robot_prompt_templates/src/:$PYTHONPATH

colcon build --symlink-install
