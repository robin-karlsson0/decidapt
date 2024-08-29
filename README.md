# robot_action_coordinator
Robot Action coordinator package


# Installation


```
cd ros2_ws

# Source action interface packages
source /PATH/TO/robot_llm/robot_llm_action_server/ros2_ws/install/setup.bash
source /PATH/TO/robot_action_reply/ros2_ws/install/setup.bash

colcon build
source install/setup.bash
source path/to/robot_llm_pkg/ros2_ws/install/setup.bash

ros2 run robot_action_controller action_decision
```


# robot_action_controller



# robot_action_decision

Service for predicting the optimal action to take based on the current state.
