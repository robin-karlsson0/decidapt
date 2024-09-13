# robot_action_coordinator
Robot Action coordinator package


# Installation

Create and source virtual environment
```
pyenv virtualenv 3.10.12 robot_action_coordinator
source ~/.pyenv/versions/robot_action_coordinator/bin/activate
```

Install depenencies
```
pip install -r requirements.txt
```

```
cd ros2_ws

# Source action interface packages
source /PATH/TO/robot_llm/robot_llm_action_server/ros2_ws/install/setup.bash
source /PATH/TO/robot_state_manager/ros2_ws/install/setup.bash
source /PATH/TO/robot_action_reply/ros2_ws/install/setup.bash

# For install dependencies
export PYTHONPATH=$PYTHONPATH:/home/USER/.pyenv/versions/robot_action_
coordinator/lib/python3.X//site-packages

colcon build

source install/setup.bash

ros2 run robot_action_controller action_decision
```


# robot_action_controller



# robot_action_decision

Service for predicting the optimal action to take based on the current state.
