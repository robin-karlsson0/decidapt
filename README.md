# Decidapt
Proactive Decision-Making Framework for Autonomous Robots

Decidapt is a ROS 2 software package providing a continuous, proactive decision-making framework for autonomous robots and AI systems. It enables robots to iteratively select actions based on their goals, internal state, and environment state through a decision cycle specified running at short fixed interval.

### Key Features

- **Continuous Decision Cycle**: Operates at configurable frequencies (e.g., 2 Hz) to continuously evaluate and select appropriate actions
- **Proactive Behavior**: Anticipates needs and initiates actions without explicit commands, creating more autonomous and responsive robot behavior
- **Comprehensive State Integration**: Incorporates AI profile (agency, preferences), robot state, and environment state into decision processes
- **Action Management**: Tracks running actions, handles conflicts, and gracefully transitions between different behaviors
- **Experiential Learning**: Learns from action-outcome experience by self-reflection
- **ROS 2 Integration**: Integrates with ROS 2 ecosystem via nodes, services, and a modular architecture

### Core Components

- **ActionDecisionManager**: Implements the cyclical decision-making process, evaluating possible actions against current state
- **ActionExecutionManager**: Handles the lifecycle of actions, including starting, monitoring, and graceful termination
- **StateRepresentation**: Maintains comprehensive models of robot capabilities, environmental context, and operational goals
- **DecisionStrategies**: Pluggable modules for different decision-making algorithms and priority handling

### Applications

Decidapt is designed for robotics researchers and developers working on:

- Socially interactive robots requiring initiative and responsiveness
- Service robots that must operate autonomously for extended periods
- Research platforms exploring proactive AI behavior and decision-making
- Multi-robot systems requiring coordinated but independent decision processes

# Installation and build

Create and source virtual environment
```
# Create virtual environment with uv
uv venv decidapt_env
source decidapt_env/bin/activate

# Source ROS 2 after activating virtual environment
source /opt/ros/jazzy/setup.bash
```

Import external ROS 2 package dependencies
```bash
vcs import ros2_ws/src/ < .rosinstall
```

Import external Python package dependencies
```bash
cd ros2_ws/src
git submodule add git@github.com:robin-karlsson0/exodapt_robot_pt.git exodapt_robot_pt
cd ../../
```
**TODO: Git submodule `exodapt_robot_pt` needs pulling?**

Update external ROS 2 package dependencies
```bash
vcs pull ros2_ws/src
```

Update external Python package dependencies
```bash
git submodule update --remote
```

Install python package dependencies
```
uv pip install -r requirements.txt
```

Build ROS 2 packages
```
cd ros2_ws

colcon build --symlink-install

source install/setup.bash
```

# Running tests

```bash
colcon test --pytest-args="-s"
```

# How to use

Launch the Action Cycle Controller node with dependent packages using the launch file. Specify topics the State Manager will subscribe to and populate the state.

Example:
```bash
ros2 launch action_cycle_controller action_cycle_controller_launch.xml \
event_topics:='[/asr, /keyboard_input]' \
continuous_topics:='[/mllm]'
```

# robot_action_controller

### Topics

`/action_event`: `ActionManager` publishes a message when an action starts or completes to `/action_event` topic.
`/action_running`: `ActionManager` publishes currently running actions (possibly empty) to `/action_running` topic every time an action starts or ends. 
`/state`: `StateManager` publishes latest state to `/state` topic every time the state changes.

### Action servers

`action_decision_action_server`: `ActionDecisionActionServer` takes a state and returns the predicted optimal action key

# robot_action_decision

Service for predicting the optimal action to take based on the current state.

The created action service is named `llm_action_server_ad_8b_action` + `_action`.


### Valid actions message

A listing of all available actions and ongoing action cancellations the action decision inference module can choose to execute

```
action_key: Do|Cancel [action_name] action_description|cancel_description
```

Example:
```
a: Do [Idle Action] A no-operation action that represents a decision to remain idle.
b: Do [Reply Action] This action allows the robot to reply to user interactions by sending a response based on the current state information.
```

```
a: Do [Idle Action] A no-operation action that represents a decision to remain idle.
b: Cancel [Reply Action] Stops the current reply generation to the user, potentially freeing the robot to formulate a different reply or take another action.
```

### Running actions message

```
action_key: [action_name] running description
```

Example:
```
b: [Reply Action] The robot is currently replying to the user based on the state information when the reply action was initiated.
```

# How to Add Actions

1. Create a ROS 2 *action server node* that implements the action

2. Create an *action definition* by adding a module to the `actions.yaml` config (`ros2_ws/src/action_cycle_controller/config/actions.yaml`):

```yaml
  - module: "actions.ACTION_DEFINITION_FILENAME" 
    class: "ACTION_CLASS_NAME"
    action_server: "ACTION_SERVER_NAME"
    action_type: "ACTION_INTERFACE_IMPORT_PTH"
    action_key: 'FREE_SINGLE_TOKEN_CHARACTER_STRING'
```

Example:
```yaml
  - module: "actions.reply_action" 
    class: "ReplyAction"
    action_server: "reply_action_server"
    action_type: "exodapt_robot_interfaces.action.ReplyAction"
    action_key: 'b'
```

3. Implement *action module* `YOUR_ACTION.py` for the action server node (`ros2_ws/src/actions/src/actions/YOUR_ACTION.py`). The action module is a subclass of `BaseAction` and implements the following abstract methods:
- execute(): Execute the action by sending a goal to the action server node
- get_action_name(): Get the human-readable name of this action module
- get_action_description(): Returns a detailed description of the action's purpose and behavior
- get_running_description(): Returns a detailed description of what the action is currently doing
- get_cancel_description(): Returns a detailed description of consequences of canceling action

Example:
```python
from actions.base_action import BaseAction
from exodapt_robot_interfaces.action import ReplyAction as ReplyActionInterface


class ReplyAction(BaseAction):

    def __init__(self, node, action_manager, config):
        super().__init__(node, action_manager, config)

    def execute(self, state: str):
        goal = ReplyActionInterface.Goal()
        goal.state = state
        # Use action server name from config
        server_name = self.config.get('action_server', 'reply_action_server')
        self.action_manager.submit_action(server_name, goal)

    def get_action_name(self) -> str:
        return 'Reply action'

    def get_action_description(self) -> str:
        return 'This action allows the robot to reply to user interactions... '

    def get_running_description(self) -> str:
        return 'The robot is currently replying to the user based on the ... '

    def get_cancel_description(self) -> str:
        return 'Stops the current reply generation to the user, potentially ...'
```

4. Make sure the action server node is running when launching ActionCycleController

# TODO
- Why prompt templates are separated from implementation (e.g. `ReplyAction` <--> `action_reply_pt()`)