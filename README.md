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

**TODO: Git submodule `exodapt_robot_pt` needs pulling?**

Install python package dependencies
```
uv pip install -r requirements.txt
uv pip install -e ros2_ws/src/exodapt_robot_pt 
```

Build ROS 2 packages
```
cd ros2_ws

colcon build --symlink-install

source install/setup.bash
```


# robot_action_controller



# robot_action_decision

Service for predicting the optimal action to take based on the current state.

The created action service is named `llm_action_server_ad_8b_action` + `_action`.