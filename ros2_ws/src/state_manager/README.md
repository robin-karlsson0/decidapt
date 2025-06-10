# State Manager

# How to use

Launch the state manager node by copy-pasting the following command in the terimal after specifying what topics to subscribe to.

Example:
```bash
ros2 launch state_manager state_manager_launch.xml \
event_topics:='[/asr, /keyboard_input]' \
continuous_topics:='[/mllm, /pose]'
```

The state will be populated by messages received from the topics.