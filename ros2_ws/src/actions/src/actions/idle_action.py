from actions.base_action import BaseAction
from std_msgs.msg import String


class IdleAction(BaseAction):
    """A no-operation action that represents a decision to remain idle.

    This action plugin implements a "do nothing" behavior within the modular
    action cycle architecture. When executed, it publishes a single
    informational message indicating the robot has chosen to take no new action,
    then remains silent on subsequent executions to prevent log spam.

    The IdleAction is typically used as a default or fallback action when the
    robot's decision-making system determines that no active intervention is
    required, allowing the system to maintain its control cycle without
    performing unnecessary operations.

    Attributes:
        prev_execution (bool): Flag to track if the action has been executed
            before, used to prevent repetitive logging of idle state messages.

    Example:
        The action is automatically loaded via the plugin system based on YAML
            configuration:

        ```yaml
        actions:
          - module: "robot_actions.idle_action"
            class_name: "IdleAction"
            action_key: "i"
        ```

    Note:
        This action does not require any action servers or external
        dependencies, making it the simplest possible action plugin
        implementation."""

    def __init__(self, node, action_manager, config):
        """Initialize the IdleAction plugin.

        Args:
            node: The ROS 2 node instance that owns this action plugin,
                providing access to ROS communication primitives and publishers.
            action_manager: The ActionManager instance responsible for
                coordinating action execution across the system (unused by
                IdleAction).
            config (dict): Configuration dictionary containing plugin-specific
                settings loaded from YAML (unused by IdleAction but required
                for interface compatibility).

        Note:
            The config parameter is not used by IdleAction but is required to
            maintain consistency with the BaseAction interface.
        """
        super().__init__(node, action_manager, config)
        self.prev_execution = False

    def execute(self, state: str):
        """Execute the idle action behavior.

        Publishes an informational message on the first execution indicating
        that the robot has decided to take no action. Subsequent calls are
        silent to prevent excessive logging while maintaining the action cycle.

        Args:
            state (str): The current state or context information from the
                robot's decision-making system (unused by IdleAction but
                required for interface compatibility).

        Behavior:
            - First execution: Publishes idle action message to
                action_event_pub topic
            - Subsequent executions: Silent operation (no message published)

        Note:
            The state parameter is not utilized by this action but is required
            to maintain consistency with the BaseAction interface.
        """
        # Prevent spam logging
        if self.prev_execution:
            return

        msg = String()
        msg.data = "Idle actions: Robot decides to take no new action."
        self.node.action_event_pub.publish(msg)
        self.prev_execution = True

    def get_action_name(self) -> str:
        """Get the human-readable name of this action plugin.

        Returns:
            str: A descriptive name for this action, used for logging,
                debugging, and user interface display purposes.
        """
        return 'Idle action'

    def get_action_description(self) -> str:
        """Return a detailed description of the action's purpose and behavior.

        This method provides additional context about what the action does,
        its intended use cases, and any important notes for the decision making
        agent.

        Returns:
            str: Detailed description of the action's functionality and
                expected behavior
        """
        return 'A no-operation action that represents a decision to remain idle.'  # noqa: E501
