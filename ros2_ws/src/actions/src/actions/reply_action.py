from actions.base_action import BaseAction
from exodapt_robot_interfaces.action import ReplyAction as ReplyActionInterface


class ReplyAction(BaseAction):
    """Reply action plugin that enables a robot to respond to user interactions.

    This plugin implements a reply functionality where the robot can acknowledge
    or respond to user states through the ActionManager system. It serves as an
    example implementation of the BaseAction interface, demonstrating how to
    create self-contained action plugins that integrate with the modular
    ActionCycleController architecture.

    The plugin submits ReplyAction goals to a configurable action server,
    allowing the robot to provide contextual responses based on the current
    state information.

    Configuration:
        The plugin expects an optional 'action_server' parameter in its config
        dictionary to specify the target action server name. If not provided,
        defaults to 'reply_action_server'.

    Example YAML configuration:
        ```yaml
        actions:
          module: "actions.reply_action"
          class: "ReplyAction"
          action_server: "reply_action_server"
          action_type: "exodapt_robot_interfaces.action.ReplyAction"
          action_key: 'b'
    """

    def __init__(self, node, action_manager, config):
        """Initialize the ReplyAction plugin.

        Sets up the reply action plugin by calling the parent BaseAction
        initializer, which handles common plugin setup including action
        server registration if specified in the configuration.

        Args:
            node: The ROS 2 node instance that owns this plugin
            action_manager: ActionManager instance for submitting action goals
            config (dict): Plugin configuration dictionary containing optional
                'action_server' specification and other parameters
        """
        super().__init__(node, action_manager, config)

    def execute(self, state: str):
        """Execute the reply action with the given state information.

        Creates and submits an ReplyAction goal to the configured action server
        through the ActionManager. The goal contains the current state string
        which can be used by the action server to generate an appropriate
        response.

        Args:
            state (str): Current state information to include in the reply goal.
                This could represent user input, system status, or any
                contextual information relevant to the reply action.

        Note:
            This method is non-blocking. The ActionManager handles the
            asynchronous submission and monitoring of the action goal.
        """
        goal = ReplyActionInterface.Goal()
        goal.state = state
        # Use action server name from config
        server_name = self.config.get('action_server', 'reply_action_server')
        self.action_manager.submit_action(server_name, goal)

    def get_action_name(self) -> str:
        """Get the human-readable name of this action plugin.

        Returns:
            str: A descriptive name for this action, used for logging,
                debugging, and user interface display purposes.
        """
        return 'Reply action'

    def get_action_description(self) -> str:
        """Return a detailed description of the action's purpose and behavior.

        This method provides additional context about what the action does,
        its intended use cases, and any important notes for the decision making
        agent.

        Returns:
            str: Detailed description of the action's functionality and
                expected behavior
        """
        return 'This action allows the robot to reply to user interactions by sending a response based on the current state information.'  # noqa: E501

    def get_running_description(self) -> str:
        """Return a detailed description of what the action is currently doing.

        This method provides information about expected outcome of an ongoing
        action to aid the decision making agent to predict subsequent actions
        taking into account the ongoing action.

        Returns:
            str: Detailed description of the action's current activity and
                expected outcome.
        """
        return 'The robot is currently replying to the user based on the state information when the reply action was initiated.'  # noqa: E501

    def get_cancel_description(self) -> str:
        """Return a detailed description of consequences of canceling action.

        This method provides information about what happens when the action is
        cancelled, including any side effects, partial completion states, or
        recovery behaviors that the decision making agent should be aware of.

        Returns:
            str: Detailed description of the cancellation consequences and
                expected behavior when the action is terminated
        """
        return 'Stops the current reply generation to the user, potentially freeing the robot to formulate a different reply or take another action.'  # noqa: E501
