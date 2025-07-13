from actions.base_action import BaseAction
from exodapt_robot_interfaces.action import \
    LongDummyAction as LongDummyActionInterface


class LongDummyAction(BaseAction):

    def __init__(self, node, action_manager, config):
        """Initialize the LongDummyAction plugin.

        Sets up the long dummy action plugin by calling the parent BaseAction
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
        goal = LongDummyActionInterface.Goal()
        # Use action server name from config
        server_name = self.config.get('action_server',
                                      'long_dummy_action_server')
        self.action_manager.submit_action(server_name, goal)

    def get_action_name(self) -> str:
        """Get the human-readable name of this action plugin.

        Returns:
            str: A descriptive name for this action, used for logging,
                debugging, and user interface display purposes.
        """
        return 'Long dummy action'

    def get_action_description(self) -> str:
        """Return a detailed description of the action's purpose and behavior.

        This method provides additional context about what the action does,
        its intended use cases, and any important notes for the decision making
        agent.

        Returns:
            str: Detailed description of the action's functionality and
                expected behavior
        """
        return 'This action allows the robot to test executing a long dummy action for system testing and debugging purposes.'  # noqa: E501

    def get_running_description(self) -> str:
        """Return a detailed description of what the action is currently doing.

        This method provides information about expected outcome of an ongoing
        action to aid the decision making agent to predict subsequent actions
        taking into account the ongoing action.

        Returns:
            str: Detailed description of the action's current activity and
                expected outcome.
        """
        return 'The robot is currently executing a long dummy action that facilitates system testing and debugging.'  # noqa: E501

    def get_cancel_description(self) -> str:
        """Return a detailed description of consequences of canceling action.

        This method provides information about what happens when the action is
        cancelled, including any side effects, partial completion states, or
        recovery behaviors that the decision making agent should be aware of.

        Returns:
            str: Detailed description of the cancellation consequences and
                expected behavior when the action is terminated
        """
        return 'Stops the currently running long dummy action execution.'
