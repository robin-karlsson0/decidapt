import os

import rclpy
from action_cycle_controller.action_manager import ActionManager
from ament_index_python.packages import get_package_share_directory
from exodapt_robot_interfaces.action import ActionDecision
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from .action_registry import ActionRegistry


class ActionCycleController(Node):
    """Modular action cycle controller for robotic systems using ROS 2.

    This node orchestrates a continuous action-decision loop at a fixed
    frequency, where the robot receives state information, requests action
    decisions from an external decision maker, and executes those decisions
    through a plugin-based action system.

    The controller operates as a pure orchestrator, delegating all action-
    specific logic to dynamically loaded plugins managed by an ActionRegistry.
    This design enables zero-code extension of robot capabilities through
    configuration files.

    Architecture:
        - Receives robot state updates via ROS topic subscription
        - Requests action decisions from external action server at fixed
          intervals
        - Validates and executes actions through plugin-based ActionRegistry
        - Publishes action events and status through ActionManager

    Key Features:
        - Plugin-based action system with runtime loading from YAML config
        - Configurable action cycle frequency
        - Automatic fallback to 'do nothing' action for invalid decisions
        - Integration with ActionManager for coordinated action execution
        - State-driven decision making with external action decision service

    Parameters:
        ac_freq (float): Action cycle frequency in Hz (default: 1.0)
        actions_config (str): Path to YAML configuration file for action plugins
                             (default: 'config/actions.yaml')

    Topics:
        Subscribed:
            - state (exodapt_robot_interfaces/srv/State): Current robot state
        Published:
            - action_event (std_msgs/String): Action execution events
            - action_running (std_msgs/String): Currently running action status

    Action Clients:
        - action_decision_action_server (exodapt_robot_interfaces/ActionDecision):
            Requests action decisions based on current state

    Example Usage:
        The node automatically starts the action cycle upon initialization.
        Actions are defined in the YAML configuration file and executed based
        on single-character decision strings (e.g., 'a' for do nothing, 'b' for
        reply).
    """  # noqa: E501

    def __init__(self):
        """Initialize the ActionCycleController node.

        Sets up the complete action cycle infrastructure including:
        - ROS 2 node parameters and timer for action cycle frequency
        - State subscription for receiving robot state updates
        - ActionManager for coordinated action execution and event publishing
        - ActionRegistry for dynamic plugin loading and management
        - ActionDecision client for requesting decisions from external services

        The initialization process loads action plugins from the configured
        YAML file and validates the action registry setup.
        """
        super().__init__('action_cycle_controller')

        self.declare_parameter('ac_freq', 1.0)
        self.declare_parameter('actions_config', 'config/actions.yaml')

        self.ac_loop_freq = float(self.get_parameter('ac_freq').value)
        action_config_relative = self.get_parameter('actions_config').value

        # Resolve the full path to the config file
        if os.path.isabs(action_config_relative):
            action_config_pth = action_config_relative
        else:
            try:
                package_share_dir = get_package_share_directory(
                    'action_cycle_controller')
                action_config_pth = os.path.join(package_share_dir,
                                                 action_config_relative)
            except Exception:
                # Fallback for development
                action_config_pth = action_config_relative

        # self.declare_parameter('num_short_chunks', 20)
        # self.num_short_chunks = self.get_parameter('num_short_chunks').value

        self.state = ''
        self.state_sub = self.create_subscription(
            String,
            'state',
            self.update_state_callback,
            10,
        )

        # Timer for action cycle [s]
        self.timer = self.create_timer(self.ac_loop_freq,
                                       self.run_action_cycle)

        # Used to prevent spamming state with high-freq. 'do nothing' actions
        self.prev_decision = None  # TODO Needed ???

        ####################
        #  Action manager
        ####################

        self.get_logger().info('Initializing ActionManager')
        self.action_event_pub = self.create_publisher(
            String,
            'action_event',
            10,
        )
        self.action_running_pub = self.create_publisher(
            String,
            'action_running',
            10,
        )

        self.action_manager = ActionManager(
            self,
            self.action_event_pub,
            self.action_running_pub,
        )

        #######################
        # #  Action registry
        # #####################

        self.get_logger().info('Initializing ActionRegistry')
        self.action_registry = ActionRegistry(self, self.action_manager)
        self.action_registry.load_from_config(action_config_pth)

        self.valid_actions_set = set(self.action_registry.get_valid_actions())
        self.valid_action_descr = self.action_registry.get_valid_action_descr()

        self.get_logger().info(
            f"Loaded {len(self.action_registry.get_valid_actions())} actions")

        #####################
        #  Action decision
        #####################

        self._ad_action_client = ActionClient(
            self,
            ActionDecision,
            'action_decision_action_server',
        )

    def run_action_cycle(self):
        """Execute a single iteration of the action cycle.

        This method implements the core action-decision loop:
        1. Captures the current robot state
        2. Requests an action decision from the external action decision server
        3. Initiates asynchronous processing of the decision response

        The method is called periodically by the ROS 2 timer at the configured
        frequency (ac_freq parameter). It sends the current state to the action
        decision server and sets up callbacks to handle the asynchronous
        response.

        The action cycle follows this sequence:
        Current State → Action Decision Request → Action Execution

        Note:
            This method is non-blocking and uses ROS 2 action client callbacks
            to handle the decision response asynchronously.
        """
        ad_goal = ActionDecision.Goal()
        ad_goal.state = self.state
        ad_goal.valid_actions = self.valid_action_descr

        self._ad_action_client.wait_for_server()
        self.ad_goal_future = self._ad_action_client.send_goal_async(ad_goal)
        self.ad_goal_future.add_done_callback(self.ad_response_callback)

    def ad_response_callback(self, ad_goal_future):
        """Handle the response from the action decision server goal submission.

        This callback is triggered when the action decision server responds to
        the goal submission request. It checks if the goal was accepted and
        sets up the next callback to handle the actual result.

        Args:
            ad_goal_future: Future object containing the goal handle response
                           from the action decision server

        Behavior:
            - If goal is accepted: Sets up result callback to wait for decision
            - If goal is rejected: Logs rejection and terminates the cycle

        Note:
            This is an intermediate callback in the action decision chain.
            The actual action decision result is handled by ad_result_callback.
        """
        ad_goal_handle = ad_goal_future.result()

        if not ad_goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.ad_result_future = ad_goal_handle.get_result_async()
        self.ad_result_future.add_done_callback(self.ad_result_callback)

    def ad_result_callback(self, ad_result_future):
        """Process the action decision result and initiate action execution.

        This callback handles the final result from the action decision server,
        extracts the predicted action from the response, and triggers the
        action execution through the execute_action method.

        Args:
            ad_result_future: Future object containing the action decision
                result with the predicted action string

        Behavior:
            - Extracts the action decision from the result
            - Logs the received action for debugging/monitoring
            - Delegates action execution to the execute_action method

        Note:
            This completes the asynchronous action decision request chain
            initiated by run_action_cycle().
        """
        ad_result = ad_result_future.result().result
        action_decision = ad_result.pred_action

        self.get_logger().info(f'Received action: {action_decision}')

        self.execute_action(action_decision)

    def execute_action(self, action_decision: str):
        """Execute the specified action through the action registry.

        This method validates the action decision, handles invalid actions with
        fallback behavior, and delegates execution to the appropriate action
        through the ActionRegistry.

        Args:
            action_decision (str): Single character or string representing the
                                 action to execute. Only the first character is
                                 used for decision mapping.

        Behavior:
            - Extracts first character from action decision string
            - Validates action against loaded plugin registry
            - Falls back to 'do nothing' action ('a') for invalid decisions
            - Routes valid actions to ActionRegistry for plugin execution
            - Logs execution status and errors
            - Updates previous decision tracker

        Action Validation:
            Invalid actions are automatically replaced with the default idle
            action ('a') to ensure system stability.
            # TODO: Read yaml config for default action

        Error Handling:
            Execution failures are logged as errors but do not interrupt the
            action cycle, allowing the system to continue operating.
        """
        # Get first character out of a potential sequence
        if len(action_decision) > 0:
            action_decision = action_decision[0]

        # Action validity check
        if action_decision not in self.valid_actions_set:
            self.get_logger().info(
                f'Invalid action received: {action_decision}')
            action_decision = 'a'
            self.get_logger().info(f'==> Do nothing: {action_decision}')

        # Execute action
        success = self.action_registry.execute_action(
            action_decision,
            self.state,
        )
        if not success:
            # TODO Replace with human-readable action name
            self.get_logger().error(
                f'Failed to execute action: {action_decision}')

        self.prev_decision = action_decision

    def update_state_callback(self, msg):
        """Update the robot's internal state representation.

        This callback function is triggered whenever a new state message is
        received on the 'state' topic. It updates the controller's internal
        state variable, which is then used in subsequent action decision
        requests.

        Args:
            msg: ROS message containing the new state data. The state
                information is extracted from msg.data and stored as a string.

        State Usage:
            The updated state is automatically included in the next action cycle
            when requesting decisions from the action decision server, enabling
            state-driven action selection.

        Note:
            This method maintains the most recent state information for use in
            the continuous action-decision loop.
        """
        self.state = msg.data


def main(args=None):
    """Main entry point for the ActionCycleController node.

    Initializes the ROS 2 system, creates an ActionCycleController instance,
    and runs the node until shutdown. This function handles the complete
    lifecycle of the action cycle controller.

    Args:
        args: Command line arguments passed to rclpy.init() (optional)

    Lifecycle:
        1. Initialize ROS 2 system
        2. Create ActionCycleController node instance
        3. Log startup message
        4. Enter ROS 2 spin loop to handle callbacks and timers
        5. Clean up resources on shutdown

    The node will continuously execute action cycles at the configured frequency
    until interrupted or shutdown.
    """
    rclpy.init(args=args)

    state_client = ActionCycleController()
    state_client.get_logger().info('Action Cycle Controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
