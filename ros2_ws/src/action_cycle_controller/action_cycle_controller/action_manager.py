import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any, Callable

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from std_msgs.msg import String


@dataclass
class ActionClientConfig:
    """Object representing an action type."""

    client: ActionClient
    timeout: float
    max_concurrent: int


@dataclass
class ActionState:
    """Object representing the ongoing execution of an action."""

    name: str
    goal: Any
    callback: Callable
    timeout: float
    future: Any = field(default=None)
    goal_handle: Any = field(default=None)


class ActionResult(StrEnum):
    """Enumeration of posible action result cases."""

    SUBMITTED = 'submitted'
    SUCCESS = 'success'
    FAILED = 'failed'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    INVALID = 'invalid'


class ActionManager:
    """Thread-safe manager for executing multiple ROS 2 actions in parallel.

    Provides registration, execution, cancellation, and monitoring of actions
    with automatic timeout handling, error recovery, and completion callbacks.

    Usage:
        1. Create manager with ROS 2 node:
            manager = ActionManager(node)
        2. Register actions:
            manager.register_action("move", MoveAction, timeout=30.0)
        3. Submit actions:
            manager.submit_action("move", goal, callback_func)
        4. Monitor:
            manager.is_action_running("move")
        5. Cancel:
            manager.cancel_action("move")
        6. Cleanup timeouts periodically:
            manager.cleanup_timeouts()

    Features:
        - Concurrent execution with configurable limits per action type
        - Automatic cleanup of completed/failed/canceled actions
        - Thread-safe operations for multi-threaded environments
        - Timeout enforcement with periodic cleanup
        - Callback system for completion notifications
        - Error handling with graceful degradation

    Managed objects:
        - ActionClientConfig: Configuration for each action type.
        - ActionState: A record of a particular action execution.

    Async chain:

        _execute_action:
            Async call (1):   client.send_goal_async(goal) --> goal_future
                              goal_future.add_done_callback()

        _handle_goal_response:
            Async result (1): goal_future.result() --> goal_handle
            Async call (2):   goal_handle.get_result_async() --> result_future
                              result_future.add_done_callback()

        _handle_result:
            Async result (2) result_future.result() --> result

    Async objects:
            - goal_future
            - goal_handle
            - result_future
            - result

    Publishes action-related messages when:
        1. An action is accepted: _handle_goal_response()
        1. An action completes: _complete_action()

    Results:
        - invalid: Action not registered or capacity exceeded
    """

    def __init__(self, node, action_event_pub=None, action_running_pub=None):
        """Initialize ActionManager with ROS 2 node.

        Args:
            node: ROS 2 node instance for creating ActionClient objects
                NOTE: The node is needed for providing ROS 2 communication
                infrastructure to ActionClient instances.
            action_event_pub: Optional ROS publisher for action events like
                submission, completion, or failure.
            action_running_pub: Optional ROS publisher for current running
                actions status.

        Note:
            Actions must be registered before use via register_action().
            Call cleanup_timeouts() periodically (e.g., in timer callback)
            to handle expired actions.
        """
        self.node = node
        self.action_event_pub = action_event_pub
        self.action_running_pub = action_running_pub
        self.action_registry = {}  # action_server_name -> ActionClientConfig
        self.running_actions = {}  # action_server_name -> ActionState
        self.lock = threading.Lock()

        # Dicts mapping Action Server information
        self.action_keys = {}  # action_server_name -> key
        self.action_names = {}  # action_server_name --> name
        self.action_descriptions = {}  # action_server_name --> description
        self.running_descriptions = {}  # action_server_name --> running descr.
        self.cancel_descriptions = {}  # action_server_name --> cancel descr.

    def register_action(
        self,
        action_server_name,
        action_type,
        action_key: str,
        action_name: str = '',
        action_description: str = '',
        running_description: str = '',
        cancel_description: str = '',
        timeout=60.0,
        max_concurrent=1,
    ):
        """Register action with metadata."""
        config = ActionClientConfig(
            client=ActionClient(self.node, action_type, action_server_name),
            timeout=timeout,
            max_concurrent=max_concurrent,
        )
        self.action_registry[action_server_name] = config
        self.action_keys[action_server_name] = action_key
        self.action_names[action_server_name] = action_name
        self.action_descriptions[action_server_name] = action_description
        self.running_descriptions[action_server_name] = running_description
        self.cancel_descriptions[action_server_name] = cancel_description

    def register_virtual_action(
        self,
        virtual_server_name: str,
        action_key: str,
        action_name: str = '',
        action_description: str = '',
        running_description: str = '',
        cancel_description: str = '',
    ):
        """Register a virtual action that doesn't need a real action server.

        This method allows actions like IdleAction to be registered in the
        ActionManager without requiring a real ROS 2 action server. Virtual
        actions are included in the valid actions list and can be executed
        directly without going through the action server infrastructure.

        Args:
            virtual_server_name (str): Unique name for the virtual action server
            action_key (str): Single character key for the action
            action_name (str): Human-readable name for the action
            action_description (str): Description of what the action does
            running_description (str): Description of what happens when running
            cancel_description (str): Description of what happens when canceled
        """
        # Store metadata without creating an actual ActionClient
        self.action_keys[virtual_server_name] = action_key
        self.action_names[virtual_server_name] = action_name
        self.action_descriptions[virtual_server_name] = action_description
        self.running_descriptions[virtual_server_name] = running_description
        self.cancel_descriptions[virtual_server_name] = cancel_description

        # Mark this as a virtual action in the registry
        # We don't create an ActionClientConfig since no real client is needed
        self.action_registry[virtual_server_name] = None

    def submit_action(self, action_name, goal, callback=None) -> ActionResult:
        """Queue and execute action with validation.

        Prepares a valid action for execution with an ActionState object to
        track the action's state.

        Thread lock prevents concurrent operations to create race conditions.
        A thread will wait for the lock to be released before proceeding.

        Args:
            action_name (str): Name of the action to execute.
            goal (object): Goal message for the action.
            callback (function): Optional callback function to handle result.

        Returns:
            ActionResult: Resulting state of action execution.
        """
        cancel_action = False
        with self.lock:
            # Catches 1) invalid action names and 2) capacity exceeded
            if not self._validate_action(action_name):
                return ActionResult.INVALID

            # TODO: Handle N concurrent actions
            if self.is_running(action_name):
                cancel_action = True

        if cancel_action:
            self.node.get_logger().info(f'Cancelling action: {action_name}')
            return self.cancel_action(action_name)

        with self.lock:
            # Re-check in case state changed between lock releases
            if self.is_running(action_name):
                self.node.get_logger().warning(
                    f'Action "{action_name}" started by another thread. Aborting submission.'  # noqa E501
                )
                return ActionResult.INVALID  # Or some other appropriate status

            # Stores particular action state
            config = self.action_registry[action_name]
            # Handle virtual actions (config is None) with default timeout
            timeout = config.timeout if config is not None else 60.0
            expiry_t = time.time() + timeout
            state = ActionState(
                name=action_name,
                goal=goal,
                callback=callback,
                timeout=expiry_t,
            )
            self.running_actions[action_name] = state

        # Execute action
        self.node.get_logger().info(f'Executing action: {action_name}')
        return self._execute_action(state)

    def _execute_action(self, state: ActionState) -> ActionResult:
        """Execute action with error handling.

        Args:
            state: Action state object

        Returns:
            ActionResult: Resulting state of action execution.
        """
        try:
            # Get the client configuration
            client_config = self.action_registry[state.name]

            # Handle virtual actions (config is None)
            if client_config is None:
                # Virtual actions complete immediately
                self._complete_action(state.name, ActionResult.SUCCESS, None)
                return ActionResult.SUCCESS

            # Unpack ActionClient for current action
            client = client_config.client

            # Asynchronously execute action through ActionClient
            goal_future = client.send_goal_async(state.goal)

            # Attach a callback to handle goal acceptance/rejection.
            # NOTE: The callback receives the future object as its argument
            #       (here named 'f'). A lambda is used to pass both 'state.name'
            #       and the future to the _handle_goal_response method,
            #       since add_done_callback only passes the future by default.
            goal_future.add_done_callback(
                lambda f: self._handle_goal_response(state.name, f))
            state.future = goal_future
            return ActionResult.SUBMITTED

        except Exception as e:
            self._complete_action(state.name, ActionResult.FAILED, str(e))
            return ActionResult.FAILED

    def _handle_goal_response(self, action_name, goal_future):
        """Handle goal acceptance/rejection."""
        try:
            goal_handle = goal_future.result()
            if action_name in self.running_actions:
                self.running_actions[action_name].goal_handle = goal_handle

            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(
                    lambda f: self._handle_result(action_name, f))

                # Update action status when action is accepted
                # self.publish_action_event_msg(action_name,
                #                               ActionResult.SUBMITTED)
                self.publish_running_actions_msg()
            else:
                self._complete_action(action_name, ActionResult.REJECTED)
        except Exception as e:
            self._complete_action(action_name, ActionResult.FAILED, str(e))

    def _handle_result(self, action_name, result_future):
        """Handle action completion."""
        try:
            result = result_future.result()
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                status = ActionResult.SUCCESS
            else:
                status = ActionResult.FAILED
            self._complete_action(action_name, status, result)
        except Exception as e:
            self._complete_action(action_name, ActionResult.FAILED, str(e))

    def cancel_action(self, action_name: str) -> ActionResult:
        """Cancel running action.

        Args:
            action_name: Name of the action to cancel.

        Returns:

        """
        # Get action state info while locked
        self.node.get_logger().info('Entered cancel_action')
        with self.lock:
            if action_name not in self.running_actions:
                self.node.get_logger().warning(
                    f'Trying to cancel not running action: {action_name}')
                return ActionResult.INVALID
            state = self.running_actions[action_name]
        self.node.get_logger().info('After lock')

        # Cancel outside the lock
        if hasattr(state, 'goal_handle') and state.goal_handle:
            self.node.get_logger().info('Sending cancel request')
            future = state.goal_handle.cancel_goal_async()
        else:
            raise Exception('Action state is missing a "goal_handle" object')
        # elif state.future:
        #     state.future.cancel()

        self.node.get_logger().info('Adding callback function')
        callback = partial(self.cancel_done, action_name=action_name)
        future.add_done_callback(callback)

        # Complete the action (this will acquire its own lock)
        # self._complete_action(action_name, ActionResult.CANCELED)
        return ActionResult.CANCELED

    def cancel_done(self, future, action_name: str):
        """
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) == 0:
            self.node.get_logger().warning(
                f'Failed to cancel action: {action_name}')
            return

        self.node.get_logger().info(
            f'Successfully cancelled action: {action_name}')
        self._complete_action(action_name, ActionResult.CANCELED)

    def _complete_action(self, action_name, result, data=None):
        """Clean up and notify completion."""
        with self.lock:
            if action_name not in self.running_actions:
                return

            state = self.running_actions[action_name]
            del self.running_actions[action_name]

        # Update action status when action is completed
        self.publish_running_actions_msg()

        # Execute callback outside lock
        if state.callback:
            try:
                state.callback(action_name, result, data)
            except Exception as e:
                self.node.get_logger().error(
                    f"Callback error for {action_name}: {e}")

    def _validate_action(self, action_name):
        """Validate action can be executed."""
        if action_name not in self.action_registry:
            self.node.get_logger().warning(
                f'Given action name "{action_name}" not in registry!')
            return False

        config = self.action_registry[action_name]

        # Handle virtual actions (config is None)
        if config is None:
            # Virtual actions can always be executed (no concurrency limits)
            return True

        # TODO: Separate concurrent check and action cancellation?
        # running_count = sum(1 for s in self.running_actions.values()
        #                     if s.name == action_name)
        # return running_count < config.max_concurrent
        return True

    def is_running(self, action_name):
        """Check if existing action should be canceled."""
        return action_name in self.running_actions

    def get_running_actions(self):
        """Get list of currently running actions."""
        with self.lock:
            return list(self.running_actions.keys())

    def is_action_running(self, action_name):
        """Check if specific action is running."""
        with self.lock:
            return action_name in self.running_actions

    def cleanup_timeouts(self):
        """Remove timed out actions (call periodically)."""
        current_time = time.time()
        expired = []

        with self.lock:
            for name, state in self.running_actions.items():
                if current_time > state.timeout:
                    expired.append(name)

        for name in expired:
            self.cancel_action(name)

    def publish_action_event_msg(self, action_name,
                                 action_result: ActionResult):
        """Publish action event message to the action_event_pub."""
        if self.action_event_pub:
            status_msg = f'Action "{action_name}" is {action_result.value}'
            msg = String()
            msg.data = status_msg
            self.action_event_pub.publish(msg)

    def publish_running_actions_msg(self):
        """Publish current action status to the action_running_pub."""
        if self.action_running_pub:
            running_actions_msg = self.create_running_action_msg()
            msg = String()
            msg.data = running_actions_msg
            self.action_running_pub.publish(msg)

    def create_running_action_msg(self) -> str:
        """Create a status message for the current running actions.
        
        Message format
            action_key: [action_name] running_description
        
        Message format example:
            b: [Reply action] Robot is currently generating a reply to respond to user interaction
        
        """  # noqa E501
        msg = []
        with self.lock:
            # Sort running actions alphabetically by action_key
            sorted_actions = sorted(self.running_actions.keys(),
                                    key=lambda action_server_name:
                                    (self.action_keys[action_server_name]))

            for action_server_name in sorted_actions:
                # Extract action information
                action_name = self.action_names[action_server_name]
                action_key = self.action_keys[action_server_name]
                running_descr = self.running_descriptions[action_server_name]
                # Append running action string
                msg.append(f'{action_key}: [{action_name}] {running_descr}')

        msg = '\n'.join(msg)
        return msg

    def create_valid_actions_msg(self) -> str:
        """Returns a formated list of valid do and cancel actions.

        Message format:
            action_key: Do|Cancel [action_name] action_descr|cancel_descr

        Message format example 1:
            a: Do [Idle action] A no-operation action that represents a decision to remain idle.
            b: Do [Reply action] This action allows the robot to reply to user interactions by sending a response based on the current state information.

        Message format example 2:
            a: Do [Idle action] A no-operation action that represents a decision to remain idle.
            b: Cancel [Reply action] Canceling will stop the current reply generation to the user, potentially freeing the robot to formulate a different reply or take another action.
        """  # noqa E501
        msg = []

        # Sort valid actions alphabetically by action_key
        sorted_names = sorted(self.action_registry.keys(),
                              key=lambda action_server_name:
                              (self.action_keys[action_server_name]))

        with self.lock:
            for action_server_name in sorted_names:
                action_key = self.action_keys[action_server_name]
                action_name = self.action_names[action_server_name]

                # Format string as `Do` or `Cancel` action
                is_running = action_server_name in self.running_actions.keys()
                if is_running:
                    action_type = 'Cancel'
                    descr = self.cancel_descriptions[action_server_name]
                else:
                    action_type = 'Do'
                    descr = self.action_descriptions[action_server_name]

                msg.append(
                    f'{action_key}: {action_type} [{action_name}] {descr}')

        msg = '\n'.join(msg)
        return msg
