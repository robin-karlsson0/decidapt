import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
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
        self.action_registry = {}  # action_name -> ActionClientConfig
        self.running_actions = {}  # action_name -> ActionState
        self.lock = threading.Lock()

        self.action_keys = {}  # action_name -> action_key

    def register_action(
        self,
        action_name,
        action_type,
        action_key: str,
        timeout=60.0,
        max_concurrent=1,
    ):
        """Register action with metadata."""
        config = ActionClientConfig(
            client=ActionClient(self.node, action_type, action_name),
            timeout=timeout,
            max_concurrent=max_concurrent,
        )
        self.action_registry[action_name] = config
        self.action_keys[action_name] = action_key

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
        with self.lock:
            # Catches 1) invalid action names and 2) capacity exceeded
            if not self._validate_action(action_name):
                return ActionResult.INVALID

            # TODO: Handle N concurrent actions
            if self.is_running(action_name):
                self.cancel_action(action_name)

            # Stores particular action state
            expiry_t = time.time() + self.action_registry[action_name].timeout
            state = ActionState(
                name=action_name,
                goal=goal,
                callback=callback,
                timeout=expiry_t,
            )
            self.running_actions[action_name] = state

        # Execute action
        return self._execute_action(state)

    def _execute_action(self, state: ActionState) -> ActionResult:
        """Execute action with error handling.

        Args:
            state: Action state object

        Returns:
            ActionResult: Resulting state of action execution.
        """
        try:
            # Unpack ActionClient for current action
            client_config = self.action_registry[state.name]
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
                self.publish_action_event_msg(action_name,
                                              ActionResult.SUBMITTED)
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

    def cancel_action(self, action_name) -> bool:
        """Cancel running action.

        Args:
            action_name (str): Name of the action to cancel.

        Returns:
            bool: True if action was canceled, False if not running.
        """
        # Get state info while locked
        with self.lock:
            if action_name not in self.running_actions:
                return False
            state = self.running_actions[action_name]

        # Cancel outside the lock
        if hasattr(state, 'goal_handle') and state.goal_handle:
            state.goal_handle.cancel_goal_async()
        elif state.future:
            state.future.cancel()

        # Complete the action (this will acquire its own lock)
        self._complete_action(action_name, ActionResult.CANCELED)
        return True

    def _complete_action(self, action_name, result, data=None):
        """Clean up and notify completion."""
        with self.lock:
            if action_name not in self.running_actions:
                return

            state = self.running_actions[action_name]
            del self.running_actions[action_name]

        # Update action status when action is completed
        self.publish_action_event_msg(action_name, result)
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
            return False

        config = self.action_registry[action_name]
        running_count = sum(1 for s in self.running_actions.values()
                            if s.name == action_name)

        return running_count < config.max_concurrent

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
        """Create a status message for the current running actions."""
        status_msg = 'Running actions:\n'
        with self.lock:
            for action_name in self.running_actions.keys():
                action_key = self.action_keys[action_name]
                status_msg += f'{action_name} (action_key \'{action_key}\')'
        return status_msg
