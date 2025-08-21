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
        """Register a ROS 2 action with the ActionManager for execution.

        This method creates and registers an ActionClient for the specified
        action type, enabling the ActionManager to submit goals and manage
        the lifecycle of the action. The registration includes metadata for
        human-readable descriptions and execution parameters.

        Args:
            action_server_name (str): Unique identifier for the action server.
                                    Used internally for action management.
            action_type: ROS 2 action type class defining the action interface.
                        Must be a valid ROS 2 action definition.
            action_key (str): Single character key for command mapping.
                             Used by decision systems to trigger actions.
            action_name (str): Human-readable name displayed in status messages.
            action_description (str): Description shown when action is available
                                    for execution ("Do" actions).
            running_description (str): Description shown when action is
                                     currently executing.
            cancel_description (str): Description shown when action can be
                                    canceled ("Cancel" actions).
            timeout (float): Maximum execution time in seconds before automatic
                           cancellation (default: 60.0).
            max_concurrent (int): Maximum number of concurrent instances of
                                this action type (default: 1).

        Registration Process:
            1. Creates ActionClient with specified action type and server name
            2. Stores configuration in action registry with timeout and limits
            3. Associates metadata (key, name, descriptions) with action
            4. Enables action for submission via submit_action()

        Note:
            Actions must be registered before they can be executed. The action
            server must be running for successful goal submission.
        """
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
        """Register virtual action executing without a ROS 2 action server.

        Virtual actions enable the ActionManager to handle actions that don't
        require actual ROS 2 action server communication, such as idle actions.
        They complete instantly upon execution and are included in the valid
        actions list for decision systems.

        Args:
            virtual_server_name (str): Unique identifier for the virtual action.
                Used internally for action management.
            action_key (str): Single character key for command mapping. Used by
                decision systems to trigger actions.
            action_name (str): Human-readable name displayed in status messages.
            action_description (str): Description shown when action is available
                for execution ("Do" actions).
            running_description (str): Description shown when action is
                currently executing (brief duration).
            cancel_description (str): Description shown when action can be
                canceled (usually not applicable).

        Registration Process:
            1. Stores action metadata without creating ActionClient
            2. Marks action as virtual in registry (config = None)
            3. Enables immediate execution through submit_action()
            4. Includes action in valid actions list for decision systems

        Use Cases:
            - Idle/no-operation actions that represent deliberate inaction
            - Immediate state changes that don't require external services
            - Placeholder actions for testing or configuration purposes

        Note:
            Virtual actions complete immediately with SUCCESS result when
            executed via submit_action().
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
        self.node.get_logger().debug(f'Executing action: {action_name}')
        return self._execute_action(state)

    def _execute_action(self, state: ActionState) -> ActionResult:
        """Execute action with error handling and asynchronous processing.

        This method initiates the actual execution of an action by sending
        the goal to the appropriate ActionClient. It handles both real ROS 2
        actions and virtual actions, setting up the asynchronous callback
        chain for goal processing.

        Args:
            state (ActionState): Action state object containing execution
                details including name, goal, callback, timeout.

        Returns:
            ActionResult: Immediate execution status:
                         - SUBMITTED: Real action goal sent to server
                         - SUCCESS: Virtual action completed immediately
                         - FAILED: Error occurred during goal submission

        Execution Flow:
            For Real Actions:
            1. Retrieves ActionClient from registry configuration
            2. Sends goal asynchronously via send_goal_async()
            3. Attaches callback to handle goal acceptance/rejection
            4. Stores future reference in action state
            5. Returns SUBMITTED status

            For Virtual Actions:
            1. Detects virtual action (config is None)
            2. Immediately completes action with SUCCESS
            3. Triggers completion callback
            4. Returns SUCCESS status

        Callback Chain:
            Real actions initiate an asynchronous chain:
            goal_future → _handle_goal_response → _handle_result →
            _complete_action

        Error Handling:
            Any exceptions during goal submission result in FAILED status
            and immediate action completion with error details.

        Note:
            This method is called internally by submit_action() after
            validation and state setup are complete.
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
        """Handle goal acceptance or rejection from the action server.

        This callback is triggered when the action server responds to a goal
        submission request. It processes the goal handle and sets up the next
        phase of the asynchronous action execution chain.

        Args:
            action_name (str): Name of the action being processed.
            goal_future: Future object containing the goal handle response
                from the action server.

        Behavior:
            Goal Accepted:
            1. Stores goal handle in action state for cancellation capability
            2. Requests result asynchronously via get_result_async()
            3. Attaches result callback to handle completion
            4. Updates running actions status via publisher

            Goal Rejected:
            1. Completes action with REJECTED status
            2. Cleans up action state
            3. Triggers completion callbacks

        Error Handling:
            Any exceptions during goal response processing result in action
            completion with FAILED status and error details logged.

        Asynchronous Chain:
            This is the second step in the action execution chain:
            submit_action → _execute_action → _handle_goal_response →
            _handle_result → _complete_action

        Note:
            This method is called automatically by the ROS 2 action client
            infrastructure when the action server responds to goal submission.
        """
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
        """Process action completion result and finalize execution.

        This callback handles the final result from the action server after
        goal execution is complete. It interprets the goal status and
        triggers action completion with appropriate result classification.

        Args:
            action_name (str): Name of the action that completed.
            result_future: Future object containing the final action result
                with status and result data.

        Result Processing:
            1. Extracts result from future object
            2. Maps GoalStatus to ActionResult:
               - STATUS_SUCCEEDED → SUCCESS
               - All other statuses → FAILED
            3. Triggers completion cleanup and callbacks

        Error Handling:
            Any exceptions during result processing result in action
            completion with FAILED status and error details logged.

        Completion:
            Delegates to _complete_action() for:
            - Action state cleanup
            - Status message publishing
            - User callback execution

        Asynchronous Chain:
            This is the final step in the action execution chain:
            submit_action → _execute_action → _handle_goal_response →
            _handle_result → _complete_action

        Note:
            This method is called automatically by the ROS 2 action client
            infrastructure when the action completes execution.
        """
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
        """Cancel a currently running action with graceful cleanup.

        This method initiates cancellation of an active action by sending
        a cancel request to the action server. It handles the asynchronous
        cancellation process and ensures proper cleanup of action state.

        Args:
            action_name (str): Name of the action to cancel. Must be currently
                             running in the action manager.

        Returns:
            ActionResult:
                - CANCELED: Cancellation request sent successfully
                - INVALID: Action not found or not currently running

        Cancellation Process:
            1. Validates action is currently running
            2. Retrieves action state under thread lock
            3. Sends cancel request via goal_handle.cancel_goal_async()
            4. Attaches callback to handle cancellation confirmation
            5. Returns CANCELED status immediately

        Error Handling:
            - Missing action: Logs warning and returns INVALID
            - Missing goal_handle: Raises exception for debugging
            - Async cancellation failures: Handled in cancel_done callback

        Thread Safety:
            Uses thread lock to safely access running actions state and
            prevent race conditions during concurrent operations.

        Asynchronous Completion:
            The cancel_done callback handles the final cleanup when the
            action server confirms cancellation.

        Note:
            This method returns immediately after sending the cancel request.
            Actual action cleanup occurs asynchronously in the cancel_done
            callback.
        """
        # Get action state info while locked
        with self.lock:
            if action_name not in self.running_actions:
                self.node.get_logger().warning(
                    f'Trying to cancel not running action: {action_name}')
                return ActionResult.INVALID
            state = self.running_actions[action_name]

        # Cancel outside the lock
        if hasattr(state, 'goal_handle') and state.goal_handle:
            self.node.get_logger().info('Sending cancel request')
            future = state.goal_handle.cancel_goal_async()
        else:
            raise Exception('Action state is missing a "goal_handle" object')
        # elif state.future:
        #     state.future.cancel()

        callback = partial(self.cancel_done, action_name=action_name)
        future.add_done_callback(callback)

        # Complete the action (this will acquire its own lock)
        # self._complete_action(action_name, ActionResult.CANCELED)
        return ActionResult.CANCELED

    def cancel_done(self, future, action_name: str):
        """Handle cancellation confirmation from the action server.

        This callback processes the response from the action server's
        cancellation request, confirming whether the action was successfully
        canceled and triggering final cleanup.

        Args:
            future: Future object containing the cancellation response from
                   the action server.
            action_name (str): Name of the action that was canceled.

        Cancellation Response Processing:
            Success (goals_canceling > 0):
            1. Logs successful cancellation
            2. Triggers _complete_action with CANCELED result
            3. Cleans up action state and notifies callbacks

            Failure (goals_canceling == 0):
            1. Logs warning about cancellation failure
            2. Action remains in running state
            3. May require manual intervention or timeout cleanup

        Error Handling:
            This method focuses on processing the server response. Any
            exceptions would be logged by the ROS 2 action client
            infrastructure.

        Final Cleanup:
            Successful cancellation triggers _complete_action() which:
            - Removes action from running_actions
            - Updates status publishers
            - Executes user-provided completion callbacks

        Note:
            This callback is automatically invoked by the ROS 2 action
            client when the cancellation request receives a response.
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) == 0:
            self.node.get_logger().warning(
                f'Failed to cancel action: {action_name}')
            return

        self.node.get_logger().debug(
            f'Successfully cancelled action: {action_name}')
        self._complete_action(action_name, ActionResult.CANCELED)

    def _complete_action(self, action_name, result, data=None):
        """Clean up completed action and execute completion callbacks.

        This method handles the final cleanup phase for any action that has
        finished executing, whether successfully, with failure, or through
        cancellation. It ensures proper resource cleanup and notification.

        Args:
            action_name (str): Name of the action being completed.
            result (ActionResult): Final result status of the action execution.
            data: Optional result data from the action execution, typically
                 containing action-specific output or error information.

        Cleanup Operations:
            1. Thread-safe removal from running_actions registry
            2. Retrieval of action state for callback execution
            3. Publishing updated running actions status
            4. Execution of user-provided completion callbacks

        Thread Safety:
            Uses thread lock to safely modify running_actions registry and
            prevent race conditions during concurrent action operations.

        Callback Execution:
            User callbacks are executed outside the thread lock to prevent
            deadlocks and reduce lock contention. Callback exceptions are
            caught and logged without affecting action manager stability.

        Status Publishing:
            Updates action status publishers to reflect the removal of the
            completed action from the running actions list.

        Error Handling:
            - Missing action: Silently returns (already cleaned up)
            - Callback exceptions: Logged as errors but don't interrupt cleanup

        Note:
            This method is called by all action completion paths:
            successful completion, failures, cancellations, and timeouts.
        """
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
        """Validate that an action can be executed by the ActionManager.

        This method performs validation checks to ensure that a requested
        action is properly registered and can be executed within the current
        system constraints.

        Args:
            action_name (str): Name of the action to validate for execution.

        Returns:
            bool: True if action can be executed, False otherwise.

        Validation Checks:
            1. Action Registration: Verifies action exists in action_registry
            2. Concurrency Limits: Checks if max_concurrent limit allows
               execution
            3. Virtual Action Support: Handles virtual actions (config=None)

        Action Types:
            Real Actions:
            - Must have valid ActionClientConfig in registry
            - Subject to concurrency limits (currently bypassed)
            - Require active action server for execution

            Virtual Actions:
            - Have None config in registry (no ActionClient needed)
            - No concurrency limits (can always execute)
            - Complete immediately without server communication

        Current Implementation:
            - Registration check: Enforced with warning logging
            - Concurrency check: Always returns True (TODO: implement limits)
            - Virtual action support: Fully implemented

        Logging:
            Invalid action names are logged as warnings to help with
            debugging action configuration issues.

        Note:
            This method is called by submit_action() before action execution
            to prevent invalid actions from entering the execution pipeline.
        """
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
        """Check if a specific action is currently executing.

        This method provides a simple boolean check to determine if a named
        action is currently in the running state within the ActionManager.

        Args:
            action_name (str): Name of the action to check for running status.

        Returns:
            bool: True if the action is currently running, False otherwise.

        Usage:
            This method is commonly used for:
            - Preventing duplicate action submissions
            - Determining if cancellation is possible
            - Status reporting and decision making
            - Validation before submitting related actions

        Thread Safety:
            This method is thread-safe as it performs a simple dictionary
            lookup without modifying shared state.

        Note:
            This method checks the presence of the action name in the
            running_actions dictionary. It does not validate if the action
            is properly registered in the action_registry.
        """
        return action_name in self.running_actions

    def get_running_actions(self):
        """Get a list of all currently running action names.

        This method provides a snapshot of all actions that are currently
        executing within the ActionManager, returned as a list of action names.

        Returns:
            list[str]: List of action names currently in running state.
                      Empty list if no actions are running.

        Thread Safety:
            Uses thread lock to ensure consistent snapshot of running actions
            state, preventing race conditions during concurrent modifications.

        Use Cases:
            - Status monitoring and reporting
            - System health checks
            - Debugging action execution state
            - Capacity planning and load assessment
            - Integration with external monitoring systems

        Data Consistency:
            The returned list is a snapshot taken under lock protection,
            ensuring consistency at the time of the call. The actual running
            state may change immediately after the method returns.

        Note:
            This method returns a copy of the action names list, so
            modifications to the returned list do not affect the internal
            state of the ActionManager.
        """
        with self.lock:
            return list(self.running_actions.keys())

    def is_action_running(self, action_name):
        """Check if a specific action is currently running with thread safety.

        This method provides a thread-safe boolean check to determine if a
        named action is currently executing within the ActionManager.

        Args:
            action_name (str): Name of the action to check for running status.

        Returns:
            bool: True if the action is currently running, False otherwise.

        Thread Safety:
            Uses thread lock to ensure atomic access to the running_actions
            dictionary, preventing race conditions during concurrent access
            from multiple threads.

        Comparison with is_running():
            This method is identical to is_running() but explicitly uses
            thread locking for maximum safety in multi-threaded environments.
            Use this method when thread safety is critical.

        Use Cases:
            - Multi-threaded action status checking
            - Critical decision points requiring guaranteed consistency
            - Integration with thread-safe action management workflows

        Note:
            For simple single-threaded usage, is_running() provides the same
            functionality with slightly less overhead.
        """
        with self.lock:
            return action_name in self.running_actions

    def cleanup_timeouts(self):
        """Remove actions that have exceeded their execution timeout.

        This method identifies and cancels actions that have been running
        longer than their configured timeout duration. It should be called
        periodically to prevent resource leaks and ensure system responsiveness.

        Timeout Processing:
            1. Captures current timestamp for comparison
            2. Thread-safe scan of all running actions
            3. Identifies actions exceeding their timeout deadline
            4. Initiates cancellation for expired actions

        Thread Safety:
            Uses thread lock to safely iterate over running actions without
            interference from concurrent action submissions or completions.

        Cleanup Process:
            For each expired action:
            1. Adds action name to expired list under lock
            2. Releases lock before cancellation operations
            3. Calls cancel_action() which handles server communication
            4. Relies on cancel_action() for final cleanup

        Timeout Calculation:
            Each action's timeout is calculated as:
            timeout_deadline = start_time + configured_timeout_duration

        Recommended Usage:
            - Call periodically via ROS 2 timer (e.g., every 1-5 seconds)
            - Integrate with system health monitoring
            - Use during system shutdown to clean up pending actions

        Performance:
            - Minimal overhead when no timeouts are present
            - Scales linearly with number of running actions
            - Cleanup operations happen outside the lock

        Note:
            This method does not directly remove actions from running_actions;
            removal occurs through the standard cancellation callback chain.
        """
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
        """Publish action event notification to the action event topic.

        This method broadcasts action lifecycle events through ROS 2 messaging,
        enabling external systems to monitor action execution status and
        respond to action state changes.

        Args:
            action_name (str): Name of the action generating the event.
            action_result (ActionResult): Current result status of the action
                (e.g., SUBMITTED, SUCCESS, FAILED).

        Message Format:
            Published message contains human-readable status:
            "Action '{action_name}' is {action_result.value}"

        Event Types:
            - SUBMITTED: Action goal sent to server
            - SUCCESS: Action completed successfully
            - FAILED: Action failed during execution
            - CANCELED: Action was canceled before completion
            - REJECTED: Action goal was rejected by server
            - INVALID: Action could not be processed

        Publisher Configuration:
            Uses the action_event_pub publisher configured during
            ActionManager initialization. If no publisher is configured,
            the method silently returns without error.

        Use Cases:
            - System monitoring and logging
            - External decision making based on action events
            - User interface status updates
            - Integration with workflow management systems
            - Debugging and diagnostics

        Note:
            This method is typically called internally by the ActionManager
            during action lifecycle transitions, but can be used externally
            for custom event notifications.
        """
        if self.action_event_pub:
            status_msg = f'Action "{action_name}" is {action_result.value}'
            msg = String()
            msg.data = status_msg
            self.action_event_pub.publish(msg)

    def publish_running_actions_msg(self):
        """Publish current running actions status to the running actions topic.

        This method broadcasts the current state of all running actions
        through ROS 2 messaging, providing real-time visibility into active
        action execution for monitoring and decision-making systems.

        Message Content:
            The published message contains a formatted string showing all
            currently running actions with their keys, names, and running
            descriptions, generated by create_running_action_msg().

        Message Format Example:
            "b: [Reply action] Robot is currently generating a reply..."

        Publisher Configuration:
            Uses the action_running_pub publisher configured during
            ActionManager initialization. If no publisher is configured,
            the method silently returns without error.

        Update Triggers:
            This method is automatically called when:
            - Actions are accepted by action servers
            - Actions complete (success, failure, or cancellation)
            - Action states change requiring status updates

        Use Cases:
            - Real-time system status monitoring
            - User interface updates showing active operations
            - External decision systems needing current action context
            - System health and capacity monitoring
            - Integration with supervisory control systems

        Thread Safety:
            Delegates to create_running_action_msg() which handles thread-safe
            access to the running actions state.

        Note:
            This method publishes the complete current state rather than
            incremental changes, ensuring subscribers always have consistent
            and complete information.
        """
        if self.action_running_pub:
            running_actions_msg = self.create_running_action_msg()
            msg = String()
            msg.data = running_actions_msg
            self.action_running_pub.publish(msg)

    def create_running_action_msg(self) -> str:
        """Create a formatted status message for all currently running actions.

        This method generates a human-readable string representation of all
        actions currently executing, including their keys, names, and running
        descriptions for comprehensive status reporting.

        Returns:
            str: Formatted multi-line string containing running action details.
                Empty string if no actions are running.

        Message Format:
            Each running action is represented as:
            "action_key: [action_name] running_description"

        Message Format Example:
            "b: [Reply action] Robot is currently generating a reply to respond to user interaction"  # noqa: E501

        Sorting:
            Actions are sorted alphabetically by action_key to ensure
            consistent message ordering across calls.

        Thread Safety:
            Uses thread lock to ensure consistent snapshot of running actions
            and their associated metadata during message generation.

        Data Sources:
            - running_actions: Current action execution state
            - action_keys: Single character keys for each action
            - action_names: Human-readable names for each action
            - running_descriptions: Descriptive text for active execution

        Use Cases:
            - Status message publishing via publish_running_actions_msg()
            - Direct status queries for monitoring systems
            - Debugging and logging current system state
            - User interface display of active operations

        Note:
            This method provides a snapshot of the current state. The actual
            running actions may change immediately after the method returns.
        """
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
        """Generate formatted list of available actions for decision systems.

        This method creates a comprehensive human-readable string containing
        all registered actions with their current availability status (Do or
        Cancel) and appropriate descriptions for use by decision-making systems.

        Returns:
            str: Formatted multi-line string containing all valid actions.
                 Each line represents one available action option.

        Message Format:
            Each action is represented as:
            "action_key: Do|Cancel [action_name] description"

        Message Format Examples:
            When action is available for execution:
            "a: Do [Idle action] A no-operation action representing
            deliberate inaction."

            When action is currently running and can be canceled:
            "b: Cancel [Reply action] Canceling will stop current reply
            generation, freeing the robot for other actions."

        Action Status Logic:
            - Do: Action is not currently running and can be started
            - Cancel: Action is currently running and can be canceled

        Sorting:
            Actions are sorted alphabetically by action_key to ensure
            consistent ordering across calls and predictable decision input.

        Thread Safety:
            Uses thread lock to ensure consistent snapshot of action registry
            and running actions state during message generation.

        Data Sources:
            - action_registry: All registered actions (real and virtual)
            - running_actions: Currently executing actions
            - action_keys, action_names: Action metadata
            - action_descriptions, cancel_descriptions: Context-specific text

        Use Cases:
            - Input to external decision-making systems
            - Action selection interfaces for operators
            - System capability reporting
            - Integration with action decision servers

        Note:
            This method is commonly used by ActionCycleController to inform
            the action decision server about currently available options.
        """
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
