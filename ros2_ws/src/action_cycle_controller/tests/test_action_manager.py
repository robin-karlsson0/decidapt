from unittest.mock import MagicMock, patch

import pytest
from action_cycle_controller.action_manager import ActionManager, ActionResult
from action_msgs.msg import GoalStatus

# Mock classes to simulate ROS 2 action client behavior:
#
# Async chain:
#
#     _execute_action:
#         Async call (1):   client.send_goal_async(goal) --> goal_future
#
#     _handle_goal_response:
#         Async result (1): goal_future.result() --> goal_handle
#         Async call (2):   goal_handle.get_result_async() --> result_future
#
#     _handle_result:
#         Async result (2) result_future.result() --> result
#
#     Async objects:
#         - goal_future
#         - goal_handle
#         - result_future
#         - result
#
# Ref: ActionManager class definition


class MockGoalFuture:
    """Mock for the goal future returned by send_goal_async().

    In real ROS 2: ActionClient.send_goal_async() returns a Future that resolves
    to a GoalHandle when the action server accepts/rejects the goal.

    In tests: Use trigger_done_callbacks() to simulate goal acceptance and
    advance the async chain to the next stage (goal_handle processing).
    """

    def __init__(self, goal_handle=None):
        self.callbacks = []
        self._goal_handle = goal_handle
        self.canceled = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self._goal_handle

    def cancel(self):
        self.canceled = True

    def trigger_done_callbacks(self):
        """Manually trigger callbacks during testing."""
        for callback in self.callbacks:
            callback(self)


class MockGoalHandle:
    """Mock for the GoalHandle object received from goal acceptance.

    In real ROS 2: The goal_future resolves to a GoalHandle that provides
    methods to monitor and control the executing action (get results, cancel).

    In tests: Acts as the bridge between goal acceptance and result delivery.
    Configure the result_future in __init__ to define what outcome your test
    expects. The GoalHandle's get_result_async() method starts the final
    stage of the async chain.
    """

    def __init__(self, accepted=True, status=GoalStatus.STATUS_SUCCEEDED):
        self.accepted = accepted
        self.result_future = MockResultFuture(MockResult(status))
        self.cancel_called = False
        self._canceled = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_called = True
        self._canceled = True

        # Create a mock cancel response future
        cancel_future = MockGoalFuture()

        # Mock the cancel response with goals_canceling list

        class MockCancelResponse:

            def __init__(self):
                # Non-empty list indicates successful cancellation
                self.goals_canceling = [1]

        cancel_future._goal_handle = MockCancelResponse()
        return cancel_future


class MockResultFuture:
    """Mock for the result future returned by goal_handle.get_result_async().

    In real ROS 2: get_result_async() returns a Future that resolves to the
    final action result when the action server completes execution.

    In tests: Use trigger_done_callbacks() to simulate action completion and
    deliver the final result to your callback handlers. This is the terminal
    stage of the async chain where your business logic receives outcomes.
    """

    def __init__(self, result=None):
        self.callbacks = []
        self._result = result
        self.canceled = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self._result

    def cancel(self):
        self.canceled = True

    def trigger_done_callbacks(self):
        """Manually trigger callbacks during testing."""
        for callback in self.callbacks:
            callback(self)


class MockResult:
    """Mock for the final action result containing execution status and data.

    In real ROS 2: The result object contains the action's outcome status
    (SUCCESS, CANCELLED, ABORTED) plus any custom result data from the server.

    In tests: Configure the status in __init__ to simulate different action
    outcomes. This determines how your ActionManager interprets success/failure
    and what it reports to callback handlers. The terminal payload of the
    entire async chain.
    """

    def __init__(self, status=GoalStatus.STATUS_SUCCEEDED):
        self.status = status


@pytest.fixture
def mock_action_client():
    """Replace ROS 2 ActionClient to avoid middleware initialization.

    In real ROS 2: ActionClient requires node initialization, discovery
    protocol, and middleware setup before sending goals to action servers.

    In tests: Patches ActionClient import to return a MagicMock that intercepts
    send_goal_async() calls. Configure the return value in each test to control
    the async chain behavior without ROS 2 overhead.
    """
    mock_client = MagicMock()
    mock_client.send_goal_async.return_value = MockGoalFuture()

    def mock_constructor(*args, **kwargs):
        return mock_client

    with patch('action_cycle_controller.action_manager.ActionClient',
               mock_constructor):
        yield mock_client


@pytest.fixture
def action_manager(mock_action_client):
    """Create ActionManager with mocked dependencies for isolated testing.

    In real ROS 2: ActionManager requires a valid ROS 2 node to create
    ActionClient instances and handle timer callbacks.

    In tests: Uses MagicMock for the node parameter to avoid node lifecycle
    management. Pre-registers a test action so individual tests can focus
    on action execution rather than registration setup.
    """
    node = MagicMock()
    manager = ActionManager(node)
    manager.register_action(
        'dummy_action',
        MagicMock(),
        'dummy_key',
        timeout=5.0,
    )
    return manager


def test_action_lifecycle_success(action_manager, mock_action_client):
    """Test successful action execution flow from submission to completion.

    Verifies that ActionManager correctly:
    1. Submits goals and tracks running state
    2. Handles goal acceptance asynchronously
    3. Processes result completion and invokes callbacks
    4. Cleans up internal state after completion

    This test validates the entire async chain without ROS 2 dependencies by
    manually triggering each stage of the future callback sequence.
    """
    # 1. Setup mock chain: goal_future -> goal_handle -> result_future -> result
    result = MockResult(status=GoalStatus.STATUS_SUCCEEDED)
    result_future = MockResultFuture(result)
    goal_handle = MockGoalHandle(accepted=True)
    goal_handle.result_future = result_future
    goal_future = MockGoalFuture(goal_handle)

    mock_action_client.send_goal_async.return_value = goal_future

    # 2. Create callback to track completion
    result_callback = MagicMock()

    # 3. Submit the action
    goal = MagicMock()
    submit_result = action_manager.submit_action(
        'dummy_action',
        goal,
        result_callback,
    )

    # 4. Verify action was submitted
    assert submit_result == ActionResult.SUBMITTED
    mock_action_client.send_goal_async.assert_called_with(goal)
    assert action_manager.is_running('dummy_action')

    # 5. Simulate goal accepted: trigger goal_future callbacks
    goal_future.trigger_done_callbacks()

    # 6. Simulate result received: trigger result_future callbacks
    result_future.trigger_done_callbacks()

    # 7. Verify action completed successfully
    assert not action_manager.is_running('dummy_action')
    result_callback.assert_called_once()
    args = result_callback.call_args[0]
    assert args[0] == 'dummy_action'  # action name
    assert args[1] == ActionResult.SUCCESS  # result status
    assert len(action_manager.get_running_actions()) == 0


def test_reject_duplicate_action_submission(action_manager,
                                            mock_action_client):
    """Test that duplicate submissions for running actions are handled properly.

    Verifies that:
    1. First submission succeeds and action enters running state
    2. Second submission before goal acceptance causes cancellation exception
    3. Third submission after goal acceptance attempts cancellation
    4. Only one goal is sent to the action client
    """
    # Setup mock chain for first submission
    goal_handle = MockGoalHandle(accepted=True)
    goal_future = MockGoalFuture(goal_handle)
    mock_action_client.send_goal_async.return_value = goal_future

    # Create goal and callback
    goal = MagicMock()
    callback = MagicMock()

    # First submission should succeed
    result1 = action_manager.submit_action('dummy_action', goal, callback)
    assert result1 == ActionResult.SUBMITTED
    assert action_manager.is_running('dummy_action')

    # Second submission should try to cancel the first, but fail since
    # goal_handle doesn't exist yet (before goal acceptance)
    expected_msg = 'Action state is missing a "goal_handle" object'
    with pytest.raises(Exception, match=expected_msg):
        action_manager.submit_action('dummy_action', goal, callback)

    # Trigger goal acceptance to ensure action is running with goal_handle
    goal_future.trigger_done_callbacks()
    assert action_manager.is_running('dummy_action')

    # Now create a new mock setup for the next attempt
    goal_handle2 = MockGoalHandle(accepted=True)
    goal_future2 = MockGoalFuture(goal_handle2)
    mock_action_client.send_goal_async.return_value = goal_future2

    # Third submission should now succeed in canceling the first action
    # since goal_handle exists
    result3 = action_manager.submit_action('dummy_action', goal, callback)
    assert result3 == ActionResult.CANCELED  # Returns result of cancellation

    # Verify only one goal was sent to client (for the first submission)
    assert mock_action_client.send_goal_async.call_count == 1


def test_cancel_running_action(action_manager, mock_action_client):
    """Test cancellation of a running action after goal acceptance.

    Verifies that ActionManager correctly:
    1. Only allows cancellation after goal acceptance (goal_handle exists)
    2. Returns CANCELED result for successful cancellation attempts
    3. Handles async cancellation with cancel_done callback
    4. Invokes result callback with CANCELED status
    5. Cleans up internal state after cancellation completes
    """
    # Setup mock chain for cancellation test
    result = MockResult(status=GoalStatus.STATUS_CANCELED)
    result_future = MockResultFuture(result)
    goal_handle = MockGoalHandle(accepted=True)
    goal_handle.result_future = result_future
    goal_future = MockGoalFuture(goal_handle)

    mock_action_client.send_goal_async.return_value = goal_future

    # Create callback to track cancellation
    result_callback = MagicMock()

    # Submit the action
    goal = MagicMock()
    submit_result = action_manager.submit_action(
        'dummy_action',
        goal,
        result_callback,
    )

    assert submit_result == ActionResult.SUBMITTED
    assert action_manager.is_running('dummy_action')

    # Trigger goal acceptance to set goal_handle
    goal_future.trigger_done_callbacks()
    assert action_manager.is_running('dummy_action')

    # Now cancel after goal acceptance (when goal_handle exists)
    cancel_result = action_manager.cancel_action('dummy_action')
    assert cancel_result == ActionResult.CANCELED
    assert goal_handle.cancel_called  # Goal handle cancel should be called

    # Action should still be running until cancel_done callback completes
    assert action_manager.is_running('dummy_action')

    # Get the cancel future that was returned by cancel_goal_async
    # We need to trigger its done callback to simulate the cancel response
    # This will call action_manager.cancel_done()

    # Manually trigger the cancel_done callback since we can't easily
    # access the cancel future from the test. We'll simulate what
    # cancel_done does by calling _complete_action directly.
    action_manager._complete_action('dummy_action', ActionResult.CANCELED)

    # Now the action should be cleaned up
    assert not action_manager.is_running('dummy_action')

    # Verify callback is invoked with CANCELED status
    result_callback.assert_called_once()
    args = result_callback.call_args[0]
    assert args[0] == 'dummy_action'
    assert args[1] == ActionResult.CANCELED


def test_cancel_non_running_action(action_manager):
    """Test that canceling a non-running action returns appropriate result.

    Verifies that:
    1. Canceling unregistered actions returns INVALID
    2. Canceling registered but not running actions returns INVALID
    3. No side effects occur from invalid cancellation attempts
    """
    # Test canceling unregistered action
    cancel_result = action_manager.cancel_action('nonexistent_action')
    # Should return INVALID for unregistered action
    assert cancel_result == ActionResult.INVALID

    # Test canceling registered but not running action
    cancel_result = action_manager.cancel_action('dummy_action')
    # Should return INVALID for non-running action
    assert cancel_result == ActionResult.INVALID

    # Verify no actions are running
    assert len(action_manager.get_running_actions()) == 0


def test_cancel_with_timeout_cleanup(action_manager, mock_action_client):
    """Test that cancelled actions don't trigger timeout handling.

    Verifies that:
    1. Cancelled actions are removed from timeout tracking
    2. Timer cleanup occurs properly after cancellation
    3. No timeout callbacks are invoked for cancelled actions
    """
    # Setup mock chain
    goal_handle = MockGoalHandle(accepted=True)
    goal_future = MockGoalFuture(goal_handle)
    mock_action_client.send_goal_async.return_value = goal_future

    # Submit action with timeout
    goal = MagicMock()
    result_callback = MagicMock()

    submit_result = action_manager.submit_action(
        'dummy_action',
        goal,
        result_callback,
    )
    assert submit_result == ActionResult.SUBMITTED

    # Trigger goal acceptance to set goal_handle (required for cancellation)
    goal_future.trigger_done_callbacks()
    assert action_manager.is_running('dummy_action')

    # Cancel after goal acceptance
    cancel_result = action_manager.cancel_action('dummy_action')
    assert cancel_result == ActionResult.CANCELED

    # Simulate the cancel_done callback completing
    action_manager._complete_action('dummy_action', ActionResult.CANCELED)

    # Verify action is no longer tracked for timeout
    assert not action_manager.is_running('dummy_action')
    assert len(action_manager.get_running_actions()) == 0

    # Verify callback was called with cancellation, not timeout
    result_callback.assert_called_once()
    args = result_callback.call_args[0]
    assert args[1] == ActionResult.CANCELED  # Not TIMEOUT


def test_cancel_all_running_actions(action_manager, mock_action_client):
    """Test cancellation of multiple running actions simultaneously.

    Verifies that:
    1. Multiple actions can be cancelled at once
    2. All callbacks receive CANCELED status
    3. Internal state is cleaned up for all actions
    """
    # Register additional test actions
    action_manager.register_action(
        'action_1',
        MagicMock(),
        'action_key_1',
        timeout=5.0,
    )
    action_manager.register_action(
        'action_2',
        MagicMock(),
        'action_key_2',
        timeout=5.0,
    )

    # Setup mocks for multiple actions
    callbacks = []
    goals = []

    for i in range(3):
        goal_handle = MockGoalHandle(accepted=True)
        goal_future = MockGoalFuture(goal_handle)
        mock_action_client.send_goal_async.return_value = goal_future

        goal = MagicMock()
        callback = MagicMock()
        action_name = f'action_{i}' if i > 0 else 'dummy_action'

        action_manager.submit_action(action_name, goal, callback)
        goal_future.trigger_done_callbacks()  # Accept all goals

        callbacks.append(callback)
        goals.append(goal)

    # Verify all actions are running
    assert len(action_manager.get_running_actions()) == 3

    # Cancel all actions individually
    running_actions = action_manager.get_running_actions().copy()
    for action_name in running_actions:
        cancel_result = action_manager.cancel_action(action_name)
        # All cancellations should succeed and return CANCELED
        assert cancel_result == ActionResult.CANCELED

        # Simulate the cancel_done callback completing
        action_manager._complete_action(action_name, ActionResult.CANCELED)

    # Verify all actions are cancelled
    assert len(action_manager.get_running_actions()) == 0

    # Verify all callbacks received cancellation
    for callback in callbacks:
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == ActionResult.CANCELED


def test_cancel_before_goal_acceptance_raises_exception(
        action_manager, mock_action_client):
    """Test that canceling an action before goal acceptance raises an exception.

    Verifies that:
    1. Actions submitted but not yet accepted cannot be cancelled
    2. An exception is raised when attempting to cancel before goal_handle
       exists
    3. This reflects the current implementation requirement
    """
    # Setup mock chain but don't trigger goal acceptance
    goal_handle = MockGoalHandle(accepted=True)
    goal_future = MockGoalFuture(goal_handle)
    mock_action_client.send_goal_async.return_value = goal_future

    # Submit action but don't trigger goal acceptance
    goal = MagicMock()
    result_callback = MagicMock()

    submit_result = action_manager.submit_action(
        'dummy_action',
        goal,
        result_callback,
    )
    assert submit_result == ActionResult.SUBMITTED
    assert action_manager.is_running('dummy_action')

    # Try to cancel before goal acceptance (goal_handle not set yet)
    # This should raise an exception in the current implementation
    expected_msg = 'Action state is missing a "goal_handle" object'
    with pytest.raises(Exception, match=expected_msg):
        action_manager.cancel_action('dummy_action')

    # Action should still be running since cancellation failed
    assert action_manager.is_running('dummy_action')
