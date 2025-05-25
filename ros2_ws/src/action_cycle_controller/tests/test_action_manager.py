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
        self.cancelled = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self._goal_handle

    def cancel(self):
        self.cancelled = True

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
        self._cancelled = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_called = True
        self._cancelled = True
        self.result_future.cancel()
        return MockGoalFuture()


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
        self.cancelled = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True

    def trigger_done_callbacks(self):
        """Manually trigger callbacks during testing."""
        for callback in self.callbacks:
            callback(self)


class MockResult:
    """Mock for the final action result containing execution status and data.

    In real ROS 2: The result object contains the action's outcome status
    (SUCCESS, CANCELED, ABORTED) plus any custom result data from the server.

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
    manager.register_action('dummy_action', MagicMock(), timeout=5.0)
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
    """Test that duplicate submissions for running actions are rejected.

    Verifies that:
    1. First submission succeeds and action enters running state
    2. Second submission is rejected while action is still running
    3. Only one goal is sent to the action client
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

    # Second submission should be rejected before first callback completes
    result2 = action_manager.submit_action('dummy_action', goal, callback)
    assert result2 == ActionResult.INVALID

    # Trigger goal acceptance to ensure action is running
    goal_future.trigger_done_callbacks()

    # Third submission should be rejected after goal acceptance
    result3 = action_manager.submit_action('dummy_action', goal, callback)
    assert result3 == ActionResult.INVALID

    # Verify only one goal was sent to client
    assert mock_action_client.send_goal_async.call_count == 1


#def test_action_cancellation(action_manager, mock_action_client):
#    """Test action cancellation."""
#    # 1. Configure the mock client for this specific test
#    goal_handle = MockGoalHandle(accepted=True)
#    goal_response_future = MockFuture(goal_handle)
#    mock_action_client.send_goal_async.return_value = goal_response_future
#
#    # 2. Create callback to track completion
#    result_callback = MagicMock()
#
#    # 3. Submit the action
#    goal = MagicMock()
#    action_manager.submit_action("test_action", goal, result_callback)
#
#    # 4. Simulate goal accepted response
#    goal_response_future.trigger_done_callbacks()
#
#    # 5. Cancel the action BEFORE result comes back
#    success = action_manager.cancel_action("test_action")
#
#    # 6. Verify cancellation
#    assert success
#    assert goal_handle.cancel_called
#    assert goal_handle._result_future.cancelled
#
#    # 7. Verify callback was called with cancelled status
#    result_callback.assert_called_once()
#    args = result_callback.call_args[0]
#    assert args[0] == "test_action"
#    assert args[1] == ActionResult.CANCELLED

# def test_action_rejection(action_manager, mock_action_client):
#     """Test action rejection handling."""
#     # 1. Configure the mock client for this specific test
#     goal_handle = MockGoalHandle(accepted=False)  # Rejected goal
#     goal_response_future = MockFuture(goal_handle)
#     mock_action_client.send_goal_async.return_value = goal_response_future
#     # 2. Create callback to track completion
#     result_callback = MagicMock()
#     # 3. Submit the action
#     goal = MagicMock()
#     action_manager.submit_action("test_action", goal, result_callback)
#     # 4. Simulate goal rejected response
#     goal_response_future.trigger_done_callbacks()
#     # 5. Verify callback was called with rejected status
#     result_callback.assert_called_once()
#     args = result_callback.call_args[0]
#     assert args[0] == "test_action"
#     assert args[1] == ActionResult.REJECTED
