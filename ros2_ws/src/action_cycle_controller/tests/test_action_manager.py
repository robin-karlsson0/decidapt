from unittest.mock import MagicMock, patch

import pytest
from action_cycle_controller.action_manager import ActionManager, ActionResult
from action_msgs.msg import GoalStatus


class MockFuture:
    """Mock for the Future returned by send_goal_async."""

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

    # Method to manually trigger callbacks during testing
    def trigger_done_callbacks(self):
        for callback in self.callbacks:
            callback(self)


class MockGoalHandle:
    """Mock for the GoalHandle object."""

    def __init__(self, accepted=True, status=GoalStatus.STATUS_SUCCEEDED):
        self.accepted = accepted
        self._result_future = MockFuture(MockResult(status))
        self.cancel_called = False
        self._cancelled = False

    def get_result_async(self):
        return self._result_future

    def cancel_goal_async(self):
        self.cancel_called = True
        self._cancelled = True
        # When cancelling, the result future should also be cancelled
        self._result_future.cancel()
        return MockFuture()


class MockResult:
    """Mock for the action result."""

    def __init__(self, status=GoalStatus.STATUS_SUCCEEDED):
        self.status = status


# Mock the entire ActionClient class to avoid ROS 2 initialization
@pytest.fixture
def mock_action_client():
    """Create a MockActionClient class to replace rclpy.action.ActionClient."""
    mock_client = MagicMock()

    # Configure the mock methods we need
    mock_client.send_goal_async.return_value = MockFuture()

    # Create a constructor that returns our mock
    def mock_constructor(*args, **kwargs):
        return mock_client

    with patch('action_cycle_controller.action_manager.ActionClient',
               mock_constructor):
        yield mock_client


@pytest.fixture
def action_manager(mock_action_client):
    """Create an ActionManager with mocked ActionClient."""
    node = MagicMock()
    manager = ActionManager(node)

    # Register a test action
    manager.register_action("test_action", MagicMock(), timeout=5.0)

    # Store the mock client for easy access in tests
    return manager


def test_action_lifecycle_success(action_manager, mock_action_client):
    """Test a complete successful action lifecycle."""

    # 1. Configure the mock client for this specific test
    goal_handle = MockGoalHandle(accepted=True,
                                 status=GoalStatus.STATUS_SUCCEEDED)
    goal_response_future = MockFuture(goal_handle)
    mock_action_client.send_goal_async.return_value = goal_response_future

    # 2. Create callback to track completion
    result_callback = MagicMock()

    # 3. Submit the action
    goal = MagicMock()
    result = action_manager.submit_action("test_action", goal, result_callback)

    # 4. Verify action was submitted
    assert result == ActionResult.SUBMITTED
    mock_action_client.send_goal_async.assert_called_with(goal)
    assert action_manager.is_running("test_action")

    # 5. Simulate goal accepted response
    goal_response_future.trigger_done_callbacks()

    # 6. Simulate result received
    goal_handle._result_future.trigger_done_callbacks()

    # 7. Verify action completed successfully
    assert not action_manager.is_running("test_action")
    result_callback.assert_called_once()
    args = result_callback.call_args[0]
    assert args[0] == "test_action"  # action name
    assert args[1] == ActionResult.SUCCESS  # result status

    # 8. Verify empty running actions
    assert len(action_manager.get_running_actions()) == 0


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
