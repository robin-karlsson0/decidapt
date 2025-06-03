import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import rclpy
from action_cycle_controller.action_cycle_controller import \
    ActionCycleController
from exodapt_robot_interfaces.action import ActionDecision
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from std_msgs.msg import String


class TestActionCycleController:
    """Unit tests for ActionCycleController ROS 2 node."""

    @classmethod
    def setup_class(cls):
        """Initialize ROS2 once for all tests."""
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS2 after all tests."""
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        """Setup before each test method."""
        self.action_cycle_controller = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.action_cycle_controller:
            self.executor.remove_node(self.action_cycle_controller)
            self.action_cycle_controller.destroy_node()

    def create_test_parameters(self):
        """Create Parameter objects for testing."""
        return [
            Parameter('ac_freq', Parameter.Type.DOUBLE, 1.0),
            Parameter('actions_config', Parameter.Type.STRING,
                      'test_config/actions.yaml')
        ]

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_initialization_with_default_parameters(self, mock_registry,
                                                    mock_manager):
        """Test ActionCycleController initialization with default parameters."""
        # Mock ActionRegistry methods
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        # Mock ActionManager
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Verify default parameters
        assert self.action_cycle_controller.ac_loop_freq == 1.0
        assert self.action_cycle_controller.get_parameter(
            'actions_config').value == 'config/actions.yaml'

        # Verify ActionRegistry initialization
        mock_registry.assert_called_once()
        mock_registry_instance.load_from_config.assert_called_once_with(
            'config/actions.yaml')

        # Verify ActionManager initialization
        mock_manager.assert_called_once()

        # Verify valid actions set
        assert self.action_cycle_controller.valid_actions_set == {'a', 'b'}

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_execute_action_valid_action(self, mock_registry, mock_manager):
        """Test execute_action with valid action decision."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry_instance.execute_action.return_value = True
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Set initial state
        self.action_cycle_controller.state = "test_state"

        # Test valid action execution
        self.action_cycle_controller.execute_action("b")

        # Verify action was executed through registry
        mock_registry_instance.execute_action.assert_called_once_with(
            "b", "test_state")
        assert self.action_cycle_controller.prev_decision == "b"

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_execute_action_invalid_action_fallback(self, mock_registry,
                                                    mock_manager):
        """Test execute_action with invalid action decision falls back to 'a'.
        """
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry_instance.execute_action.return_value = True
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Set initial state
        self.action_cycle_controller.state = "test_state"

        # Test invalid action execution
        self.action_cycle_controller.execute_action("c")  # Invalid action

        # Verify fallback to 'a' action
        mock_registry_instance.execute_action.assert_called_once_with(
            "a", "test_state")
        assert self.action_cycle_controller.prev_decision == "a"

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_execute_action_execution_failure(self, mock_registry,
                                              mock_manager):
        """Test execute_action handles execution failure gracefully."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry_instance.execute_action.return_value = False  # Simulate failure
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Set initial state
        self.action_cycle_controller.state = "test_state"

        # Test action execution failure
        with patch.object(self.action_cycle_controller.get_logger(),
                          'error') as mock_logger:
            self.action_cycle_controller.execute_action("b")

            # Verify error was logged
            mock_logger.assert_called_once_with('Failed to execute action: b')

        # Verify action was attempted
        mock_registry_instance.execute_action.assert_called_once_with(
            "b", "test_state")
        assert self.action_cycle_controller.prev_decision == "b"

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_run_action_cycle_ad_request(self, mock_registry, mock_manager):
        """Test run_action_cycle initiates action decision request."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Mock action client
        mock_ad_action_client = MagicMock()
        mock_ad_future = MagicMock()
        mock_ad_action_client.send_goal_async.return_value = mock_ad_future
        self.action_cycle_controller._ad_action_client = mock_ad_action_client

        # Set test state
        self.action_cycle_controller.state = "test_robot_state"

        # Test action cycle execution
        self.action_cycle_controller.run_action_cycle()

        # Verify action client interactions
        mock_ad_action_client.wait_for_server.assert_called_once()
        mock_ad_action_client.send_goal_async.assert_called_once()

        # Verify goal content
        call_args = mock_ad_action_client.send_goal_async.call_args[0][0]
        assert call_args.state == "test_robot_state"

        # Verify callback setup
        mock_ad_future.add_done_callback.assert_called_once_with(
            self.action_cycle_controller.ad_response_callback)

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_ad_response_callback_goal_accepted(self, mock_registry,
                                                mock_manager):
        """Test ad_response_callback when goal is accepted."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Mock goal future and handle
        mock_goal_future = MagicMock()
        mock_goal_handle = MagicMock()
        mock_goal_handle.accepted = True
        mock_result_future = MagicMock()
        mock_goal_handle.get_result_async.return_value = mock_result_future
        mock_goal_future.result.return_value = mock_goal_handle

        # Test callback
        self.action_cycle_controller.ad_response_callback(mock_goal_future)

        # Verify result future setup
        mock_goal_handle.get_result_async.assert_called_once()
        mock_result_future.add_done_callback.assert_called_once_with(
            self.action_cycle_controller.ad_result_callback)

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_ad_response_callback_goal_rejected(self, mock_registry,
                                                mock_manager):
        """Test ad_response_callback when goal is rejected."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Mock goal future and handle
        mock_goal_future = MagicMock()
        mock_goal_handle = MagicMock()
        mock_goal_handle.accepted = False
        mock_goal_future.result.return_value = mock_goal_handle

        # Test callback with logging
        with patch.object(self.action_cycle_controller.get_logger(),
                          'info') as mock_logger:
            self.action_cycle_controller.ad_response_callback(mock_goal_future)

            # Verify rejection was logged
            mock_logger.assert_called_once_with('Goal rejected')

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_ad_result_callback(self, mock_registry, mock_manager):
        """Test ad_result_callback processes action decision result."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b', 'c']
        mock_registry_instance.execute_action.return_value = True
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Mock result future
        mock_result_future = MagicMock()
        mock_result = MagicMock()
        mock_result.result.pred_action = "c"
        mock_result_future.result.return_value = mock_result

        # Set test state
        self.action_cycle_controller.state = "test_state"

        # Test callback with logging
        with patch.object(self.action_cycle_controller.get_logger(),
                          'info') as mock_logger:
            self.action_cycle_controller.ad_result_callback(mock_result_future)

            # Verify action was logged
            mock_logger.assert_called_once_with('Received action: c')

        # Verify action execution
        mock_registry_instance.execute_action.assert_called_once_with(
            "c", "test_state")
        assert self.action_cycle_controller.prev_decision == "c"

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_state_subscription_integration(self, mock_registry, mock_manager):
        """Test state subscription integration with published messages."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Create state publisher
        state_publisher_node = rclpy.create_node('state_publisher')
        state_publisher = state_publisher_node.create_publisher(
            String, 'state', 10)
        self.executor.add_node(state_publisher_node)

        # Publish state message
        state_msg = String()
        state_msg.data = "integration_test_state"
        state_publisher.publish(state_msg)

        # Process message
        self.executor.spin_once(timeout_sec=0.1)

        # Verify state was updated
        assert self.action_cycle_controller.state == "integration_test_state"

        # Cleanup
        self.executor.remove_node(state_publisher_node)
        state_publisher_node.destroy_node()

    @patch('action_cycle_controller.action_cycle_controller.ActionManager')
    @patch('action_cycle_controller.action_cycle_controller.ActionRegistry')
    def test_publishers_creation(self, mock_registry, mock_manager):
        """Test that required publishers are created."""
        # Setup mocks
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_valid_actions.return_value = ['a', 'b']
        mock_registry.return_value = mock_registry_instance

        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance

        self.action_cycle_controller = ActionCycleController()
        self.executor.add_node(self.action_cycle_controller)

        # Verify publishers exist
        assert hasattr(self.action_cycle_controller, 'action_event_pub')
        assert hasattr(self.action_cycle_controller, 'action_running_pub')

        # Verify ActionManager was initialized with publishers
        mock_manager.assert_called_once_with(
            self.action_cycle_controller,
            self.action_cycle_controller.action_event_pub,
            self.action_cycle_controller.action_running_pub)
