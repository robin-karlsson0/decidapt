from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from state_manager.state_manager import StateManager


class TestStateManager:

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
        self.state_manager = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.state_manager:
            self.executor.remove_node(self.state_manager)
            self.state_manager.destroy_node()

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_state_manager_is_node(self, mock_tokenizer):
        """Test that StateManager is properly initialized as a ROS2 Node."""
        # Mock the tokenizer to avoid loading actual model
        mock_tokenizer.return_value = MagicMock()

        # Initialize StateManager with default parameters
        self.state_manager = StateManager()
        self.executor.add_node(self.state_manager)

        # Test that it's a proper ROS2 Node
        assert isinstance(self.state_manager, Node)

        # Process any pending callbacks
        self.executor.spin_once(timeout_sec=0.1)


def test_dummy():
    """A dummy test to ensure pytest runs."""
    assert 1 + 1 == 2, 'This is a dummy test to ensure pytest runs correctly.'
