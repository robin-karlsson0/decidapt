from unittest.mock import MagicMock, patch

import pytest
import rclpy
from action_cycle_controller.action_cycle_controller import \
    ActionCycleController


@pytest.fixture
def action_cycle_controller():
    """Fixture to create an instance of mocked ActionCycleController."""
    rclpy.init()
    try:
        # Mock other objects by "with patch.object(N), patch.object(M), ..., as"
        with patch.object(ActionCycleController,
                          'create_client') as mock_create_client:
            mock_create_client.return_value = MagicMock()
            action_controller = ActionCycleController()

            return action_controller
    finally:
        rclpy.shutdown()


def test_action_cycle_controller_creation(action_cycle_controller):
    """Test the creation of the ActionCycleController class."""
    # Check publishers
    assert hasattr(action_cycle_controller, 'action_dec_pub')
    assert hasattr(action_cycle_controller, 'action_resp_pub')
