import os
import tempfile
from typing import Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest
import rclpy
import yaml
from action_cycle_controller.action_registry import ActionRegistry
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

PKG_NAME = 'action_cycle_controller'


class TestActionRegistry:
    """Test suite for ActionRegistry class."""

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
        self.node = rclpy.create_node('test_action_registry_node')
        self.action_manager = MagicMock()
        self.action_registry = ActionRegistry(self.node, self.action_manager)
        self.executor.add_node(self.node)

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.node:
            self.executor.remove_node(self.node)
            self.node.destroy_node()

    def get_actions_config_path(self) -> str:
        """Get the path to the actions.yaml config file."""
        # For development with --symlink-install, read directly from source
        test_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(test_dir)
        config_path = os.path.join(package_root, PKG_NAME, 'config',
                                   'actions.yaml')

        if os.path.exists(config_path):
            return config_path

        # Fallback to installed package location (for production/CI)
        try:
            package_share_dir = get_package_share_directory(
                'action_cycle_controller')
            installed_config_path = os.path.join(package_share_dir, 'config',
                                                 'actions.yaml')
            if os.path.exists(installed_config_path):
                return installed_config_path
        except Exception:
            pass

        raise FileNotFoundError(
            f"actions.yaml not found in source ({config_path}) or installed package"
        )

    def test_init(self):
        """Test ActionRegistry initialization."""

        action_registry = self.action_registry

        assert action_registry.node == self.node
        assert action_registry.action_manager == self.action_manager
        assert isinstance(action_registry.actions, dict)
        assert isinstance(action_registry.action_mapping, dict)
        assert len(action_registry.actions) == 0
        assert len(action_registry.action_mapping) == 0

    def test_load_real_config(self):
        """Test loading the actual actions.yaml config file."""
        config_path = self.get_actions_config_path()

        # Verify the file exists
        assert os.path.exists(
            config_path), f"Config file not found at {config_path}"

        # Load and test with real config
        self.action_registry.load_from_config(config_path)

        # Add assertions based on your actual config content
        assert len(self.action_registry.actions) > 0
        assert len(self.action_registry.action_mapping) > 0
