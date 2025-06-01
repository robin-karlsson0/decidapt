from abc import ABC, abstractmethod
from typing import Any, Dict

from rclpy.node import Node


class BaseAction(ABC):
    """Abstract base class for all action plugins"""

    def __init__(self, node: Node, action_manager, config: Dict[str, Any]):
        self.node = node
        self.action_manager = action_manager
        self.config = config

        # Register action servers if specified in config
        self._register_action_servers()

    def _register_action_servers(self):
        """Register action servers based on configuration"""
        # Handle single action server
        if 'action_server' in self.config:
            server_name = self.config['action_server']
            action_type = self.config.get('action_type')
            if action_type:
                try:
                    self.action_manager.register_action(
                        server_name, action_type)
                    self.node.get_logger().info(
                        f"Registered action server: {server_name} for {self.get_action_name()}"
                    )
                except Exception as e:
                    self.node.get_logger().error(
                        f"Failed to register action server {server_name}: {e}")

        # Handle multiple action servers
        elif 'action_servers' in self.config:
            for server_config in self.config['action_servers']:
                server_name = server_config['name']
                action_type = server_config['type']
                try:
                    self.action_manager.register_action(
                        server_name, action_type)
                    self.node.get_logger().info(
                        f"Registered action server: {server_name} for {self.get_action_name()}"
                    )
                except Exception as e:
                    self.node.get_logger().error(
                        f"Failed to register action server {server_name}: {e}")

    @abstractmethod
    def execute(self, state: str) -> None:
        """Execute the action with given state"""
        pass

    @abstractmethod
    def get_action_key(self) -> str:
        """Return the single character key for this action"""
        pass

    @abstractmethod
    def get_action_name(self) -> str:
        """Return human-readable name of the action"""
        pass
