import importlib
from typing import Dict

import yaml

from .base_action import BaseAction


class ActionRegistry:
    """Registry for dynamically loading and managing action plugins"""

    def __init__(self, node, action_manager):
        self.node = node
        self.action_manager = action_manager
        self.plugins: Dict[str, BaseAction] = {}
        self.action_mapping: Dict[str, str] = {}

    def load_from_config(self, config_path: str):
        """Load action plugins from YAML configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            for action_config in config.get('actions', []):
                self._load_plugin(action_config)

        except Exception as e:
            self.node.get_logger().error(f"Failed to load config: {e}")

    def _load_plugin(self, action_config: dict):
        """Load a single action plugin"""
        try:
            module_name = action_config['module']
            class_name = action_config['class']

            # Dynamic import
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)

            # Instantiate plugin with config
            plugin = plugin_class(self.node, self.action_manager,
                                  action_config)

            # Register plugin
            action_key = plugin.get_action_key()
            self.plugins[action_key] = plugin
            self.action_mapping[action_key] = plugin.get_action_name()

            self.node.get_logger().info(
                f"Loaded action plugin: {action_key} -> {plugin.get_action_name()}"
            )

        except Exception as e:
            self.node.get_logger().error(
                f"Failed to load plugin {action_config}: {e}")

    def execute_action(self, action_key: str, state: str) -> bool:
        """Execute action by key, return True if successful"""
        if action_key in self.plugins:
            self.plugins[action_key].execute(state)
            return True
        return False

    def get_valid_actions(self) -> set:
        """Return set of valid action keys"""
        return set(self.plugins.keys())
