import importlib
from typing import Dict

import yaml

from .base_action import BaseAction


class ActionRegistry:
    """Dynamic plugin registry for loading and managing robot action plugins.

    The ActionRegistry serves as the central hub for the modular action system,
    providing runtime loading of action plugins from YAML configuration files.
    It handles the complete lifecycle of action plugins including discovery,
    instantiation, registration, and execution routing.

    This registry enables a plugin-based architecture where new robot actions
    can be added by simply creating a plugin class and updating the
    configuration file, without requiring any modifications to the core
    controller code.

    Attributes:
        node: ROS 2 node instance for logging and communication
        action_manager: ActionManager instance for handling ROS 2 action
            operations
        actions: Dictionary mapping action keys to instantiated action plugin
            objects
        action_mapping: Dictionary mapping action keys to human-readable action
            names

    Example:
        >>> registry = ActionRegistry(node, action_manager)
        >>> registry.load_from_config('config/actions.yaml')
        >>> success = registry.execute_action('a', 'current_state')
    """

    def __init__(self, node, action_manager):
        """Initialize the ActionRegistry with required dependencies.
        Args:
            node: ROS 2 node instance used for logging and accessing node
                functionality
            action_manager: ActionManager instance for handling ROS 2 action
                client operations and goal submissions
        """
        self.node = node
        self.action_manager = action_manager
        self.actions: Dict[str, BaseAction] = {}
        self.action_mapping: Dict[str, str] = {}

    def load_from_config(self, config_path: str):
        """Load and instantiate action plugins from a YAML configuration file.

        Parses the specified YAML configuration file and dynamically loads each
        action plugin defined in the 'actions' section. Each action is
        instantiated with its configuration parameters and registered in the
        internal registry.

        The YAML configuration should follow this structure:
        ```yaml
        actions:
          - module: 'package.module'
            class: 'ActionClassName'
            key: 'a'
            name: 'descriptive_name'
            # additional plugin-specific configuration...
        ```

        Args:
            config_path: Path to the YAML configuration file containing action
                definitions

        Raises:
            Logs errors if configuration file cannot be read or parsed, but
                does not raise exceptions to prevent system failure
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            for action_config in config.get('actions', []):
                self._load_action(action_config)

        except Exception as e:
            self.node.get_logger().error(f"Failed to load config: {e}")

    def _load_action(self, action_config: dict):
        """Dynamically load and register a single action plugin from config.

        Performs runtime import of the specified module and class, instantiates
        the action plugin with the provided configuration, and registers it in
        the internal action registry using the plugin's action key.

        This method handles the complete plugin loading pipeline:
        1. Dynamic module import using importlib
        2. Class instantiation with dependency injection
        3. Action key extraction and registration
        4. Logging of successful loading

        Args:
            action_config: Dictionary containing plugin configuration with
                required keys 'module' and 'class', plus optional
                plugin-specific configuration parameters

        Raises:
            Logs errors for any failures in module import, class instantiation,
            or registration, but does not propagate exceptions to maintain
            system stability
        """
        try:
            module_name = action_config['module']
            class_name = action_config['class']

            # Dynamic import
            module = importlib.import_module(module_name)
            action_class = getattr(module, class_name)

            # Instantiate action with config
            action = action_class(self.node, self.action_manager,
                                  action_config)

            # Register action
            action_key = action.get_action_key()
            self.actions[action_key] = action
            self.action_mapping[action_key] = action.get_action_name()

            self.node.get_logger().info(
                f"Loaded action: {action_key} -> {action.get_action_name()}")

        except Exception as e:
            self.node.get_logger().error(
                f"Failed to load action {action_config}: {e}")

    def execute_action(self, action_key: str, state: str) -> bool:
        """Execute a registered action plugin by its unique key identifier.

        Routes the action execution request to the appropriate plugin based on
        the provided action key. The action is executed with the current system
        state, allowing plugins to make context-aware decisions.

        This method serves as the primary interface between the controller and
        the action plugins, providing a unified execution interface regardless
        of the specific action implementation.

        Args:
            action_key: Single character string identifier for the action to
                execute (e.g., 'a', 'b', 'c')
            state: Current system state string passed to the action plugin for
                context-aware execution

        Returns:
            bool: True if the action was found and executed successfully, False
                if the action key is not registered in the system

        Example:
            >>> success = registry.execute_action('a', 'robot_idle')
            >>> if not success:
            ...     print("Unknown action key")"""
        if action_key in self.actions:
            self.actions[action_key].execute(state)
            return True
        return False

    def get_valid_actions(self) -> set:
        """Retrieve the set of all currently registered action keys.

        Returns a set containing all valid action key identifiers that can be
        used with execute_action(). This is useful for validation, user
        interface generation, and debugging purposes.

        Returns:
            set: Set of string action keys that are currently loaded and
                available for execution (e.g., {'a', 'b', 'c'})

        Example:
            >>> valid_keys = registry.get_valid_actions()
            >>> if 'x' not in valid_keys:
            ...     print("Action 'x' is not available")
        """
        return set(self.actions.keys())
