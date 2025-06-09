import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict

from rclpy.node import Node


class BaseAction(ABC):
    """Abstract base class for action plugins in ActionCycleController.

    This class provides the foundation for implementing action plugins that can
    be dynamically loaded and executed by the ActionCycleController. It handles
    automatic action server registration based on configuration and defines the
    contract that all action plugins must implement.

    The plugin architecture enables zero-code extension of the robot's action
    capabilities by simply creating new plugin classes and updating the
    configuration file.

    Attributes:
        node (Node): The ROS 2 node instance for logging and communication
        action_manager: The ActionManager instance for submitting action goals
        config (Dict[str, Any]): Configuration dictionary loaded from YAML file

    Example:
        ```python
        class MyCustomAction(BaseAction):
            def execute(self, state: str) -> None:
                # Custom action implementation
                pass

            def get_action_key(self) -> str:
                return "c"

            def get_action_name(self) -> str:
                return "Custom Action"
        ```
    """

    def __init__(self, node: Node, action_manager, config: Dict[str, Any]):
        """Initialize the action plugin with necessary dependencies and config.

        Automatically registers any action servers specified in the config,
        enabling the plugin to be self-contained and manage its own
        dependencies.

        Args:
            node (Node): ROS 2 node instance for logging and ROS communication
            action_manager: ActionManager instance for registering and
                submitting actions
            config (Dict[str, Any]): Configuration dictionary containing
                plugin-specific settings including action server specifications

        Raises:
            Exception: If action server registration fails, logged as error but
                doesn't prevent plugin initialization
        """
        self.node = node
        self.action_manager = action_manager
        self.config = config

        # Register action servers if specified in config
        self._register_action_servers()

    def _register_action_servers(self):
        """Register action servers based on configuration specifications.

        Supports two configuration patterns:
        1. Single action server: 'action_server' and 'action_type' keys
        2. Multiple action servers: 'action_servers' list with 'name' and
            'type' for each

        This method enables plugins to automatically set up their required
        action servers without manual intervention, supporting the self-managing
        plugin architecture.

        Configuration Examples:
            Single server:
            ```yaml
            action_server: "reply_server"
            action_type: "my_interfaces/action/Reply"
            ```

            Multiple servers:
            ```yaml
            action_servers:
              - name: "move_server"
                type: "geometry_msgs/action/MoveBase"
              - name: "grasp_server" 
                type: "manipulation_msgs/action/Grasp"
            ```

        Note:
            Registration failures are logged as errors but don't prevent plugin
            initialization, allowing for graceful degradation."""
        # Handle single action server
        if 'action_server' in self.config:
            server_name = self.config['action_server']
            action_type_str = self.config.get('action_type')
            if action_type_str:
                try:
                    # Resolve string to actual action type
                    action_type = self._resolve_action_type(action_type_str)

                    self.action_manager.register_action(
                        server_name, action_type)
                    self.node.get_logger().info(
                        f"Registered action server: {server_name} for {self.get_action_name()}"  # noqa: E501
                    )
                except Exception as e:
                    self.node.get_logger().error(
                        f"Failed to register action server {server_name}: {e}")

        # Handle multiple action servers
        elif 'action_servers' in self.config:
            for server_config in self.config['action_servers']:
                server_name = server_config['name']
                action_type_str = server_config['type']
                try:
                    # Resolve string to actual action type
                    action_type = self._resolve_action_type(action_type_str)

                    self.action_manager.register_action(
                        server_name, action_type)
                    self.node.get_logger().info(
                        f"Registered action server: {server_name} for {self.get_action_name()}"  # noqa: E501
                    )
                except Exception as e:
                    self.node.get_logger().error(
                        f"Failed to register action server {server_name}: {e}")

        try:
            self.action_key = self.config['action_key']
        except KeyError:
            self.node.get_logger().error(
                f"Action key not found in config for {self.get_action_name()}."
                "Please ensure 'action_key' is defined in the configuration.")
            raise KeyError("Action key not found in config")

    def _resolve_action_type(self, action_type_str: str):
        """Resolve action type string to actual class for ROS 2 action clients.

        This method performs runtime class loading to convert a string
        representation of a ROS 2 action type into the actual Python class
        object required by ActionClient constructors. This enables dynamic
        action type specification through configuration files rather than
        hardcoded imports.

        The method uses Python's importlib to dynamically import the module and
        extract the class, supporting the plugin architecture where action types
        are specified in YAML configuration files.

        Args:
            action_type_str (str): Fully qualified action type string in the
                format "package.module.ClassName" (e.g.,
                "exodapt_robot_interfaces.action.ReplyAction")

        Returns:
            type: The actual action class object that can be used to
                instantiate action clients. This is equivalent to importing the
                class directly but done at runtime.

        Raises:
            ImportError: If the specified module cannot be imported (package not
                found, not sourced, etc.)
            AttributeError: If the specified class name is not in the module
            ValueError: If the action_type_str format is invalid

        Example:
            ```python
            # Input string
            action_str = "exodapt_robot_interfaces.action.ReplyAction"

            # Method transforms this to equivalent of:
            # from exodapt_robot_interfaces.action import ReplyAction
            action_class = self._resolve_action_type(action_str)

            # Now action_class can be used with ActionClient:
            client = ActionClient(node, action_class, 'server_name')
            ```

        Note:
            This transformation is necessary because ROS 2 ActionClient
            requires the actual class object, not a string representation. The
            method enables configuration-driven action type specification while
            maintaining type safety at runtime.
        """
        try:
            # Split module and class name
            module_path, class_name = action_type_str.rsplit('.', 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            action_class = getattr(module, class_name)

            return action_class

        except Exception as e:
            self.node.get_logger().error(
                f"Failed to resolve action type '{action_type_str}': {e}")
            raise

    @abstractmethod
    def execute(self, state: str) -> None:
        """Execute the action with the given state parameter.

        This is the core method that defines what the action does when
        triggered. Implementations should handle the specific logic for their
        action type, such as submitting goals to action servers, updating robot
        state, or performing computations.

        Args:
            state (str): State parameter that may influence action execution.
                        The interpretation of this parameter is action-specific.

        Returns:
            None

        Note:
            Implementations should handle exceptions gracefully and log errors
            appropriately to maintain system stability during rapid action
            cycles.
        """
        pass

    def get_action_key(self) -> str:
        """Return the single character key that identifies this action.

        This key is used by the ActionCycleController to map decision inputs
        to specific action plugins. Each action plugin must have a unique key
        within the system.

        Returns:
            str: Single character key (e.g., 'a', 'b', 'c') that uniquely
                 identifies this action plugin

        Example:
            ```python
            def get_action_key(self) -> str:
                return "r"  # 'r' for reply action
            ```
        """
        return self.action_key

    @abstractmethod
    def get_action_name(self) -> str:
        """Return a human-readable name for this action.

        This name is used for logging, debugging, and user interfaces to
        provide clear identification of the action being executed.

        Returns:
            str: Descriptive name of the action (e.g., "Reply Action",
                 "Navigation Action", "Grasp Action")

        Example:
            ```python
            def get_action_name(self) -> str:
                return "Robot Reply Action"
            ```
        """
        pass

    @abstractmethod
    def get_action_description(self) -> str:
        """Return a detailed description of the action's purpose and behavior.

        This method provides additional context about what the action does,
        its intended use cases, and any important notes for the decision making
        agent.

        Returns:
            str: Detailed description of the action's functionality and
                expected behavior

        Example:
            ```python
            def get_action_description(self) -> str:
                return "Reply action enables the robot to respond to user interactions."  # noqa: E501
            ```
        """
        pass
