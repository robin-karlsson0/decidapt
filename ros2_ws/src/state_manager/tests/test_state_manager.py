from unittest.mock import MagicMock, patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from state_manager.state_manager import StateManager
from std_msgs.msg import String


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

    def create_test_parameters(self):
        """Create Parameter objects for testing."""
        return [
            Parameter('event_topics', Parameter.Type.STRING_ARRAY,
                      ['/asr', '/thought', '/reply_action']),
            Parameter('continuous_topics', Parameter.Type.STRING_ARRAY,
                      ['/mllm', '/face_recognition']),
            Parameter('event_queue_max_tokens', Parameter.Type.INTEGER, 400),
            Parameter('continuous_queue_max_tokens', Parameter.Type.INTEGER,
                      200),
            Parameter('llm_model_name', Parameter.Type.STRING,
                      'Qwen/Qwen3-32B'),
            Parameter('long_term_memory_file_pth', Parameter.Type.STRING,
                      '/tmp/test_ltm.txt')
        ]

    def test_state_manager_event_subscriptions(self):
        """Test that event topics are properly subscribed to."""
        # Initialize StateManager with custom parameters
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers
        asr_node = rclpy.create_node('asr_publisher')
        thought_node = rclpy.create_node('thought_publisher')
        reply_action_node = rclpy.create_node('reply_action_publisher')

        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        thought_publisher = thought_node.create_publisher(
            String, '/thought', 10)
        reply_action_publisher = reply_action_node.create_publisher(
            String, '/reply_action', 10)

        self.executor.add_node(asr_node)
        self.executor.add_node(thought_node)
        self.executor.add_node(reply_action_node)

        asr_publisher.publish(String(data='Test ASR message'))
        thought_publisher.publish(String(data='Test Thought message'))
        reply_action_publisher.publish(
            String(data='Test Reply Action message'))

        self.executor.spin_once(timeout_sec=0.1)
        self.executor.spin_once(timeout_sec=0.1)
        self.executor.spin_once(timeout_sec=0.1)

        assert len(self.state_manager.event_queue) == 3
        assert 'Test ASR message' in self.state_manager.event_queue[0][0]
        assert 'Test Thought message' in self.state_manager.event_queue[1][0]
        assert 'Test Reply Action message' in self.state_manager.event_queue[
            2][0]

        # Create unrelated topic to ensure it does not affect event queue
        unrelated_node = rclpy.create_node('unrelated_publisher')
        unrelated_publisher = unrelated_node.create_publisher(
            String, '/unrelated_topic', 10)
        self.executor.add_node(unrelated_node)
        unrelated_publisher.publish(String(data='Unrelated message'))
        self.executor.spin_once(timeout_sec=0.1)
        # Ensure unrelated topic does not affect event queue
        assert len(self.state_manager.event_queue) == 3

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(thought_node)
        self.executor.remove_node(reply_action_node)
        self.executor.remove_node(unrelated_node)
        asr_node.destroy_node()
        thought_node.destroy_node()
        reply_action_node.destroy_node()
        unrelated_node.destroy_node()

    def test_state_manager_continuous_subscriptions(self):
        """Test that continuous topics are properly subscribed to."""
        # Initialize StateManager with custom parameters
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers
        mllm_node = rclpy.create_node('mllm_publisher')
        face_recognition_node = rclpy.create_node('face_recognition_publisher')

        mllm_publisher = mllm_node.create_publisher(String, '/mllm', 10)
        face_recognition_publisher = face_recognition_node.create_publisher(
            String, '/face_recognition', 10)

        self.executor.add_node(mllm_node)
        self.executor.add_node(face_recognition_node)

        mllm_publisher.publish(String(data='Test MLLM message'))
        face_recognition_publisher.publish(
            String(data='Test Face Recognition message'))

        self.executor.spin_once(timeout_sec=0.1)
        self.executor.spin_once(timeout_sec=0.1)

        assert len(self.state_manager.continuous_queue) == 2
        assert 'Test MLLM message' in self.state_manager.continuous_queue[0][0]
        assert 'Test Face Recognition message' in self.state_manager.continuous_queue[
            1][0]

        # Cleanup
        self.executor.remove_node(mllm_node)
        self.executor.remove_node(face_recognition_node)
        mllm_node.destroy_node()
        face_recognition_node.destroy_node()
