import os
import tempfile

import pytest
import rclpy
from rclpy.exceptions import InvalidParameterTypeException
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from state_manager.state_manager_2 import StateManager2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from transformers import AutoTokenizer


class TestStateManager2:
    """Comprehensive test suite for StateManager2 ROS2 node.

    Tests cover:
    - Topic subscriptions (event, continuous, thought)
    - Token counting and state representation
    - State sequence management and trimming
    - State suffix updates (running actions, robot state info)
    - Thread safety
    - Service calls (sweep state)
    - Long-term memory file writing
    """

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
        # Create temporary files for testing
        self.temp_ltm_file = None
        self.temp_state_file = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.state_manager:
            self.executor.remove_node(self.state_manager)
            self.state_manager.destroy_node()

        # Clean up temporary files
        if self.temp_ltm_file and os.path.exists(self.temp_ltm_file):
            os.remove(self.temp_ltm_file)
        if self.temp_state_file and os.path.exists(self.temp_state_file):
            os.remove(self.temp_state_file)

    @staticmethod
    def token_len(input_str: str, tokenizer: AutoTokenizer) -> int:
        """Return the number of tokens representing an input string."""
        return len(tokenizer(input_str)['input_ids'])

    @staticmethod
    def get_token_length_from_state_seq(state_seq):
        """Helper function to get total token length from state sequence."""
        return len(state_seq)

    def create_test_parameters(self, **kwargs):
        """Create Parameter objects for testing.

        Args:
            **kwargs: Override default parameter values
        """
        # Default parameters
        defaults = {
            'event_topics': ['/asr', '/thought', '/reply_action'],
            'continuous_topics': ['/mllm', '/face_recognition'],
            'thought_topics': ['/agency'],
            'state_max_tokens': 500,
            'state_seq_clear_ratio': 0.5,
            'llm_model_name': 'Qwen/Qwen3-32B',
            'long_term_memory_file_pth': '',
            'state_file_pth': '',
            'enable_sweep_service': True,
        }

        # Override with provided kwargs
        defaults.update(kwargs)

        return [
            Parameter('event_topics', Parameter.Type.STRING_ARRAY,
                      defaults['event_topics']),
            Parameter('continuous_topics', Parameter.Type.STRING_ARRAY,
                      defaults['continuous_topics']),
            Parameter('thought_topics', Parameter.Type.STRING_ARRAY,
                      defaults['thought_topics']),
            Parameter('state_max_tokens', Parameter.Type.INTEGER,
                      defaults['state_max_tokens']),
            Parameter('state_seq_clear_ratio', Parameter.Type.DOUBLE,
                      defaults['state_seq_clear_ratio']),
            Parameter('llm_model_name', Parameter.Type.STRING,
                      defaults['llm_model_name']),
            Parameter('long_term_memory_file_pth', Parameter.Type.STRING,
                      defaults['long_term_memory_file_pth']),
            Parameter('state_file_pth', Parameter.Type.STRING,
                      defaults['state_file_pth']),
            Parameter('enable_sweep_service', Parameter.Type.BOOL,
                      defaults['enable_sweep_service']),
        ]

    def test_state_manager_initialization(self):
        """Test that StateManager2 initializes correctly with parameters."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Verify parameters
        assert self.state_manager.state_max_tokens == 500
        assert self.state_manager.state_seq_clear_ratio == 0.5
        assert len(self.state_manager.event_topics) == 3
        assert len(self.state_manager.continuous_topics) == 2
        assert len(self.state_manager.thought_topics) == 1

        # Verify state components exist
        assert hasattr(self.state_manager, 'state_prefix')
        assert hasattr(self.state_manager, 'state_seq')
        assert hasattr(self.state_manager, '_cached_state_chunks_str')
        assert hasattr(self.state_manager, '_cached_running_actions')
        assert hasattr(self.state_manager, '_cached_robot_state_info')

        # Verify initial state is published
        assert self.state_manager.state_seq.get_num_state_chunks() == 0

    def test_event_topics_subscription(self):
        """Test that event topics are properly subscribed to and processed.

        Event topics are for discrete, important events that should always
        be captured (e.g., speech recognition, user commands, system events).

        Note: In this test config, event_topics includes /asr and
        /reply_action which represent discrete events in the operation cycle.
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers for event topics only
        asr_node = rclpy.create_node('asr_publisher')
        reply_action_node = rclpy.create_node('reply_action_publisher')

        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        reply_action_publisher = reply_action_node.create_publisher(
            String, '/reply_action', 10)

        self.executor.add_node(asr_node)
        self.executor.add_node(reply_action_node)

        # Publish event messages
        asr_publisher.publish(String(data='User said: hello robot'))
        reply_action_publisher.publish(String(data='Saying: Hello human!'))

        # Process event messages
        for _ in range(2):
            self.executor.spin_once(timeout_sec=0.1)

        # Verify both event messages were captured
        assert self.state_manager.state_seq.get_num_state_chunks() == 2

        # Verify event messages are in state
        state = self.state_manager.get_state()
        assert 'User said: hello robot' in state
        assert 'Saying: Hello human!' in state
        assert '/asr' in state
        assert '/reply_action' in state

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(reply_action_node)
        asr_node.destroy_node()
        reply_action_node.destroy_node()

    def test_continuous_topics_subscription(self):
        """Test that continuous topics are properly subscribed to and processed.

        Continuous topics provide ongoing perception data that updates
        frequently (e.g., vision, sensor readings). These may be sampled
        or processed differently than event topics in the future.
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers for continuous topics only
        mllm_node = rclpy.create_node('mllm_publisher')
        face_recognition_node = rclpy.create_node('face_recognition_publisher')

        mllm_publisher = mllm_node.create_publisher(String, '/mllm', 10)
        face_recognition_publisher = face_recognition_node.create_publisher(
            String, '/face_recognition', 10)

        self.executor.add_node(mllm_node)
        self.executor.add_node(face_recognition_node)

        # Publish continuous perception updates
        mllm_publisher.publish(
            String(data='Detected: person standing, table, chair'))
        face_recognition_publisher.publish(
            String(data='Recognized: John Doe, confidence 0.95'))

        # Process continuous messages
        for _ in range(2):
            self.executor.spin_once(timeout_sec=0.1)

        # Verify both continuous messages were captured
        assert self.state_manager.state_seq.get_num_state_chunks() == 2

        # Verify continuous messages are in state
        state = self.state_manager.get_state()
        assert 'Detected: person standing, table, chair' in state
        assert 'Recognized: John Doe' in state
        assert '/mllm' in state
        assert '/face_recognition' in state

        # Cleanup
        self.executor.remove_node(mllm_node)
        self.executor.remove_node(face_recognition_node)
        mllm_node.destroy_node()
        face_recognition_node.destroy_node()

    def test_thought_topics_subscription(self):
        """Test that thought topics are properly subscribed to and processed.

        Thought topics capture the robot's internal reasoning and decision
        making process (e.g., agency decisions, planning). These represent
        the robot's cognitive state and may require special handling.
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publisher for thought topic only
        agency_node = rclpy.create_node('agency_publisher')
        agency_publisher = agency_node.create_publisher(String, '/agency', 10)

        self.executor.add_node(agency_node)

        # Publish thought/reasoning message
        agency_publisher.publish(
            String(data='Decision: approach user to offer assistance'))

        # Process thought message
        self.executor.spin_once(timeout_sec=0.1)

        # Verify thought message was captured
        assert self.state_manager.state_seq.get_num_state_chunks() == 1

        # Verify thought message is in state
        state = self.state_manager.get_state()
        assert 'Decision: approach user to offer assistance' in state
        assert '/agency' in state

        # Cleanup
        self.executor.remove_node(agency_node)
        agency_node.destroy_node()

    def test_unsubscribed_topics_ignored(self):
        """Test that messages from unsubscribed topics are ignored.

        This ensures the state manager only processes configured topics
        and doesn't capture unrelated system messages.
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Add one valid message first
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        asr_publisher.publish(String(data='Valid message'))
        self.executor.spin_once(timeout_sec=0.1)

        # Verify we have 1 chunk
        assert self.state_manager.state_seq.get_num_state_chunks() == 1

        # Create publisher for unrelated topic
        unrelated_node = rclpy.create_node('unrelated_publisher')
        unrelated_publisher = unrelated_node.create_publisher(
            String, '/unrelated_topic', 10)
        self.executor.add_node(unrelated_node)

        # Publish to unrelated topic
        unrelated_publisher.publish(String(data='Should be ignored'))
        self.executor.spin_once(timeout_sec=0.1)

        # Verify unrelated message was not added to state
        assert self.state_manager.state_seq.get_num_state_chunks() == 1
        state = self.state_manager.get_state()
        assert 'Should be ignored' not in state
        assert 'Valid message' in state

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(unrelated_node)
        asr_node.destroy_node()
        unrelated_node.destroy_node()

    def test_mixed_topic_types_processing(self):
        """Test processing messages from different topic types together.

        Verifies that event, continuous, and thought topics can all be
        processed correctly in the same state manager instance, maintaining
        proper ordering and content.
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Create publishers for all three types
        asr_node = rclpy.create_node('asr_publisher')
        mllm_node = rclpy.create_node('mllm_publisher')
        agency_node = rclpy.create_node('agency_publisher')

        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        mllm_publisher = mllm_node.create_publisher(String, '/mllm', 10)
        agency_publisher = agency_node.create_publisher(String, '/agency', 10)

        self.executor.add_node(asr_node)
        self.executor.add_node(mllm_node)
        self.executor.add_node(agency_node)

        # Publish messages from different types in sequence
        asr_publisher.publish(String(data='Event: user command'))
        mllm_publisher.publish(String(data='Continuous: scene update'))
        agency_publisher.publish(String(data='Thought: planning action'))

        # Process all messages
        for _ in range(3):
            self.executor.spin_once(timeout_sec=0.1)

        # Verify all messages captured
        assert self.state_manager.state_seq.get_num_state_chunks() == 3

        # Verify all message types present in state
        state = self.state_manager.get_state()
        assert 'Event: user command' in state
        assert 'Continuous: scene update' in state
        assert 'Thought: planning action' in state

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(mllm_node)
        self.executor.remove_node(agency_node)
        asr_node.destroy_node()
        mllm_node.destroy_node()
        agency_node.destroy_node()

    def test_state_manager_state_token_lengths(self):
        """Test that token counts are reasonably accurate.

        Note: Token counts are approximate due to template overhead not being
        included in state_chunks_num_tokens for performance reasons. We allow
        a small margin of error while ensuring the counts are close enough for
        practical use (overflow detection).
        """
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        tokenizer = AutoTokenizer.from_pretrained(
            self.state_manager.llm_model_name)

        # Test initial state (prefix + empty chunks + suffix)
        initial_state = self.state_manager.get_state()
        calculated_initial = len(self.state_manager)
        actual_initial = self.token_len(initial_state, tokenizer)

        # Allow small margin for template overhead (~20 tokens)
        margin = 20
        assert abs(calculated_initial - actual_initial) <= margin, (
            f"Token length difference too large: "
            f"calculated={calculated_initial}, actual={actual_initial}, "
            f"difference={abs(calculated_initial - actual_initial)}")

        # Add some messages and verify token counting
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        # Publish messages
        for idx in range(5):
            asr_publisher.publish(String(data=f'Test message {idx}'))
            self.executor.spin_once(timeout_sec=0.1)

        # Check that all messages are in the state sequence
        assert self.state_manager.state_seq.get_num_state_chunks() == 5

        # Test that token length is reasonably accurate
        state = self.state_manager.get_state()
        calculated_len = len(self.state_manager)
        actual_len = self.token_len(state, tokenizer)

        # Allow small margin for template overhead
        assert abs(calculated_len - actual_len) <= margin, (
            f"Token length difference too large: "
            f"calculated={calculated_len}, actual={actual_len}, "
            f"difference={abs(calculated_len - actual_len)}")

        # Verify component token counts sum correctly
        prefix_tokens = self.state_manager.state_prefix_num_tokens
        chunks_tokens = self.state_manager.state_chunks_num_tokens
        suffix_tokens = self.state_manager.state_suffix_num_tokens
        total_tokens = prefix_tokens + chunks_tokens + suffix_tokens

        assert calculated_len == total_tokens, \
            "Component token counts don't sum to total"

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_state_sequence_trimming(self):
        """Test state sequence trimming when max tokens exceeded."""
        # Use small token limit for easier testing
        test_params = self.create_test_parameters(state_max_tokens=200)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publisher
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        # Start with empty state sequence
        assert self.state_manager.state_seq.get_num_state_chunks() == 0

        # Add messages until we exceed the token limit
        message_count = 0

        # Keep adding messages until we're close to capacity
        max_allowed = self.state_manager.state_max_tokens - 50
        while len(self.state_manager) < max_allowed:
            test_message = f'Test ASR message {message_count}'
            asr_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)
            message_count += 1

            # Safety check to prevent infinite loop
            if message_count > 100:
                break

        chunks_before_trim = self.state_manager.state_seq.get_num_state_chunks(
        )
        print(f"Chunks before triggering trim: {chunks_before_trim}")
        print(f"Token count before trim: {len(self.state_manager)}")

        # Add messages to trigger trimming
        for i in range(5):
            test_message = f'Trigger trim message {i}'
            asr_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

        # Verify trimming occurred
        chunks_after_trim = self.state_manager.state_seq.get_num_state_chunks()
        print(f"Chunks after trim: {chunks_after_trim}")
        print(f"Token count after trim: {len(self.state_manager)}")

        # After trimming, we should have fewer chunks
        assert chunks_after_trim < chunks_before_trim or \
               chunks_after_trim == chunks_before_trim, \
               "State sequence should have been trimmed or stayed same"

        # Verify token limit is still respected
        current_tokens = len(self.state_manager)
        max_tokens = self.state_manager.state_max_tokens
        assert current_tokens <= max_tokens, (
            f"State token count {current_tokens} exceeds max {max_tokens}")

        # Verify newest messages are kept (FIFO - oldest removed first)
        state = self.state_manager.get_state()
        assert 'Trigger trim message 4' in state, \
            "Newest message should be kept after trimming"

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_state_suffix_running_actions_update(self):
        """Test that running actions updates are reflected in state suffix."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publisher for running actions
        action_node = rclpy.create_node('action_publisher')
        running_actions_publisher = action_node.create_publisher(
            String, '/action_running', 10)
        self.executor.add_node(action_node)

        # Test initial state (default running actions)
        assert self.state_manager._cached_running_actions == 'None'

        # Publish running action
        running_actions_publisher.publish(String(data='move_forward'))
        self.executor.spin_once(timeout_sec=0.1)

        # Verify cached running actions updated
        assert self.state_manager._cached_running_actions == 'move_forward'

        # Verify state suffix contains the action
        state_suffix = self.state_manager.state_suffix
        assert 'move_forward' in state_suffix

        # Test empty running actions (should reset to default)
        running_actions_publisher.publish(String(data=''))
        self.executor.spin_once(timeout_sec=0.1)

        assert self.state_manager._cached_running_actions == 'None'

        # Cleanup
        self.executor.remove_node(action_node)
        action_node.destroy_node()

    def test_state_suffix_robot_info_update(self):
        """Test that robot state info updates are reflected in state suffix."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publisher for robot state info
        robot_info_node = rclpy.create_node('robot_info_publisher')
        robot_info_publisher = robot_info_node.create_publisher(
            String, '/robot_state_info', 10)
        self.executor.add_node(robot_info_node)

        # Test initial state
        initial_info = self.state_manager._cached_robot_state_info
        assert initial_info == 'Robot state information not yet received'

        # Publish robot state info
        robot_info_publisher.publish(
            String(data='Battery: 85%, Location: Room A'))
        self.executor.spin_once(timeout_sec=0.1)

        # Verify cached robot state info updated
        assert 'Battery: 85%' in self.state_manager._cached_robot_state_info

        # Verify state suffix contains the info
        state_suffix = self.state_manager.state_suffix
        assert 'Battery: 85%' in state_suffix

        # Cleanup
        self.executor.remove_node(robot_info_node)
        robot_info_node.destroy_node()

    def test_sweep_state_service(self):
        """Test the sweep_state service clears all state chunks."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Add some messages first
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        for i in range(5):
            asr_publisher.publish(String(data=f'Test message {i}'))
            self.executor.spin_once(timeout_sec=0.1)

        # Verify chunks exist
        assert self.state_manager.state_seq.get_num_state_chunks() == 5
        assert len(self.state_manager._cached_state_chunks_str) > 0

        # Create service client
        client_node = rclpy.create_node('sweep_client')
        self.executor.add_node(client_node)
        client = client_node.create_client(Trigger, 'sweep_state')

        # Wait for service to be available
        assert client.wait_for_service(timeout_sec=2.0), \
            "Sweep state service not available"

        # Call sweep service
        request = Trigger.Request()
        future = client.call_async(request)

        # Spin until response received
        while not future.done():
            self.executor.spin_once(timeout_sec=0.1)

        response = future.result()

        # Verify response
        assert response.success is True
        assert 'Cleared 5 chunks' in response.message

        # Verify state sequence is cleared
        assert self.state_manager.state_seq.get_num_state_chunks() == 0
        assert self.state_manager._cached_state_chunks_str == ''

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(client_node)
        asr_node.destroy_node()
        client_node.destroy_node()

    def test_long_term_memory_file_writing(self):
        """Test state chunks are written to long-term memory file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            self.temp_ltm_file = f.name

        test_params = self.create_test_parameters(
            long_term_memory_file_pth=self.temp_ltm_file)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publisher
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        # Publish messages
        test_messages = ['First message', 'Second message', 'Third message']
        for msg in test_messages:
            asr_publisher.publish(String(data=msg))

        # Process all published messages
        # Spin a few extra times to ensure all callbacks are processed
        for _ in range(len(test_messages) + 2):
            self.executor.spin_once(timeout_sec=0.1)

        # Read the LTM file
        with open(self.temp_ltm_file, 'r') as f:
            ltm_content = f.read()

        # Verify all messages were written
        for msg in test_messages:
            assert msg in ltm_content, f"Message '{msg}' not found in LTM file"

        # Verify proper formatting
        assert 'topic: /asr' in ltm_content
        assert 'ts:' in ltm_content
        assert 'data:' in ltm_content
        assert '---' in ltm_content

        # Verify chunks are separated by newlines
        assert ltm_content.count('---\n') == 3, \
            "Chunks should be separated with proper newlines"

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_state_file_writing(self):
        """Test that current state is written to state file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            self.temp_state_file = f.name

        test_params = self.create_test_parameters(
            state_file_pth=self.temp_state_file)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Add a message
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        asr_publisher.publish(String(data='Test state file message'))
        self.executor.spin_once(timeout_sec=0.1)

        # Read the state file
        with open(self.temp_state_file, 'r') as f:
            state_content = f.read()

        # Verify message is in state file
        assert 'Test state file message' in state_content

        # Verify it contains all state components
        assert 'Robot description' in state_content  # From state_prefix
        assert 'topic: /asr' in state_content  # From state_chunks

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_concurrent_message_processing(self):
        """Test thread safety with concurrent message processing."""
        # Use larger token limit to accommodate all test messages
        test_params = self.create_test_parameters(state_max_tokens=5000)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Create multiple publishers
        nodes = []
        publishers = []
        topics = ['/asr', '/mllm', '/agency']

        for topic in topics:
            node = rclpy.create_node(f'publisher_{topic.replace("/", "")}')
            pub = node.create_publisher(String, topic, 10)
            self.executor.add_node(node)
            nodes.append(node)
            publishers.append((pub, topic))

        # Publish many messages rapidly from different topics
        num_messages = 10
        for i in range(num_messages):
            for pub, topic in publishers:
                pub.publish(String(data=f'Message {i} from {topic}'))

        # Process all published messages
        # Need to spin enough times to process all messages
        expected_chunks = num_messages * len(topics)
        for _ in range(expected_chunks):
            self.executor.spin_once(timeout_sec=0.1)

        # Verify all messages were processed
        actual_chunks = self.state_manager.state_seq.get_num_state_chunks()
        assert actual_chunks == expected_chunks, \
            f"Expected {expected_chunks} chunks, got {actual_chunks}"

        # Verify state consistency
        state = self.state_manager.get_state()
        state_token_len = len(self.state_manager)

        # Token length should be reasonably consistent
        # (allow small margin for template overhead)
        tokenizer = AutoTokenizer.from_pretrained(
            self.state_manager.llm_model_name)
        actual_token_len = self.token_len(state, tokenizer)
        margin = 20
        assert abs(state_token_len - actual_token_len) <= margin, (
            f"State token length difference too large after concurrent "
            f"processing: calculated={state_token_len}, "
            f"actual={actual_token_len}, "
            f"difference={abs(state_token_len - actual_token_len)}")

        # Cleanup
        for node in nodes:
            self.executor.remove_node(node)
            node.destroy_node()

    def test_state_components_integration(self):
        """Test all state components integrate correctly."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Get initial state
        state = self.state_manager.get_state()

        # Verify state contains prefix
        assert 'Robot description' in state
        assert 'Personality profile' in state

        # Add a message and verify chunks section
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        asr_publisher.publish(String(data='Integration test message'))
        self.executor.spin_once(timeout_sec=0.1)

        state = self.state_manager.get_state()
        assert 'Integration test message' in state

        # Update running actions and verify suffix
        action_node = rclpy.create_node('running_actions_publisher')
        running_actions_publisher = action_node.create_publisher(
            String, '/action_running', 10)
        self.executor.add_node(action_node)

        running_actions_publisher.publish(String(data='test_action'))
        self.executor.spin_once(timeout_sec=0.1)

        state = self.state_manager.get_state()
        assert 'test_action' in state

        # Verify all components are present in correct order
        prefix_idx = state.find('Robot description')
        chunks_idx = state.find('Integration test message')
        suffix_idx = state.find('test_action')

        assert prefix_idx < chunks_idx < suffix_idx, \
            "State components not in correct order: prefix -> chunks -> suffix"

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(action_node)
        asr_node.destroy_node()
        action_node.destroy_node()

    def test_format_state_chunk(self):
        """Test the static format_state_chunk method."""
        chunk = StateManager2.format_state_chunk('/test_topic',
                                                 '2025-10-07 12:30:45',
                                                 'Test message data')

        # Verify formatting
        assert 'topic: /test_topic' in chunk
        assert 'ts: 2025-10-07 12:30:45' in chunk
        assert 'data: Test message data' in chunk
        assert chunk.endswith('---')
        assert chunk.startswith('\n')

    def test_clear_ratio_behavior(self):
        """Test that clear ratio correctly determines how many chunks to keep.

        When state sequence exceeds token limit, it should trim to keep
        only the configured ratio of the newest messages.
        """
        # Test with 0.3 ratio (keep 30% newest chunks after clearing)
        test_params = self.create_test_parameters(state_max_tokens=200,
                                                  state_seq_clear_ratio=0.3)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Add messages
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        # Add enough messages to approach token limit
        message_count = 0
        max_allowed = self.state_manager.state_max_tokens - 50
        while len(self.state_manager) < max_allowed:
            asr_publisher.publish(String(data=f'Fill message {message_count}'))
            self.executor.spin_once(timeout_sec=0.05)
            message_count += 1

            # Safety check
            if message_count > 100:
                break

        chunks_before_clear = (
            self.state_manager.state_seq.get_num_state_chunks())
        print(f"Chunks before clearing: {chunks_before_clear}")
        print(f"Token count before clear: {len(self.state_manager)}")

        # Trigger clearing by adding a large message
        large_message = 'Trigger clear: ' + 'x' * 100
        asr_publisher.publish(String(data=large_message))
        self.executor.spin_once(timeout_sec=0.1)

        chunks_after_clear = (
            self.state_manager.state_seq.get_num_state_chunks())
        print(f"Chunks after clearing: {chunks_after_clear}")
        print(f"Token count after clear: {len(self.state_manager)}")

        # Verify clearing occurred
        assert chunks_after_clear < chunks_before_clear, \
            "State sequence should have been cleared"

        # Verify approximately 30% of chunks were kept
        # (plus the new message that triggered the clear)
        expected_kept = int(chunks_before_clear * 0.3)
        # Allow some tolerance due to token-based clearing logic
        tolerance = 2
        assert abs(chunks_after_clear - expected_kept - 1) <= tolerance, (
            f"Expected ~{expected_kept + 1} chunks (30% + trigger message), "
            f"got {chunks_after_clear}")

        # Verify newest messages are kept (FIFO - oldest removed first)
        state = self.state_manager.get_state()
        assert large_message in state, "Trigger message should be kept"

        # Verify old messages were removed
        assert 'Fill message 0' not in state, \
            "Oldest messages should have been removed"

        # Verify token limit is still respected
        assert len(self.state_manager) <= self.state_manager.state_max_tokens

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_no_topics_raises_error(self):
        """Test that initialization fails when no topics are specified.

        Note: ROS2 parameter system raises InvalidParameterTypeException
        when passing empty lists for STRING_ARRAY parameters, before
        StateManager2's own validation can run.
        """
        test_params = self.create_test_parameters(event_topics=[],
                                                  continuous_topics=[],
                                                  thought_topics=[])

        with pytest.raises((IOError, InvalidParameterTypeException)):
            self.state_manager = StateManager2(parameter_overrides=test_params)

    def test_cached_state_chunks_consistency(self):
        """Test _cached_state_chunks_str consistency with state_seq.

        Tests cache consistency across multiple scenarios:
        1. Initial empty state
        2. After adding messages
        3. After sequence clearing (overflow)
        4. After complete sweep
        """
        # Use moderate token limit - small enough to trigger clearing later,
        # but large enough to hold initial test messages
        test_params = self.create_test_parameters(state_max_tokens=300,
                                                  state_seq_clear_ratio=0.5)
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        # Test 1: Initial empty state - cache should be empty
        assert self.state_manager._cached_state_chunks_str == '', \
            "Initial cache should be empty"
        chunks_from_seq = '\n'.join(
            [chunk.chunk for chunk in self.state_manager.state_seq])
        assert self.state_manager._cached_state_chunks_str == chunks_from_seq, \
            "Empty cache should match empty sequence"

        # Test 2: After adding messages - cache should match all chunks
        for i in range(3):
            asr_publisher.publish(String(data=f'Message {i}'))

        # Process all messages - add extra spins for reliability
        for _ in range(5):
            self.executor.spin_once(timeout_sec=0.1)

        cached = self.state_manager._cached_state_chunks_str
        assert 'Message 0' in cached
        assert 'Message 1' in cached
        assert 'Message 2' in cached

        chunks_from_seq = '\n'.join(
            [chunk.chunk for chunk in self.state_manager.state_seq])
        assert cached == chunks_from_seq, \
            "Cache should match sequence after adding messages"

        # Test 3: After sequence clearing - cache should update correctly
        # Add messages until we're near the limit
        message_count = 3
        max_allowed = self.state_manager.state_max_tokens - 50
        while len(self.state_manager) < max_allowed:
            asr_publisher.publish(String(data=f'Fill {message_count}'))
            self.executor.spin_once(timeout_sec=0.05)
            message_count += 1
            if message_count > 100:
                break

        chunks_before_clear = (
            self.state_manager.state_seq.get_num_state_chunks())

        # Trigger clearing with large message
        trigger_msg = 'Trigger: ' + 'x' * 100
        asr_publisher.publish(String(data=trigger_msg))
        self.executor.spin_once(timeout_sec=0.1)

        chunks_after_clear = (
            self.state_manager.state_seq.get_num_state_chunks())

        # Verify clearing occurred
        assert chunks_after_clear < chunks_before_clear, \
            "Clearing should have occurred"

        # Verify cache matches sequence after clearing
        cached_after_clear = self.state_manager._cached_state_chunks_str
        chunks_from_seq_after_clear = '\n'.join(
            [chunk.chunk for chunk in self.state_manager.state_seq])
        assert cached_after_clear == chunks_from_seq_after_clear, \
            "Cache should match sequence after clearing"

        # Verify old messages were removed from cache
        assert 'Message 0' not in cached_after_clear, \
            "Old messages should not be in cache after clearing"

        # Verify trigger message is in cache
        assert trigger_msg in cached_after_clear, \
            "Trigger message should be in cache"

        # Test 4: After complete sweep - cache should be empty again
        client_node = rclpy.create_node('sweep_client')
        self.executor.add_node(client_node)
        client = client_node.create_client(Trigger, 'sweep_state')

        assert client.wait_for_service(timeout_sec=2.0), \
            "Sweep service not available"

        request = Trigger.Request()
        future = client.call_async(request)

        while not future.done():
            self.executor.spin_once(timeout_sec=0.1)

        response = future.result()
        assert response.success is True

        # Verify cache is empty after sweep
        assert self.state_manager._cached_state_chunks_str == '', \
            "Cache should be empty after sweep"
        assert self.state_manager.state_seq.get_num_state_chunks() == 0, \
            "Sequence should be empty after sweep"

        chunks_from_seq_after_sweep = '\n'.join(
            [chunk.chunk for chunk in self.state_manager.state_seq])
        cache_after_sweep = self.state_manager._cached_state_chunks_str
        assert cache_after_sweep == chunks_from_seq_after_sweep, \
            "Empty cache should match empty sequence after sweep"

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(client_node)
        asr_node.destroy_node()
        client_node.destroy_node()

    def test_timestamp_in_chunks(self):
        """Test that timestamps are correctly included in state chunks."""
        test_params = self.create_test_parameters()
        self.state_manager = StateManager2(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Add a message
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        asr_publisher.publish(String(data='Timestamp test'))

        # Process message with extra spins for reliability
        for _ in range(3):
            self.executor.spin_once(timeout_sec=0.1)

        # Verify timestamp exists in state
        state = self.state_manager.get_state()
        assert 'ts:' in state

        # Verify timestamp format (YYYY-MM-DD HH:MM:SS)
        import re
        timestamp_pattern = r'ts: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, state), \
            "Timestamp not found in expected format"

        # Verify timestamp is stored in StateChunk
        chunk = list(self.state_manager.state_seq)[0]
        assert chunk.ts > 0, "StateChunk should have timestamp"

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()
