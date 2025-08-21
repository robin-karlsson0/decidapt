import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from state_manager.state_manager import StateManager
from std_msgs.msg import String
from transformers import AutoTokenizer


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

    @staticmethod
    def token_len(input_str: str, tokenizer: AutoTokenizer) -> int:
        """Return the number of tokens representing an input string."""
        return len(tokenizer(input_str)['input_ids'])

    @staticmethod
    def get_token_length_from_queue(queue):
        """Helper function to get total token length from a queue with tuples
        (message, timestamp, token_length)."""
        return sum(item[2] for item in queue)

    def create_test_parameters(self):
        """Create Parameter objects for testing."""
        return [
            Parameter('event_topics', Parameter.Type.STRING_ARRAY,
                      ['/asr', '/thought', '/reply_action']),
            Parameter('continuous_topics', Parameter.Type.STRING_ARRAY,
                      ['/mllm', '/face_recognition']),
            Parameter('thought_topics', Parameter.Type.STRING_ARRAY, [
                '/agency',
            ]),
            Parameter('event_queue_max_tokens', Parameter.Type.INTEGER, 500),
            Parameter('continuous_queue_max_tokens', Parameter.Type.INTEGER,
                      400),
            Parameter('thought_queue_max_tokens', Parameter.Type.INTEGER, 400),
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
        assert 'Test Face Recognition message' in self.state_manager.continuous_queue[  # noqa: E501
            1][0]

        # Cleanup
        self.executor.remove_node(mllm_node)
        self.executor.remove_node(face_recognition_node)
        mllm_node.destroy_node()
        face_recognition_node.destroy_node()

    def test_state_manager_thought_subscriptions(self):
        """Test that thought topics are properly subscribed to."""
        # Initialize StateManager with custom parameters
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers
        thought_node = rclpy.create_node('thought_publisher')
        thought_publisher = thought_node.create_publisher(
            String, '/agency', 10)

        self.executor.add_node(thought_node)

        thought_publisher.publish(String(data='Test Thought message'))

        self.executor.spin_once(timeout_sec=0.1)

        assert len(self.state_manager.thought_queue) == 1
        assert 'Test Thought message' in self.state_manager.thought_queue[0][0]

        # Cleanup
        self.executor.remove_node(thought_node)
        thought_node.destroy_node()

    def test_state_manager_state_token_lengths(self):
        """Test that number of tokens are counted correctly."""
        # Initialize StateManager with custom parameters
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        tokenizer = AutoTokenizer.from_pretrained(
            self.state_manager.llm_model_name)

        # Initialize mock publishers
        asr_node = rclpy.create_node('asr_publisher')
        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        self.executor.add_node(asr_node)

        thought_node = rclpy.create_node('thought_publisher')
        thought_publisher = thought_node.create_publisher(
            String, '/agency', 10)
        self.executor.add_node(thought_node)

        #################
        #  Event queue
        #################
        for idx in range(10):
            asr_publisher.publish(String(data=f'Test ASR message {idx}'))
            self.executor.spin_once(timeout_sec=0.1)

        #################
        #  Thought queue
        #################
        for idx in range(5):
            thought_publisher.publish(
                String(data=f'Test thought message {idx}'))
            self.executor.spin_once(timeout_sec=0.1)

        # Check that all messages are in the queues
        assert len(self.state_manager.event_queue) == 10
        assert len(self.state_manager.thought_queue) == 5

        # Test that len() state representation token length is correct
        state = self.state_manager.get_state()
        assert len(self.state_manager) == self.token_len(state, tokenizer)

        # Cleanup
        self.executor.remove_node(asr_node)
        asr_node.destroy_node()

    def test_state_manager_queue_trimming(self):
        """Test that queue trimming works correctly for all queues."""
        # Initialize StateManager with small token limits for easier testing
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Initialize mock publishers
        asr_node = rclpy.create_node('asr_publisher')
        mllm_node = rclpy.create_node('mllm_publisher')
        thought_node = rclpy.create_node('thought_publisher')

        asr_publisher = asr_node.create_publisher(String, '/asr', 10)
        mllm_publisher = mllm_node.create_publisher(String, '/mllm', 10)
        thought_publisher = thought_node.create_publisher(
            String, '/agency', 10)

        self.executor.add_node(asr_node)
        self.executor.add_node(mllm_node)
        self.executor.add_node(thought_node)

        ##############################
        # Test event queue trimming
        ##############################

        # Start with empty queue
        assert len(self.state_manager.event_queue) == 0, \
            'Event queue should be empty'

        # Add messages until we exceed the token limit
        message_count = 0
        total_tokens = 0

        # Keep adding messages until we start trimming
        while total_tokens < self.state_manager.event_queue_max_tokens:
            test_message = f'Test ASR message {message_count}'
            asr_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

            total_tokens += self.state_manager.event_queue[-1][2]
            message_count += 1

            # Safety check to avoid infinite loop
            if message_count > 100:
                break

        initial_queue_size = len(self.state_manager.event_queue)
        print(f"Event queue size before trimming: {initial_queue_size}")
        print(f"Total tokens in event queue before trimming: {total_tokens}")
        print(f"Max tokens allowed in event queue: \
                {self.state_manager.event_queue_max_tokens}")

        # Verify queue is at or near capacity
        current_tokens = sum(item[2]
                             for item in self.state_manager.event_queue)
        assert current_tokens <= self.state_manager.event_queue_max_tokens, \
            'Event queue should be at or near capacity'

        # Add one more message to trigger trimming
        test_message = f'Test ASR message  {message_count}'
        asr_publisher.publish(String(data=test_message))
        self.executor.spin_once(timeout_sec=0.1)

        # Verify trimming occurred
        final_queue_size = len(self.state_manager.event_queue)

        # The new message should have removed the oldest message
        assert final_queue_size == initial_queue_size, \
            'Event queue should have been trimmed (1)'

        # Verify token limit is still respected
        final_tokens = self.get_token_length_from_queue(
            self.state_manager.event_queue)

        # Add several more messages to trigger trimming
        for i in range(5):
            test_message = f'Large test ASR message {message_count}' * 10
            asr_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

        # Verify trimming occurred
        final_queue_size = len(self.state_manager.event_queue)
        print(f"Final event queue size after trimming: {final_queue_size}")

        # Queue should have been trimmed (fewer items than before)
        assert final_queue_size <= initial_queue_size, \
            'Event queue should have been trimmed (2)'

        # Verify token limit is still respected
        final_tokens = self.get_token_length_from_queue(
            self.state_manager.event_queue)
        assert final_tokens <= self.state_manager.event_queue_max_tokens, \
            'Event queue token limit should still be respected'

        # Verify that newest messages are kept (FIFO - oldest removed first)
        if len(self.state_manager.event_queue) > 0:
            last_chunk = self.state_manager.event_queue[-1][0]
            assert 'Large test ASR message' in last_chunk, \
                "Newest messages should be retained"

        ###################################
        # Test continuous queue trimming
        ###################################

        # Start with empty continuous queue
        assert len(self.state_manager.continuous_queue) == 0, \
            'Continuous queue should be empty'

        # Add messages until we exceed the token limit
        message_count = 0
        total_tokens = 0

        # Keep adding messages until we start trimming
        while total_tokens < self.state_manager.continuous_queue_max_tokens:
            test_message = f'Test MLLM message {message_count}'
            mllm_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

            total_tokens += self.state_manager.continuous_queue[-1][2]
            message_count += 1

            # Safety check to avoid infinite loop
            if message_count > 50:
                break

        initial_queue_size = len(self.state_manager.continuous_queue)
        print(f"Continuous queue size before trimming: {initial_queue_size}")
        print(
            f"Total tokens in continuous queue before trimming: {total_tokens}"
        )
        print(f"Max tokens allowed in continuous queue: \
                {self.state_manager.continuous_queue_max_tokens}")

        # Verify queue is at or near capacity
        current_tokens = self.get_token_length_from_queue(
            self.state_manager.continuous_queue)
        assert current_tokens <= self.state_manager.continuous_queue_max_tokens, 'Continuous queue should be at or near capacity'  # noqa: E501

        # Add several more messages to trigger trimming
        for i in range(5):
            test_message = f'Large test MLLM message {i} ' * 10
            mllm_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

        # Verify trimming occurred
        final_queue_size = len(self.state_manager.continuous_queue)
        print(
            f"Final continuous queue size after trimming: {final_queue_size}")

        # Queue should have been trimmed
        assert final_queue_size <= initial_queue_size, \
            "Continuous queue should have been trimmed"

        # Verify token limit is still respected
        final_tokens = self.get_token_length_from_queue(
            self.state_manager.continuous_queue)
        assert final_tokens <= self.state_manager.continuous_queue_max_tokens, \
            'Continuous queue token limit should still be respected'

        # Verify that newest messages are kept (FIFO - oldest removed first)
        if len(self.state_manager.continuous_queue) > 0:
            last_chunk = self.state_manager.continuous_queue[-1][0]
            assert 'Large test MLLM message' in last_chunk, \
                "Newest messages should be retained"

        ###############################
        # Test thought queue trimming
        ###############################

        # Start with empty thought queue
        assert len(self.state_manager.thought_queue) == 0, \
            'Thought queue should be empty'

        # Add messages until we exceed the token limit
        message_count = 0
        total_tokens = 0

        # Keep adding messages until we start trimming
        while total_tokens < self.state_manager.thought_queue_max_tokens:
            test_message = f'Test agency message {message_count}'
            thought_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

            total_tokens += self.state_manager.thought_queue[-1][2]
            message_count += 1

            # Safety check to avoid infinite loop
            if message_count > 50:
                break

        initial_queue_size = len(self.state_manager.thought_queue)
        print(f"Thought queue size before trimming: {initial_queue_size}")
        print(f"Total tokens in thought queue before trimming: {total_tokens}")
        print(f"Max tokens allowed in thought queue: \
                {self.state_manager.thought_queue_max_tokens}")

        # Verify queue is at or near capacity
        current_tokens = self.get_token_length_from_queue(
            self.state_manager.thought_queue)
        assert current_tokens <= self.state_manager.thought_queue_max_tokens, \
            'Thought queue should be at or near capacity'

        # Add several more messages to trigger trimming
        for i in range(5):
            test_message = f'Large test agency message {i} ' * 10
            thought_publisher.publish(String(data=test_message))
            self.executor.spin_once(timeout_sec=0.1)

        # Verify trimming occurred
        final_queue_size = len(self.state_manager.thought_queue)
        print(f"Final thought queue size after trimming: {final_queue_size}")

        # Queue should have been trimmed
        assert final_queue_size <= initial_queue_size, \
            "Thought queue should have been trimmed"

        # Verify token limit is still respected
        final_tokens = self.get_token_length_from_queue(
            self.state_manager.thought_queue)
        assert final_tokens <= self.state_manager.thought_queue_max_tokens, \
            'Thought queue token limit should still be respected'

        # Verify that newest messages are kept (FIFO - oldest removed first)
        if len(self.state_manager.thought_queue) > 0:
            last_chunk = self.state_manager.thought_queue[-1][0]
            assert 'Large test agency message' in last_chunk, \
                "Newest messages should be retained"

        #################
        # Test Edge Cases
        #################

        # Test adding a single message that exceeds the token limit
        test_message = 'Very very large test ASR message' * int(1e4)
        asr_publisher.publish(String(data=test_message))
        self.executor.spin_once(timeout_sec=0.1)

        final_tokens = self.get_token_length_from_queue(
            self.state_manager.event_queue)
        assert final_tokens <= self.state_manager.event_queue_max_tokens, \
            'Event queue token limit should still be respected after very very large message'  # noqa: E501

        # Cleanup
        self.executor.remove_node(asr_node)
        self.executor.remove_node(mllm_node)
        self.executor.remove_node(thought_node)
        asr_node.destroy_node()
        mllm_node.destroy_node()
        thought_node.destroy_node()

    def test_add_to_chunk_dt_functionality(self):
        """Test the add_to_chunk_dt method that adds 'time ago' to ts."""
        from datetime import datetime, timedelta

        # Initialize StateManager with custom parameters
        test_params = self.create_test_parameters()
        self.state_manager = StateManager(parameter_overrides=test_params)
        self.executor.add_node(self.state_manager)

        # Test chunk without timestamp (should remain unchanged)
        chunk_no_ts = """topic: /test
data: Test message without timestamp
---"""
        result = self.state_manager.add_to_chunk_dt(chunk_no_ts)
        assert result == chunk_no_ts, "Chunk without timestamp should remain unchanged"  # noqa: E501

        # Test chunk with recent timestamp
        current_time = datetime.now()
        recent_time = current_time - timedelta(minutes=5, seconds=30)
        recent_ts_str = recent_time.strftime("%Y-%m-%d %H:%M:%S")

        chunk_recent = f"""topic: /test_recent
ts: {recent_ts_str}
data: Recent test message
---"""

        result_recent = self.state_manager.add_to_chunk_dt(chunk_recent)
        assert f"ts: {recent_ts_str} (" in result_recent, "Should add time ago to recent timestamp"  # noqa: E501
        assert "5m" in result_recent and "30s" in result_recent, "Should show correct time difference"  # noqa: E501
        assert "ago)" in result_recent, "Should end with 'ago)'"

        # Test chunk with old timestamp (days ago)
        old_time = current_time - timedelta(days=2, hours=3, minutes=15)
        old_ts_str = old_time.strftime("%Y-%m-%d %H:%M:%S")

        chunk_old = f"""topic: /test_old
ts: {old_ts_str}
data: Old test message
---"""

        result_old = self.state_manager.add_to_chunk_dt(chunk_old)
        assert f"ts: {old_ts_str} (" in result_old, "Should add time ago to old timestamp"  # noqa: E501
        assert "2d" in result_old, "Should show days in time difference"
        assert "3h" in result_old, "Should show hours in time difference"
        assert "15m" in result_old, "Should show minutes in time difference"
        assert "ago)" in result_old, "Should end with 'ago)'"

        # Test chunk with very recent timestamp (seconds only)
        very_recent_time = current_time - timedelta(seconds=45)
        very_recent_ts_str = very_recent_time.strftime("%Y-%m-%d %H:%M:%S")

        chunk_very_recent = f"""topic: /test_very_recent
ts: {very_recent_ts_str}
data: Very recent test message
---"""

        result_very_recent = self.state_manager.add_to_chunk_dt(
            chunk_very_recent)
        assert f"ts: {very_recent_ts_str} (" in result_very_recent, "Should add time ago to very recent timestamp"  # noqa: E501
        assert "45s ago)" in result_very_recent, "Should show seconds only for very recent timestamp"  # noqa: E501

        # Test chunk with multiple lines and preserve formatting
        multi_line_chunk = f"""topic: /test_multiline
ts: {recent_ts_str}
data: This is a test message
with multiple lines
and various content
---"""

        result_multi = self.state_manager.add_to_chunk_dt(multi_line_chunk)
        assert f"ts: {recent_ts_str} (" in result_multi, "Should add time ago to multi-line chunk"  # noqa: E501
        assert "with multiple lines" in result_multi, "Should preserve multi-line content"  # noqa: E501
        assert "and various content" in result_multi, "Should preserve all content"  # noqa: E501

        # Test static method _format_time_ago directly
        from datetime import timedelta

        # Test zero time difference
        zero_diff = timedelta(seconds=0)
        assert self.state_manager._format_time_ago(zero_diff) == "0s ago"

        # Test seconds only
        seconds_diff = timedelta(seconds=42)
        assert self.state_manager._format_time_ago(seconds_diff) == "42s ago"

        # Test minutes and seconds
        min_sec_diff = timedelta(minutes=3, seconds=25)
        assert self.state_manager._format_time_ago(
            min_sec_diff) == "3m 25s ago"

        # Test hours, minutes, and seconds
        hour_min_sec_diff = timedelta(hours=2, minutes=15, seconds=30)
        assert self.state_manager._format_time_ago(
            hour_min_sec_diff) == "2h 15m 30s ago"

        # Test days, hours, minutes, and seconds
        full_diff = timedelta(days=1, hours=5, minutes=45, seconds=12)
        assert self.state_manager._format_time_ago(
            full_diff) == "1d 5h 45m 12s ago"

        # Test edge case: only days
        days_only_diff = timedelta(days=3)
        assert self.state_manager._format_time_ago(days_only_diff) == "3d ago"

        # Test edge case: only hours
        hours_only_diff = timedelta(hours=4)
        assert self.state_manager._format_time_ago(hours_only_diff) == "4h ago"

        print("All add_to_chunk_dt tests passed!")
