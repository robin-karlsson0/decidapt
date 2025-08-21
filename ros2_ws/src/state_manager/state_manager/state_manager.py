import re
import time
from collections import deque
from datetime import datetime

import rclpy
from exodapt_robot_pt import state_representation_pt
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from transformers import AutoTokenizer

RUNNING_ACTIONS_DEFAULT_STR = 'None'


class StateManager(Node):
    """ROS2 node that listens to topics and maintains a queue of state chunks.

    StateManager is a ROS2 node that listens to a list of topics and maintains
    a FIFO queue of state chunks. The StateManager publish the latest state
    when a new state chunk is added.

    State chunks are textual representations of any percepts or internal
    states.

    Structure of state chunks:
    ---
    topic: Name of ROS 2 topic publishing message
    ts: Timestamp of message formatted as 'year-month-day hour:minute:second'
    data: Message content
    ---

    Structure of state:
    ---
    Header
        Describes the structure of the state representation.
    ---
    Retrieved memory
        Task-relevant information retrieved and summarized from the long-term
        memory.
    ---
    State chunk 0
        The oldest state chunk in the queue.
    ---
    ...
    ---
    State chunk M
        The most recent state chunk.
    ---

    NOTE: A new state must be published for EVERY incoming update, including
        when receiving 'action_running_topic' messages!

    How to run:
        ros2 run decidapt state_manager --ros-args \
            -p event_topics:="['/asr', '/user_commands', '/navigation_goals']" \
            -p continuous_topics:="['/vision', '/sensors', '/pose']" \
            -p thought_topics:="['/agency', '/self_reflection']" \
            -p event_queue_max_tokens:=8000 \
            -p continuous_queue_max_tokens:=2000 \
            -p thought_queue_max_tokens:=2000

    TODO:
        - LLM based state chunk filter functionality.
    """

    def __init__(self, **kwargs):
        super().__init__('state_manager', **kwargs)

        # Parameters
        string_array_param = Parameter.Type.STRING_ARRAY
        self.declare_parameter('state_topic_name', '/state')
        self.declare_parameter('llm_model_name', 'Qwen/Qwen3-32B')
        self.declare_parameter('event_queue_max_tokens', 8000)
        self.declare_parameter('continuous_queue_max_tokens', 2000)
        self.declare_parameter('thought_queue_max_tokens', 2000)
        self.declare_parameter('event_topics', string_array_param)
        self.declare_parameter('continuous_topics', string_array_param)
        self.declare_parameter('thought_topics', string_array_param)
        self.declare_parameter('action_running_topic', '/action_running')
        self.declare_parameter('long_term_memory_file_pth', '')
        self.declare_parameter('state_file_pth', '')

        self.state_topic_name = self.get_parameter('state_topic_name').value
        self.llm_model_name = self.get_parameter('llm_model_name').value
        self.event_queue_max_tokens = int(
            self.get_parameter('event_queue_max_tokens').value)
        self.continuous_queue_max_tokens = int(
            self.get_parameter('continuous_queue_max_tokens').value)
        self.thought_queue_max_tokens = int(
            self.get_parameter('thought_queue_max_tokens').value)
        # Topics to store to memory set by input parameters
        # Ex: string_topics_to_mem:="['/topic1', '/topic2', '/topic3']"
        self.event_topics = self.get_parameter('event_topics').value
        self.continuous_topics = self.get_parameter('continuous_topics').value
        self.thought_topics = self.get_parameter('thought_topics').value
        self.action_running_topic = self.get_parameter(
            'action_running_topic').value
        self.ltm_file_pth = self.get_parameter(
            'long_term_memory_file_pth').value
        self.state_file_pth = self.get_parameter('state_file_pth').value

        self.get_logger().info(
            'StateManager initializing\n'
            'Parameters:\n'
            f'  state_topic_name: {self.state_topic_name}\n'
            f'  llm_model_name: {self.llm_model_name}\n'
            f'  event_queue_max_tokens: {self.event_queue_max_tokens}\n'
            f'  continuous_queue_max_tokens: {self.continuous_queue_max_tokens}\n'  # noqa
            f'  thought_queue_max_tokens: {self.thought_queue_max_tokens}\n'
            f'  event_topics: {self.event_topics}\n'
            f'  continuous_topics: {self.continuous_topics}\n'
            f'  thought_topics: {self.thought_topics}\n'
            f'  action_running_topic: {self.action_running_topic}\n'
            f'  long_term_memory_file_pth: {self.ltm_file_pth}\n'
            f'  state_file_pth: {self.state_file_pth}')

        # LLM tokenizer for measuring state length
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        if len(self.continuous_topics) == 0 \
                and len(self.event_topics) == 0 \
                and len(self.thought_topics) == 0:
            self.get_logger().error(
                'No topics specified. Use continuous_topics and/or event_topics parameters.\n',  # noqa
                'Ex: continuous_topics:=\"[\'/vision\', \'/sensors\']\" event_topics:=\"[\'/asr\', \'/commands\', thought_topics:=\"\'/agency\'\"]\"'  # noqa
            )
            raise Exception()

        # Retrieved from long-term memory
        self.state_retr_mem = ''
        self.state_retr_mem_len = 0

        # Queue stores (state_chunk, timestamp, token_length) tuples
        self.event_queue = deque()
        self.continuous_queue = deque()
        self.thought_queue = deque()

        # Long-term memory file initialization
        if self.ltm_file_pth:
            self.long_term_memory_file_pth = self.ltm_file_pth
            self.get_logger().info(
                f'Writing Long-term memory to file: {self.long_term_memory_file_pth}'  # noqa
            )
            with open(self.long_term_memory_file_pth, 'w') as f:
                f.write('')
        else:
            self.long_term_memory_file_pth = None

        # Create subscribers for both queue types
        self.subscribers = []
        self._create_subscribers()

        # Create running actions subscriber
        self.running_action_sub = self.create_subscription(
            String,
            self.action_running_topic,
            self._action_running_sub_callback,
            10,
        )
        self.running_actions = RUNNING_ACTIONS_DEFAULT_STR

        # Create QoS profile with transient local durability
        qos_profile = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(
            String,
            self.state_topic_name,
            qos_profile,
        )

        # Publish initial state
        initial_state = self.get_state()
        initial_msg = String()
        initial_msg.data = initial_state
        self.state_pub.publish(initial_msg)
        self.get_logger().info('Initial state published.')
        self.get_logger().info(
            f'Initial state length: {self.token_len(initial_state)} tokens')

        # Optional "state-to-file" functionality for easily observing state
        if self.state_file_pth:
            self.get_logger().info(
                f'Writing state to file: {self.state_file_pth}')
            state = self.get_state()
            with open(self.state_file_pth, 'w') as f:
                f.write(state)
        else:
            self.state_file_pth = None
            self.get_logger().info('No state file specified')

    def _create_subscribers(self):
        """Create subscribers for continuous and event-driven topics."""
        # Event topics
        for topic in self.event_topics:
            self.get_logger().info(f'Subscribing to event topic: {topic}')
            self.subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         event_msg_callback(msg, topic),
                                         10))

        # Continuous topics
        for topic in self.continuous_topics:
            self.get_logger().info(f'Subscribing to continuous topic: {topic}')
            self.subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         continuous_msg_callback(msg, topic),
                                         10))

        # Thought topics
        for topic in self.thought_topics:
            self.get_logger().info(f'Subscribing to thought topic: {topic}')
            self.subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         thought_msg_callback(msg, topic),
                                         10))

    def event_msg_callback(self, msg, topic):
        """Handle event topic messages (token-limited FIFO queue)."""
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = msg_ts.sec + msg_ts.nanosec * 1e-9

        state_chunk_str = self.format_state_chunk(topic, msg_ts_str, msg.data)
        token_len = self.token_len(state_chunk_str)

        # Add to event queue with timestamp and token tracking
        self.event_queue.append((state_chunk_str, timestamp, token_len))

        # Trim event queue if it exceeds token limit
        popped_chunks = self._trim_event_queue()

        # Write to long-term memory
        if self.long_term_memory_file_pth:
            with open(self.long_term_memory_file_pth, 'a') as f:
                f.write(self.format_state_chunk(topic, msg_ts_str, msg.data))

        self._publish_state()
        self.get_logger().debug(
            f'Added event state chunk from {topic}. Popped {popped_chunks} chunks.'  # noqa
        )

    def continuous_msg_callback(self, msg, topic):
        """Handle continuous topic messages (token-limited FIFO queue)."""
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = msg_ts.sec + msg_ts.nanosec * 1e-9

        state_chunk_str = self.format_state_chunk(topic, msg_ts_str, msg.data)
        token_len = self.token_len(state_chunk_str)

        # Add to continuous queue with timestamp and token tracking
        self.continuous_queue.append((state_chunk_str, timestamp, token_len))

        # Trim continuous queue if it exceeds token limit
        popped_chunks = self._trim_continuous_queue()

        # Write to long-term memory
        if self.long_term_memory_file_pth:
            with open(self.long_term_memory_file_pth, 'a') as f:
                f.write(self.format_state_chunk(topic, msg_ts_str, msg.data))

        self._publish_state()
        self.get_logger().debug(
            f'Added continuous state chunk from {topic}. Popped {popped_chunks} chunks.'  # noqa
        )

    def thought_msg_callback(self, msg, topic):
        """Handle thought topic messages (token-limited FIFO queue)."""
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = msg_ts.sec + msg_ts.nanosec * 1e-9

        state_chunk_str = self.format_state_chunk(topic, msg_ts_str, msg.data)
        token_len = self.token_len(state_chunk_str)

        # Add to continuous queue with timestamp and token tracking
        self.thought_queue.append((state_chunk_str, timestamp, token_len))

        # Trim continuous queue if it exceeds token limit
        popped_chunks = self._trim_thought_queue()

        # Write to long-term memory
        if self.long_term_memory_file_pth:
            with open(self.long_term_memory_file_pth, 'a') as f:
                f.write(self.format_state_chunk(topic, msg_ts_str, msg.data))

        self._publish_state()
        self.get_logger().debug(
            f'Added thought state chunk from {topic}. Popped {popped_chunks} chunks.'  # noqa
        )

    def _trim_event_queue(self) -> int:
        """Trim the event queue if it exceeds the maximum token length."""
        popped_chunks = 0
        total_tokens = sum(item[2]
                           for item in self.event_queue)  # Sum token lengths
        while total_tokens > self.event_queue_max_tokens and self.event_queue:
            removed_item = self.event_queue.popleft()
            total_tokens -= removed_item[2]
            popped_chunks += 1
        return popped_chunks

    def _trim_continuous_queue(self) -> int:
        """Trim the continuous queue if it exceeds the maximum token length."""
        popped_chunks = 0
        total_tokens = sum(
            item[2] for item in self.continuous_queue)  # Sum token lengths
        while total_tokens > self.continuous_queue_max_tokens and self.continuous_queue:  # noqa
            removed_item = self.continuous_queue.popleft()
            total_tokens -= removed_item[2]
            popped_chunks += 1
        return popped_chunks

    def _trim_thought_queue(self) -> int:
        """Trim the thought queue if it exceeds the maximum token length."""
        popped_chunks = 0
        total_tokens = sum(item[2]
                           for item in self.thought_queue)  # Sum token lengths
        while total_tokens > self.thought_queue_max_tokens and self.thought_queue:  # noqa
            removed_item = self.thought_queue.popleft()
            total_tokens -= removed_item[2]
            popped_chunks += 1
        return popped_chunks

    def _action_running_sub_callback(self, msg):
        """Stores latest running actions msg and publish updated state."""
        running_actions = msg.data
        # Default string if message no running actions
        if len(running_actions) > 0:
            self.running_actions = msg.data
        else:
            self.running_actions = RUNNING_ACTIONS_DEFAULT_STR

        self._publish_state()

    def _publish_state(self):
        """Publish the current state."""
        msg = String()
        state = self.get_state()
        msg.data = state
        self.state_pub.publish(msg)

        # Optionally write state to file
        if self.state_file_pth:
            with open(self.state_file_pth, 'w') as f:
                f.write(state)

    @staticmethod
    def format_state_chunk(topic: str, ts: str, data: str) -> str:
        """Format a state chunk as a string.

        Args:
            topic: Name of ROS 2 topic publishing message
            ts: Timestamp of message formatted as
                'year-month-day hour:minute:second'
            data: Message content
        """
        return f'\ntopic: {topic}\nts: {ts}\ndata: {data}\n---'

    def __len__(self):
        state = self.get_state()
        return self.token_len(state)

    @staticmethod
    def _format_time_ago(diff):
        """Format time difference as human-readable string.

        Format: (1d 2h 3m 4s ago)
        """
        parts = []
        if diff.days > 0:
            parts.append(f"{diff.days}d")

        hours = diff.seconds // 3600
        if hours > 0:
            parts.append(f"{hours}h")

        minutes = (diff.seconds % 3600) // 60
        if minutes > 0:
            parts.append(f"{minutes}m")

        seconds = diff.seconds % 60
        if seconds > 0:
            parts.append(f"{seconds}s")

        return " ".join(parts) + " ago" if parts else "0s ago"

    def add_to_chunk_dt(self, chunk: str) -> str:
        """Returns a state chunk with added (T time ago) after timestamp entry.

        Converts the substring
            ts: 2025-08-21 15:38:49 --> ts: 2025-08-21 15:38:49 (25m 15s ago)
        """
        current_time = datetime.now()

        def replace_ts(match):
            ts_line = match.group(0)
            ts_str = match.group(1)  # Just the timestamp part

            # Parse timestamp and calculate difference
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            diff = current_time - ts
            time_ago = self._format_time_ago(diff)

            return f"{ts_line} ({time_ago})"

        # Capture just the timestamp portion after "ts: "
        pattern = r'^ts: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$'
        return re.sub(pattern, replace_ts, chunk, flags=re.MULTILINE)

    def get_state(self) -> str:
        """Return the state as a string with chunks ordered by timestamp."""
        ##################
        #  State chunks
        ##################
        # Merge and sort chunks from both queues by timestamp
        all_chunks = []

        # Add event queue chunks
        for chunk_str, timestamp, _ in self.event_queue:
            chunk_str_dt = self.add_to_chunk_dt(chunk_str)
            all_chunks.append((chunk_str_dt, timestamp))

        # Add continuous queue chunks
        for chunk_str, timestamp, _ in self.continuous_queue:
            chunk_str_dt = self.add_to_chunk_dt(chunk_str)
            all_chunks.append((chunk_str_dt, timestamp))

        # Add thought queue chunks
        for chunk_str, timestamp, _ in self.thought_queue:
            chunk_str_dt = self.add_to_chunk_dt(chunk_str)
            all_chunks.append((chunk_str_dt, timestamp))

        # Sort by timestamp (oldest first)
        all_chunks.sort(key=lambda x: x[1])

        # Add sorted chunks to state
        state_chunks = []
        for chunk_str, _ in all_chunks:
            state_chunks.append(chunk_str)
        state_chunks = '\n'.join(state_chunks)

        ##########################
        #  State representation
        ##########################
        # TODO Components
        situation_assessment = ''
        current_context_summary = ''
        personality_static = ''
        personality_dynamic = ''
        retrieved_mem = ''

        state = state_representation_pt(
            situation_assessment,
            current_context_summary,
            personality_static,
            personality_dynamic,
            retrieved_mem,
            state_chunks,
            self.running_actions,
        )

        return state

    def get_visual_information(self) -> str:
        """Placeholder for visual information retrieval."""
        # This method wasn't implemented in the original code
        return ''

    @staticmethod
    def str_time() -> str:
        """Return a timestamp in the format '1729514281_1626208'."""
        return str(time.time()).replace('.', '_')

    def token_len(self, input_str: str) -> int:
        """Return the number of tokens representing an input string."""
        return len(self.tokenizer(input_str)['input_ids'])


def main(args=None):
    rclpy.init(args=args)

    state_manager = StateManager()

    rclpy.spin(state_manager)

    state_manager.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
