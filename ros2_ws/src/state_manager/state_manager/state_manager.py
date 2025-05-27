import time
from collections import deque
from datetime import datetime

import rclpy
from exodapt_robot_pt import state_description_pt
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from transformers import AutoTokenizer


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

    How to run:
        ros2 run decidapt state_manager --ros-args \
            -p event_topics:="['/asr', '/user_commands', '/navigation_goals']" \
            -p continuous_topics:="['/vision', '/sensors', '/pose']" \
            -p event_queue_max_tokens:=8000 \
            -p continuous_queue_max_tokens:=2000

    TODO:
        - LLM based state chunk filter functionality.
    """

    def __init__(self):
        super().__init__('state_manager')

        # Parameters
        string_array_param = Parameter.Type.STRING_ARRAY
        self.declare_parameter('llm_model_name', 'Qwen/Qwen3-32B')
        self.declare_parameter('event_queue_max_tokens', 8000)
        self.declare_parameter('continuous_queue_max_tokens', 2000)
        self.declare_parameter('event_topics', string_array_param)
        self.declare_parameter('continuous_topics', string_array_param)

        self.event_queue_max_tokens = int(
            self.get_parameter('event_queue_max_tokens').value)
        self.continuous_queue_max_tokens = int(
            self.get_parameter('continuous_queue_max_tokens').value)

        # LLM tokenizer for measuring state length
        llm_model_name = self.get_parameter('llm_model_name').value
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Topics to store to memory set by input parameters
        # Ex: string_topics_to_mem:="['/topic1', '/topic2', '/topic3']"
        self.event_topics = self.get_parameter('event_topics').value
        self.continuous_topics = self.get_parameter('continuous_topics').value

        if len(self.continuous_topics) == 0 and len(self.event_topics) == 0:
            self.get_logger().error(
                'No topics specified. Use continuous_topics and/or event_topics parameters.\n',  # noqa
                'Ex: continuous_topics:=\"[\'/vision\', \'/sensors\']\" event_topics:=\"[\'/asr\', \'/commands\']\"'  # noqa
            )
            raise Exception()

        self.state_header = self.create_state_header()
        self.state_header_len = self.token_len(self.state_header)

        # Retrieved from long-term memory
        self.state_retr_mem = ''
        self.state_retr_mem_len = 0

        # Dual queue system - now storing (state_chunk, timestamp, token_length) tuples
        self.event_queue = deque()
        self.continuous_queue = deque()

        # TODO: Long-term memory
        # self.long_term_memory_file = '/tmp/robot_memory.txt'
        # with open(self.long_term_memory_file, 'w') as f:
        #     f.write('')

        # Create subscribers for both queue types
        self.subscribers = []
        self._create_subscribers()

        self.state_pub = self.create_publisher(String, '/state', 10)

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

    def event_msg_callback(self, msg, topic):
        """Handle event topic messages (token-limited FIFO queue)."""
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = msg_ts.sec + msg_ts.nanosec * 1e-9  # Convert to float for sorting

        state_chunk_str = self.format_state_chunk(topic, msg_ts_str, msg.data)
        token_len = self.token_len(state_chunk_str)

        # Add to event queue with timestamp and token tracking
        self.event_queue.append((state_chunk_str, timestamp, token_len))

        # Trim event queue if it exceeds token limit
        popped_chunks = self._trim_event_queue()

        # Write to long-term memory
        # with open('/tmp/robot_memory.txt', 'a') as f:
        #     f.write(self.format_state_chunk(topic, msg_ts_str, msg.data))

        self._publish_state()
        self.get_logger().info(
            f'Added event state chunk from {topic}. Popped {popped_chunks} chunks.'  # noqa
        )

    def continuous_msg_callback(self, msg, topic):
        """Handle continuous topic messages (token-limited FIFO queue)."""
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = msg_ts.sec + msg_ts.nanosec * 1e-9  # Convert to float for sorting

        state_chunk_str = self.format_state_chunk(topic, msg_ts_str, msg.data)
        token_len = self.token_len(state_chunk_str)

        # Add to continuous queue with timestamp and token tracking
        self.continuous_queue.append((state_chunk_str, timestamp, token_len))

        # Trim continuous queue if it exceeds token limit
        popped_chunks = self._trim_continuous_queue()

        # Write to long-term memory
        # with open('/tmp/robot_memory.txt', 'a') as f:
        #     f.write(self.format_state_chunk(topic, msg_ts_str, msg.data))

        self._publish_state()
        self.get_logger().info(
            f'Added continuous state chunk from {topic}. Popped {popped_chunks} chunks.'  # noqa
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
        while total_tokens > self.continuous_queue_max_tokens and self.continuous_queue:
            removed_item = self.continuous_queue.popleft()
            total_tokens -= removed_item[2]
            popped_chunks += 1
        return popped_chunks

    def _publish_state(self):
        """Publish the current state."""
        msg = String()
        msg.data = self.get_state()
        self.state_pub.publish(msg)

    def create_state_header(self):

        # TODO Current goal state read from topic?
        goal_state = 'Current task goal: ' + \
            'Follow instructions from the user to help users with task. ' + \
            'Provide information and assistance as needed.\n' + \
            '\n'

        state_descr = state_description_pt()

        return goal_state + state_descr

    def trim_queue(self) -> int:
        """Trim the queue if it exceeds the maximum token length.

        Returns the number of poppoed chunks.
        """
        popped_chunks = 0
        while sum(self.queue_len) > self.max_token_len:
            self.queue_len.popleft()
            self.queue.popleft()
            popped_chunks += 1

        return popped_chunks

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

    def get_state(self) -> str:
        """Return the current state as a string with chunks ordered by timestamp."""
        state = ''
        # Header
        state += '<state_header>\n'
        state += self.state_header
        state += '</state_header>\n\n'
        # Retrieved memory
        state += '<retrieved_memory>\n'
        state += self.state_retr_mem
        state += '</retrieved_memory>\n\n'
        # Visual information
        state += '<visual_information>'
        state += self.get_visual_information()
        state += '</visual_information>\n\n'

        # Merge and sort chunks from both queues by timestamp
        all_chunks = []

        # Add event queue chunks
        for chunk_str, timestamp, _ in self.event_queue:
            all_chunks.append((chunk_str, timestamp))

        # Add continuous queue chunks
        for chunk_str, timestamp, _ in self.continuous_queue:
            all_chunks.append((chunk_str, timestamp))

        # Sort by timestamp (oldest first)
        all_chunks.sort(key=lambda x: x[1])

        # Add sorted chunks to state
        state += '<state_chunks>'
        for chunk_str, _ in all_chunks:
            state += chunk_str
        state += '\n</state_chunks>'

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
