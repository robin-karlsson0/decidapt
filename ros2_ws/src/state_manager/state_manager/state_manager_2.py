from datetime import datetime

import rclpy
from exodapt_robot_pt import (appended_state_chunks_pt,
                              dynamic_state_suffix_pt,
                              state_representation_2_pt,
                              static_state_prefix_pt)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from state_chunk_sequence import StateChunk, StateChunkSequence
from std_msgs.msg import String
from std_srvs.srv import Trigger
from transformers import AutoTokenizer

RUNNING_ACTIONS_DEFAULT_STR = 'None'


class StateManager2(Node):

    def __init__(self, **kwargs):
        super().__init__('state_manager_2', **kwargs)

        # Parameters
        string_array_param = Parameter.Type.STRING_ARRAY
        self.declare_parameter('state_topic_name', '/state')
        self.declare_parameter('llm_model_name',
                               'Qwen/Qwen3-30B-A3B-Instruct-2507')
        self.declare_parameter('state_max_tokens', 70000)
        self.declare_parameter('state_seq_clear_ratio', 0.5)
        self.declare_parameter('event_topics', string_array_param)
        self.declare_parameter('continuous_topics', string_array_param)
        self.declare_parameter('thought_topics', string_array_param)
        self.declare_parameter('action_running_topic', '/action_running')
        self.declare_parameter('long_term_memory_file_pth', '')
        self.declare_parameter('state_file_pth', '')
        self.declare_parameter('enable_sweep_service', True)

        self.state_topic_name = self.get_parameter('state_topic_name').value
        self.llm_model_name = self.get_parameter('llm_model_name').value
        self.state_max_tokens = int(
            self.get_parameter('state_max_tokens').value)
        self.state_seq_clear_ratio = float(
            self.get_parameter('state_seq_clear_ratio').value)
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
        self.enable_sweep_service = self.get_parameter(
            'enable_sweep_service').value

        self.get_logger().info(
            'StateManager initializing\n'
            'Parameters:\n'
            f'  state_topic_name: {self.state_topic_name}\n'
            f'  llm_model_name: {self.llm_model_name}\n'
            f'  state_max_tokens: {self.state_max_tokens}\n'
            f'  state_seq_clear_ratio: {self.state_seq_clear_ratio}\n'
            f'  event_topics: {self.event_topics}\n'
            f'  continuous_topics: {self.continuous_topics}\n'
            f'  thought_topics: {self.thought_topics}\n'
            f'  action_running_topic: {self.action_running_topic}\n'
            f'  long_term_memory_file_pth: {self.ltm_file_pth}\n'
            f'  state_file_pth: {self.state_file_pth}\n'
            f'  enable_sweep_service: {self.enable_sweep_service}')

        # LLM tokenizer for measuring state length
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        if len(self.continuous_topics) == 0 \
                and len(self.event_topics) == 0 \
                and len(self.thought_topics) == 0:
            self.get_logger().error(
                'No topics specified. Use continuous_topics and/or event_topics parameters.\n',  # noqa
                'Ex: continuous_topics:=\"[\'/vision\', \'/sensors\']\" event_topics:=\"[\'/asr\', \'/commands\', thought_topics:=\"\'/agency\'\"]\"'  # noqa
            )
            raise IOError()

        # Retrieved from long-term memory
        self.state_retr_mem = ''
        self.state_retr_mem_len = 0

        # Queue stores (state_chunk, timestamp, token_length) tuples
        self.state_seq = StateChunkSequence(self.state_max_tokens)

        # Cached state chunks string to avoid regenerating on every get_state()
        self._cached_state_chunks_str = ''

        # Initialize cached token counts for incremental tracking
        self.state_prefix = static_state_prefix_pt('', '', '', '', '', '', '')
        self.state_chunks = appended_state_chunks_pt('')
        self.state_suffix = dynamic_state_suffix_pt('', '', '', '', '')

        self.state_prefix_tokens = self.get_token_len(self.state_prefix)
        self.state_chunks_tokens = self.get_token_len(self.state_chunks)
        self.state_suffix_tokens = self.get_token_len(self.state_suffix)

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
        self.subscribers = self._create_subscribers()

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
            f'Initial state length: {self.get_token_len(initial_state)} tokens'
        )

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

        # Create state sweep service if enabled
        if self.enable_sweep_service:
            self.sweep_service = self.create_service(
                Trigger, 'sweep_state', self._sweep_state_callback)
            self.get_logger().info('Sweep state service created')
        else:
            self.get_logger().info('Sweep state service disabled')

    ####################
    #  Initialization
    ####################

    def _create_subscribers(self) -> list:
        """Create subscribers for state chunk topics."""
        subscribers = []

        for topic in self.event_topics:
            self.get_logger().info(f'Subscribing to event topic: {topic}')
            subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         event_msg_callback(msg, topic),
                                         10))

        # Continuous topics
        for topic in self.continuous_topics:
            self.get_logger().info(f'Subscribing to continuous topic: {topic}')
            subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         continuous_msg_callback(msg, topic),
                                         10))

        # Thought topics
        for topic in self.thought_topics:
            self.get_logger().info(f'Subscribing to thought topic: {topic}')
            subscribers.append(
                self.create_subscription(String,
                                         topic,
                                         lambda msg, topic=topic: self.
                                         thought_msg_callback(msg, topic),
                                         10))

        return subscribers

    ######################################################
    #  State sequence processing and publishing methods
    ######################################################

    def event_msg_callback(self, msg, topic):
        """Handle event topic messages."""
        self._process_message(msg, topic)

    def continuous_msg_callback(self, msg, topic):
        """Handle continuous topic messages."""
        self._process_message(msg, topic)

    def thought_msg_callback(self, msg, topic):
        """Handle thought topic messages."""
        self._process_message(msg, topic)

    def _process_message(
        self,
        msg: String,
        topic: str,
    ):
        """Common message processing logic for all state chunk types.

        Args:
            msg: ROS message
            topic: Topic name

        Returns:
            int: Number of chunks popped from the queue
        """
        # Get timestamp
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        ts = msg_ts.sec + msg_ts.nanosec * 1e-9

        # Create state chunk
        new_chunk = self.format_state_chunk(topic, msg_ts_str, msg.data)
        new_num_tokens = self.get_token_len(new_chunk)

        # TODO: Add lock
        # Check if state chunk sequence token length will overflow
        # Use cached token count instead of calling get_state_len()
        total_with_new = self._cached_total_state_tokens + new_num_tokens
        if total_with_new >= self.state_max_tokens:
            self.state_seq.clear(self.state_seq_clear_ratio)
            # Regenerate cached string from remaining chunks
            state_chunks = [
                state_chunk.chunk for state_chunk in self.state_seq
            ]
            self._cached_state_chunks_str = '\n'.join(state_chunks)
            # Recalculate state chunks token count after clearing
            self._state_chunks_tokens = self.state_seq.get_token_len()
            self._cached_total_state_tokens = (self._static_template_tokens +
                                               self._state_chunks_tokens +
                                               self._running_actions_tokens)

        self.state_seq.append(StateChunk(new_chunk, new_num_tokens, ts))

        # Append new chunk to cached string
        if self._cached_state_chunks_str:
            self._cached_state_chunks_str += '\n' + new_chunk
        else:
            self._cached_state_chunks_str = new_chunk

        # Update token counts incrementally
        self._state_chunks_tokens += new_num_tokens
        self._cached_total_state_tokens += new_num_tokens

        # Write to long-term memory
        if self.long_term_memory_file_pth:
            with open(self.long_term_memory_file_pth, 'a') as f:
                f.write(new_chunk)

        # Publish updated state
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

    def get_state(self) -> str:
        """Return current state as a string with appended state chunks order."""

        # Use cached state chunk sequence string
        state_chunks = self._cached_state_chunks_str
        self.state_chunks = appended_state_chunks_pt(state_chunks)

        # TODO Components
        dynamic_situation_assessment = ''
        dynamic_current_context_summary = ''
        dynamic_retrieved_mem = ''
        dynamic_running_actions = ''
        dynamic_robot_state_info = ''

        self.state_suffix = dynamic_state_suffix_pt(
            dynamic_situation_assessment,
            dynamic_current_context_summary,
            dynamic_retrieved_mem,
            dynamic_running_actions,
            dynamic_robot_state_info,
        )

        state = self.state_prefix + self.state_chunks + self.state_suffix

        return state

    ##############################
    #  Supporting functionality
    ##############################

    def _action_running_sub_callback(self, msg):
        """Stores latest running actions msg and publish updated state."""
        running_actions = msg.data
        # Default string if message no running actions
        if len(running_actions) > 0:
            self.running_actions = msg.data
        else:
            self.running_actions = RUNNING_ACTIONS_DEFAULT_STR

        # Update running actions token count
        old_running_actions_tokens = self._running_actions_tokens
        self._running_actions_tokens = self.get_token_len(self.running_actions)
        self._cached_total_state_tokens += (self._running_actions_tokens -
                                            old_running_actions_tokens)

        self._publish_state()

    def _sweep_state_callback(self, request, response):
        """Sweep (clear) all state chunks from state sequence."""
        # Count chunks before clearing for logging
        total_count = self.state_seq.get_num_state_chunks()

        # Clear state sequence (100% = clear all)
        self.state_seq.clear(1.0)

        # Clear cached state chunks string
        self._cached_state_chunks_str = ''

        # Reset state chunks token count
        self._state_chunks_tokens = 0
        self._cached_total_state_tokens = (self._static_template_tokens +
                                           self._running_actions_tokens)

        # Publish updated (empty) state
        self._publish_state()

        # Prepare response
        response.success = True
        response.message = (
            f'State swept successfully. Cleared {total_count} chunks')

        # Log the operation
        self.get_logger().info(f'Sweep state completed: {response.message}')

        return response

    #####################
    #  Utility methods
    #####################

    @staticmethod
    def format_state_chunk(topic: str, ts_str: str, data: str) -> str:
        """Format a state chunk as a string.

        Args:
            topic: Name of ROS 2 topic publishing message
            ts_str: Timestamp of message formatted as
                'year-month-day hour:minute:second'
            data: Message content
        """
        return f'\ntopic: {topic}\nts: {ts_str}\ndata: {data}\n---'

    def __len__(self):
        return self.get_state_len()

    def get_state_len(self):
        """Return the token sequence length of the current state."""
        # Use cached token count instead of regenerating state
        return self._cached_total_state_tokens

    def get_token_len(self, s: str) -> int:
        """Return the number of tokens representing a string."""
        return len(self.tokenizer(s)['input_ids'])


def main(args=None):
    rclpy.init(args=args)

    state_manager_2 = StateManager2()

    rclpy.spin(state_manager_2)

    state_manager_2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
