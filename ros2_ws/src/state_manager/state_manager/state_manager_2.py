import threading
from datetime import datetime

import rclpy
from exodapt_robot_pt import (appended_state_chunks_pt,
                              dynamic_state_suffix_pt, static_state_prefix_pt)
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger
from transformers import AutoTokenizer

from .state_chunk_sequence import StateChunk, StateChunkSequence

DEFAULT_RUNNING_ACTIONS = 'None'
DEFAULT_ROBOT_STATE_INFO = 'Robot state information not yet received'

# To be replaced with actual implemnetation
ROBOT_DESCRIPTION = 'Robot description'
STATE_REPRESENTATION_STRUCTURE = 'State representation structure'
PERSONALITY_PROFILE = 'Personality profile'
AGENCY_DESCRIPTION = 'Agency description'
TOPIC_AND_ACTION_DESCRIPTION = 'Topic and action description'


class StateManager2(Node):
    """

    Maintained core variables:
        
        # State strings
        state_prefix - Static prefix containing robot description, personality, etc.
        state_chunks (property) - Formatted state chunks using template
        state_suffix (property) - Dynamic suffix with running actions and robot info

        # Token counts
        state_prefix_num_tokens: Token count of the static prefix (cached)
        state_chunks_num_tokens (property): Token count from state_seq
        state_suffix_num_tokens (property): Token count of the dynamic suffix (recomputed each access)

        state_seq: StateChunkSequence object managing the queue of state chunks
            _cached_state_chunks_str: Cached string representation of all state chunks (for performance)
        _cached_running_actions: Cached current running actions string
        _cached_robot_state_info: Cached robot state information string

    Core methods
        get_state()
        get_state_token_len() or len(state_manager_2)
        _publish_state()
        

    State chunk processing flow:
        1. A subscribed topic publishes a msg

        2. A callback function processes the msg
            event_msg_callback()
            _process_message()

        3. The msg is formatted into a StateChunk and appended to sequence
            - Sequence is cleared if its length will be over max token length
        
        4. The updated state is published
            get_state()
            state_pub.publish()

    State Structure:

        [Static prefix]
        - Token count: Stored in StateManager2
        >>> self.state_prefix
        >>> self.state_prefix_num_tokens

        [Appended state chunk sequence]
        - Token count: Managed by StateChunkSequence
        >>> self.state_chunks
                _cached_state_chunks_str
                template(_cached_state_chunks_str)
        >>> self.state_chunks_num_tokens

        [Dynamic suffix]
        - Token count: Recomputed by StateManager2
        >>> self.state_suffix
        >>> self.state_suffix_num_tokens
    
    NOTE: The considered state token sequence lengths are not considered exact!
        The provided values are close enough for determining state sequence
        clearing while providing high performance.
    """  # noqa

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
        self.declare_parameter('robot_state_info_topic', '/robot_state_info')
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
        self.robot_state_info_topic = self.get_parameter(
            'robot_state_info_topic').value
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
            f'  robot_state_info_topic: {self.robot_state_info_topic}\n'
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

        ##########################
        #  State Representation
        ##########################

        # Queue stores (state_chunk, timestamp, token_length) tuples
        self.state_seq = StateChunkSequence(self.state_max_tokens)

        # State prefix and token count are static values
        self.state_prefix = static_state_prefix_pt(
            ROBOT_DESCRIPTION,
            STATE_REPRESENTATION_STRUCTURE,
            PERSONALITY_PROFILE,
            AGENCY_DESCRIPTION,
            TOPIC_AND_ACTION_DESCRIPTION,
        )
        self.state_prefix_num_tokens = self.get_token_len(self.state_prefix)

        # Appended state chunks managed by state chunk sequence and templated
        # by StateManager2
        self._cached_state_chunks_str = ''

        # Cached results of state suffix components
        self._cached_running_actions = ''
        self._cached_robot_state_info = ''

        self.lock = threading.Lock()

        ######################
        #  Long-Term Memory
        ######################

        # Long-term memory file initialization
        if self.ltm_file_pth:
            self.long_term_memory_file_pth = self.ltm_file_pth
            self.get_logger().info(
                f'Writing Long-term memory to file: {self.long_term_memory_file_pth}'  # noqa
            )
            try:
                with open(self.long_term_memory_file_pth, 'w') as f:
                    f.write('')
            except IOError as e:
                self.get_logger().error(
                    f'Failed to initialize long-term memory file: {e}')
                self.long_term_memory_file_pth = None
        else:
            self.long_term_memory_file_pth = None

        ################################
        #  Subscribers and Publishers
        ################################

        # Create subscribers for both queue types
        self.subscribers = self._create_subscribers()

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

        # State suffix subscriptions
        self.running_action_sub = self.create_subscription(
            String,
            self.action_running_topic,
            self._action_running_sub_callback,
            10,
        )
        self._cached_running_actions = DEFAULT_RUNNING_ACTIONS

        self.robot_state_info_sub = self.create_subscription(
            String,
            self.robot_state_info_topic,
            self._robot_state_info_callback,
            10,
        )
        self._cached_robot_state_info = DEFAULT_ROBOT_STATE_INFO

        #######################
        #  Optional Features
        #######################

        # Optional "state-to-file" functionality for easily observing state
        if self.state_file_pth:
            self.get_logger().info(
                f'Writing state to file: {self.state_file_pth}')
            try:
                state = self.get_state()
                with open(self.state_file_pth, 'w') as f:
                    f.write(state)
            except IOError as e:
                self.get_logger().error(
                    f'Failed to initialize state file: {e}')
                self.state_file_pth = None
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

    @property
    def state_chunks(self) -> str:
        """Returns a formatted state chunks string."""
        return appended_state_chunks_pt(self._cached_state_chunks_str)

    @property
    def state_chunks_num_tokens(self) -> int:
        """Returns the token count from the state chunk sequence.

        Note: This returns the sum of individual chunk token counts, which does
        not include the template wrapper tokens (~13 tokens). This approximation
        is acceptable since the token count is used for soft limits and the
        error is negligible (<0.02% at typical state sizes).
        """
        return len(self.state_seq)

    @property
    def state_suffix(self):
        """Generate a state suffix using cached and real-time information.

        Note: Cached variables may be updated concurrently by callbacks.
        Brief inconsistency is acceptable for this use case.
        """
        return dynamic_state_suffix_pt(
            self._cached_running_actions,
            self._cached_robot_state_info,
        )

    @property
    def state_suffix_num_tokens(self):
        """Counts number of tokens in the current state suffix.

        NOTE: The state suffix is presumed very small and dynamic. Recounting
            the number of tokens every time is sensible.
        """
        return self.get_token_len(self.state_suffix)

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
        """Handle event topic messages.

        Currently delegates to _process_message. This routing layer is
        maintained for future type-specific processing capabilities.

        Args:
            msg: ROS String message
            topic: Topic name from which message was received
        """
        self._process_message(msg, topic)

    def continuous_msg_callback(self, msg, topic):
        """Handle continuous topic messages.

        Currently delegates to _process_message. This routing layer is
        maintained for future type-specific processing capabilities.

        Args:
            msg: ROS String message
            topic: Topic name from which message was received
        """
        self._process_message(msg, topic)

    def thought_msg_callback(self, msg, topic):
        """Handle thought topic messages.

        Currently delegates to _process_message. This routing layer is
        maintained for future type-specific processing capabilities.

        Args:
            msg: ROS String message
            topic: Topic name from which message was received
        """
        self._process_message(msg, topic)

    def _process_message(
        self,
        msg: String,
        topic: str,
    ):
        """Common message processing logic for all state chunk types.

        Thread-safe: Uses lock to protect shared state during modifications.

        Args:
            msg: ROS message
            topic: Topic name
        """
        # Get timestamp
        msg_ts = self.get_clock().now().to_msg()
        msg_ts_str = datetime.fromtimestamp(
            msg_ts.sec).strftime('%Y-%m-%d %H:%M:%S')
        ts = msg_ts.sec + msg_ts.nanosec * 1e-9

        # Create state chunk
        new_chunk = self.format_state_chunk(topic, msg_ts_str, msg.data)
        new_num_tokens = self.get_token_len(new_chunk)

        # Critical section: check overflow and append
        with self.lock:
            total_with_new = self.get_state_token_len() + new_num_tokens
            if total_with_new >= self.state_max_tokens:
                self.state_seq.clear(self.state_seq_clear_ratio)
                # Regenerate cached string from remaining chunks
                state_chunks = [
                    state_chunk.chunk for state_chunk in self.state_seq
                ]
                self._cached_state_chunks_str = '\n'.join(state_chunks)

            # Append new chunk
            self.state_seq.append(StateChunk(new_chunk, new_num_tokens, ts))

            # Update cached string
            if self._cached_state_chunks_str:
                self._cached_state_chunks_str += '\n' + new_chunk
            else:
                self._cached_state_chunks_str = new_chunk

        # Write to long-term memory
        if self.long_term_memory_file_pth:
            try:
                with open(self.long_term_memory_file_pth, 'a') as f:
                    f.write(new_chunk + '\n')
            except IOError as e:
                self.get_logger().error(
                    f'Failed to write to long-term memory file: {e}')

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
            try:
                with open(self.state_file_pth, 'w') as f:
                    f.write(state)
            except IOError as e:
                self.get_logger().error(f'Failed to write state to file: {e}')

    def get_state(self) -> str:
        """Return current state as a string with appended state chunks.

        Note: Reads cached variables that may be updated concurrently.
        Brief inconsistency between components is acceptable.

        Returns:
            str: Complete state representation including prefix, chunks,
            and suffix.
        """
        state = self.state_prefix + self.state_chunks + self.state_suffix
        return state

    ##############################
    #  Supporting functionality
    ##############################

    def _action_running_sub_callback(self, msg: String):
        """Stores the latest running actions msg and publish updated state.

        Thread-safe: Updates cached running actions under lock protection.
        """
        running_actions = msg.data.strip()

        with self.lock:
            if running_actions:
                self._cached_running_actions = running_actions
            else:
                self._cached_running_actions = DEFAULT_RUNNING_ACTIONS

        self._publish_state()

    def _robot_state_info_callback(self, msg: String):
        """Stores the latest robot state info msg.

        Thread-safe: Updates cached robot state info under lock protection.
        """
        with self.lock:
            self._cached_robot_state_info = msg.data

        self._publish_state()

    def _sweep_state_callback(self, request, response):
        """Sweep (clear) all state chunks from state sequence."""
        # Thread-safe: Use lock to protect state modifications
        with self.lock:
            # Count chunks before clearing for logging
            total_count = self.state_seq.get_num_state_chunks()

            # Clear state sequence (0% = clear all)
            self.state_seq.clear(0.0)

            # Clear cached state chunks string
            self._cached_state_chunks_str = ''

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
        """Return the total token count of the current state.

        Returns:
            int: Sum of prefix, chunks, and suffix token counts.
        """
        return self.get_state_token_len()

    def get_state_token_len(self):
        """Return the token sequence length of the current state."""
        num_tokens = (self.state_prefix_num_tokens +
                      self.state_chunks_num_tokens +
                      self.state_suffix_num_tokens)

        return num_tokens

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
