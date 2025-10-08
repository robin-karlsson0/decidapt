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
    """ROS2 node managing robot state with an appended state chunk sequence.

    This node maintains a dynamic state representation by aggregating
    messages from subscribed topics into a token-limited sequence of state
    chunks. The appended state is structured as
            [static prefix] + [appended chunks] + [dynamic suffix]
        and is published to a configurable state topic.

    Usage:
        Configure via ROS2 parameters and launch. Subscribe topics to
        event_topics, continuous_topics, or thought_topics to build
        state chunks. Access current state via the published state
        topic or get_state() method.

        Example configuration:
            state_max_tokens: 70000
            state_seq_clear_ratio: 0.5  # Clear 50% when overflow
            event_topics: ['/asr', '/commands']
            continuous_topics: ['/vision', '/sensors']
            thought_topics: ['/agency']
            action_running_topic: '/action_running'
            robot_state_info_topic: '/robot_state_info'

    State Chunk Processing Flow:
        1. Subscribed topic publishes a message
        2. Callback processes message (event/continuous/thought)
        3. Message formatted into StateChunk with timestamp and token count
        4. If total tokens would exceed max, clear oldest chunks by ratio
        5. New chunk appended to sequence and cached string updated
        6. Updated state published to state topic

    Core Variables:
        state_prefix: Static prefix with robot description, personality
        state_seq: StateChunkSequence managing the chunk queue
        state_suffix: Dynamic suffix with current robot system information
        _cached_state_chunks_str: Performance-optimized chunk string cache
        _cached_running_actions: Current running actions (from subscription)
        _cached_robot_state_info: Robot state info (from subscription)

    Token Counting:
        - state_prefix_num_tokens: Static, cached at initialization
        - state_chunks_num_tokens: Sum from StateChunkSequence
        - state_suffix_num_tokens: Recomputed on each access
        Total via get_state_token_len() or len(state_manager)

        NOTE: state_chunks_num_tokens is approximate and does not contain
            state template tokens!

    Key Methods:
        - get_state() -> str: Get complete current state
        - get_state_token_len() -> int: Get total token count
        - sweep_state service: Clear all state chunks via ROS2 service

    Properties:
        - state_chunks: Formatted state chunks with template wrapper
        - state_suffix: Dynamic suffix with running actions & robot info
        - state_chunks_num_tokens: Token count of chunk sequence (without state
            template tokens)
        - state_suffix_num_tokens: Token count of dynamic suffix

    Thread Safety:
        All state modifications protected by self.lock. Safe for concurrent
        topic callbacks and state access from multiple threads.

    Performance Notes:
        - Token counts are approximations (~13 token template overhead)
        - Acceptable error: <0.02% at typical state sizes (70k tokens)
        - Cached strings updated incrementally for O(1) append performance
    """

    def __init__(self, **kwargs):
        """Initialize StateManager2 node with parameters and subscriptions.

        Sets up tokenizer, state sequences, subscriptions to configured
        topics, and optional features (long-term memory file, state file,
        sweep service). Publishes initial state with prefix and suffix.

        Raises:
            IOError: If no topics are specified in parameters.
        """
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
        """Return formatted state chunks with template wrapper applied.
        
        Returns:
            str: Template-wrapped concatenation of cached state chunks.
        """
        return appended_state_chunks_pt(self._cached_state_chunks_str)

    @property
    def state_chunks_num_tokens(self) -> int:
        """Return token count from state chunk sequence.

        Returns sum of individual chunk token counts, excluding template
        wrapper tokens (~13 tokens). This approximation is acceptable for
        soft limits with negligible error (<0.02% at 70k token sizes).

        Returns:
            int: Sum of all chunk token counts in sequence.
        """
        return len(self.state_seq)

    @property
    def state_suffix(self):
        """Generate dynamic state suffix from cached variables.

        Uses cached running actions and robot state info, which may be
        updated concurrently by callbacks. Brief inconsistency is acceptable.

        Returns:
            str: Template formatted dynamic suffix with current system state.
        """
        return dynamic_state_suffix_pt(
            self._cached_running_actions,
            self._cached_robot_state_info,
        )

    @property
    def state_suffix_num_tokens(self):
        """Count tokens in current dynamic state suffix.

        Recomputes on each access since suffix is small and changes
        frequently. More efficient than caching for this use case.

        Returns:
            int: Number of tokens in current state suffix.
        """
        return self.get_token_len(self.state_suffix)

    ####################
    #  Initialization
    ####################

    def _create_subscribers(self) -> list:
        """Create subscribers for all configured topic types.

        Returns:
            list: ROS2 subscription objects for event, continuous, and
                thought topics.
        """
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
        """Process incoming message into state chunk and publish update.

        Creates timestamped state chunk, manages token overflow by clearing
        old chunks if needed, appends to sequence, writes to long-term
        memory (if configured), and publishes updated state.

        Thread-safe: Uses lock to protect shared state.

        Args:
            msg: ROS String message containing data
            topic: Topic name from which message was received
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
        """Publish current state and optionally write to state file."""
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
        """Get complete state representation.

        Concatenates static prefix, formatted state chunks (via template),
        and dynamic suffix. Reads cached variables that may update
        concurrently; brief inconsistency is acceptable.

        Returns:
            str: Full state string (prefix + chunks + suffix).
        """
        state = self.state_prefix + self.state_chunks + self.state_suffix
        return state

    ##############################
    #  Supporting functionality
    ##############################

    def _action_running_sub_callback(self, msg: String):
        """Update running actions and republish state.

        Thread-safe: Updates under lock protection.

        Args:
            msg: String message containing current running actions.
        """
        running_actions = msg.data.strip()

        with self.lock:
            if running_actions:
                self._cached_running_actions = running_actions
            else:
                self._cached_running_actions = DEFAULT_RUNNING_ACTIONS

        self._publish_state()

    def _robot_state_info_callback(self, msg: String):
        """Update robot state info and republish state.

        Thread-safe: Updates under lock protection.

        Args:
            msg: String message containing robot state information.
        """
        with self.lock:
            self._cached_robot_state_info = msg.data

        self._publish_state()

    def _sweep_state_callback(self, request, response):
        """Service callback to clear all state chunks.

        Thread-safe: Uses lock to protect state modifications.

        Args:
            request: Trigger service request (unused).
            response: Trigger service response to populate.

        Returns:
            Trigger.Response: Success status and message with chunk count.
        """
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
        """Format message into standardized state chunk string.

        Args:
            topic: ROS2 topic name
            ts_str: Timestamp as 'YYYY-MM-DD HH:MM:SS'
            data: Message content

        Returns:
            str: Formatted chunk with topic, timestamp, data, and delimiter.
        """
        return f'\ntopic: {topic}\nts: {ts_str}\ndata: {data}\n---'

    def __len__(self):
        """Return total token count of current state.

        Usage: len(state_manager) returns token count.

        Returns:
            int: Sum of prefix, chunks, and suffix token counts.
        """
        return self.get_state_token_len()

    def get_state_token_len(self):
        """Calculate total token count of current state.

        Returns:
            int: Sum of state_prefix_num_tokens, state_chunks_num_tokens,
                and state_suffix_num_tokens.
        """
        num_tokens = (self.state_prefix_num_tokens +
                      self.state_chunks_num_tokens +
                      self.state_suffix_num_tokens)

        return num_tokens

    def get_token_len(self, s: str) -> int:
        """Count tokens in string using configured tokenizer.

        Args:
            s: String to tokenize

        Returns:
            int: Number of tokens in string.
        """
        return len(self.tokenizer(s)['input_ids'])


def main(args=None):
    rclpy.init(args=args)

    state_manager_2 = StateManager2()

    rclpy.spin(state_manager_2)

    state_manager_2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
