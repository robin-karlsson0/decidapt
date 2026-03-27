import json
import threading
from datetime import datetime

import rclpy
from exodapt_robot_interfaces.srv import StartReconciliation
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
            asr_service_name: 'start_reconciliation'

    State Chunk Processing Flow:
        1. Subscribed topic publishes a message
        2. Callback processes message (event/continuous/thought)
        3. Message formatted into StateChunk with timestamp and token count
        4. If total tokens would exceed max, _evict() is called:
               a. Clears oldest chunks in state_seq by state_seq_clear_ratio
               b. Bumps state_seq_ver k and captures evicted_static_len
               c. Returns the evicted state string X_eps
               d. _call_start_reconciliation() notifies ASRManager
        5. New chunk appended to sequence and cached string updated
        6. Updated state published to state topic

    Core Variables:
        state_prefix: Static prefix with robot description, personality
        state_seq: StateChunkSequence managing the chunk queue
        state_suffix: Dynamic suffix with current robot system information
        _cached_state_chunks_str: Performance-optimized chunk string cache
        _cached_running_actions: Current running actions (from subscription)
        _cached_robot_state_info: Robot state info (from subscription)
        state_seq_ver: Sequence version counter k, incremented on every eviction
        evicted_static_len: len(pre + chunks) captured at the moment of eviction
            (j_eps in the ASR algorithm)

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
        - _evict() -> str: Evict oldest chunks and notify ASRManager
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
        self.declare_parameter('asr_service_name', 'start_reconciliation')

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
        self.asr_service_name = self.get_parameter('asr_service_name').value

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
            f'  enable_sweep_service: {self.enable_sweep_service}\n'
            f'  asr_service_name: {self.asr_service_name}')

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
        self.state_prefix_len = len(self.state_prefix)
        self.state_prefix_num_tokens = self.get_token_len(self.state_prefix)

        # Appended state chunks managed by state chunk sequence and templated
        # by StateManager2
        self._cached_state_chunks_str = ''

        # Cached results of state suffix components
        self._cached_running_actions = ''
        self._cached_robot_state_info = ''

        self.state_idx = 0
        self.state_seq_ver = 0
        self.evicted_static_len = 0

        self.lock = threading.Lock()

        ####################################
        #  ASR StartReconciliation client
        ####################################

        self._asr_client = self.create_client(
            StartReconciliation, self.asr_service_name)
        self.get_logger().info(
            f'StartReconciliation client created for service: '
            f'{self.asr_service_name}')

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

        self.get_logger().info('StateManager2 initialisation complete.')

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

        # Evict+reconcile outside the lock if the new chunk would overflow.
        # _evict_and_reconcile() acquires the lock internally for the state
        # mutation, then releases it before calling the ASR service (I/O).
        with self.lock:
            total_with_new = self.get_state_token_len() + new_num_tokens
            needs_eviction = total_with_new >= self.state_max_tokens

        if needs_eviction:
            self._evict_and_reconcile()

        # Append new chunk under lock
        with self.lock:
            self.state_seq.append(StateChunk(new_chunk, new_num_tokens, ts))

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
        try:
            self.state_pub.publish(msg)
            self.state_idx += 1

            # Optionally write state to file
            if self.state_file_pth:
                try:
                    with open(self.state_file_pth, 'w') as f:
                        f.write(state)
                except IOError as e:
                    self.get_logger().error(f'Failed to write state to file: {e}')  # noqa
        except Exception as e:
            self.get_logger().error(f'Could not publish state: {e}')

    def get_state(self) -> str:
        """Get complete state representation.

        Returns JSON string with static prefix, formatted state chunks,
        dynamic suffix, and tracking metadata aligned with the ASR protocol
        (j_t, k_t, j_epsilon_t cursors used by ASRManager).

        Returns:
            str: JSON string containing the state and metadata.
        """
        state_chunks = self.state_chunks
        static_char_len = self.state_prefix_len + len(state_chunks)
        state_dict = {
            "pre": self.state_prefix,
            "chunks": state_chunks,
            "dyn": self.state_suffix,
            "metadata": {
                "state_idx": self.state_idx,
                # k_t — sequence version, incremented on each eviction
                "state_seq_ver": self.state_seq_ver,
                # j_t — length of static portion (pre + chunks)
                "static_char_len": static_char_len,
                # j_epsilon_t — static char len at last eviction boundary
                "evicted_char_length": self.evicted_static_len
            }
        }
        return json.dumps(state_dict)

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

        Delegates to _evict_and_reconcile() (with ratio=0 to clear all chunks).

        Thread-safe: lock is acquired inside _evict_and_reconcile().

        Args:
            request: Trigger service request (unused).
            response: Trigger service response to populate.

        Returns:
            Trigger.Response: Success status and message with chunk count.
        """
        with self.lock:
            total_count = self.state_seq.get_num_state_chunks()
        self._evict_and_reconcile(ratio=0.0)

        # Publish updated (empty) state
        self._publish_state()
        self.state_idx += 1

        response.success = True
        response.message = (
            f'State swept successfully. Cleared {total_count} chunks')
        self.get_logger().info(f'Sweep state completed: {response.message}')
        return response

    #####################
    #  Eviction
    #####################

    def _evict_and_reconcile(self, ratio: float | None = None) -> None:
        """Evict oldest chunks and immediately notify ASRManager to reconcile.

        This is the single entry point for all eviction operations. It ensures
        that StartReconciliation is always called after every eviction, so the
        two steps cannot be decoupled by accident during future refactoring.

        Internally, the lock-protected _evict() runs first, then
        _call_start_reconciliation() is invoked outside the lock (I/O must
        not be held under the state lock).

        Args:
            ratio: Fraction of chunks to retain after eviction. Defaults to
                self.state_seq_clear_ratio. Pass 0.0 to clear all chunks
                (used by sweep_state).
        """
        with self.lock:
            evicted_state = self._evict(ratio)
        self._call_start_reconciliation(evicted_state, self.state_seq_ver)

    def _evict(self, ratio: float | None = None) -> str:
        """Evict oldest state chunks and prepare the evicted state for ASR.

        Clears the oldest (1 - ratio) fraction of chunks from state_seq,
        rebuilds the cached chunk string from the survivors, increments
        state_seq_ver (k), and captures evicted_static_len (j_ε) as the
        static char length *after* eviction but *before* any new chunk is
        appended.

        MUST be called with self.lock held. Prefer _evict_and_reconcile()
        at call sites to guarantee the paired reconciliation notification.

        Args:
            ratio: Fraction of chunks to retain.  Defaults to
                self.state_seq_clear_ratio.  Pass 0.0 to clear all chunks
                (used by sweep_state).

        Returns:
            str: The evicted state string X_ε = pre + surviving chunks,
                passed to ASRManager via StartReconciliation.
        """
        if ratio is None:
            ratio = self.state_seq_clear_ratio

        self.state_seq.clear(ratio)
        self.state_seq_ver += 1

        # Rebuild cached string from surviving chunks
        surviving_chunks = [sc.chunk for sc in self.state_seq]
        self._cached_state_chunks_str = '\n'.join(surviving_chunks)

        # j_ε: static char length at eviction boundary (pre + surviving chunks)
        # This is captured *before* the triggering new chunk is appended.
        self.evicted_static_len = (
            self.state_prefix_len + len(self._cached_state_chunks_str)
        )

        # X_ε: full static content handed to ASRManager for KV-cache warmup
        evicted_state = self.state_prefix + self._cached_state_chunks_str

        self.get_logger().info(
            f'[EVICT] Eviction complete: '
            f'ratio={ratio:.2f} '
            f'k={self.state_seq_ver} '
            f'|X_ε|={len(evicted_state)}c '
            f'j_ε={self.evicted_static_len}c'
        )
        return evicted_state

    def _call_start_reconciliation(
        self, evicted_state: str, k_target: int
    ) -> None:
        """Send a non-blocking StartReconciliation request to ASRManager.

        Fire-and-forget: uses async_send_request so the calling thread is not
        blocked waiting for ASRManager to acknowledge.  A callback logs the
        outcome.

        Safe to call without self.lock held.

        Args:
            evicted_state: X_ε string (pre + surviving chunks after eviction).
            k_target: New sequence version (state_seq_ver after increment).
        """
        if not self._asr_client.service_is_ready():
            self.get_logger().warn(
                f'[RECON] StartReconciliation service not ready; '
                f'skipping notification for k_target={k_target}.'
            )
            return

        req = StartReconciliation.Request()
        req.evicted_state = evicted_state
        req.evicted_state_seq_ver = k_target

        future = self._asr_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_start_reconciliation_response(f, k_target)
        )
        self.get_logger().info(
            f'[RECON] StartReconciliation sent: '
            f'k_target={k_target} |X_ε|={len(evicted_state)}c'
        )

    def _on_start_reconciliation_response(
        self, future, k_target: int
    ) -> None:
        """Log the outcome of a StartReconciliation service call."""
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    f'[RECON] StartReconciliation acknowledged: '
                    f'k_target={k_target}'
                )
            else:
                self.get_logger().warn(
                    f'[RECON] StartReconciliation rejected by ASRManager: '
                    f'k_target={k_target}'
                )
        except Exception as e:
            self.get_logger().error(
                f'[RECON] StartReconciliation call failed: '
                f'k_target={k_target} error={e}'
            )

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
        num_tokens = (self.state_prefix_num_tokens
                      + self.state_chunks_num_tokens
                      + self.state_suffix_num_tokens)

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
