import json
import socket
import time
import uuid
from typing import Generator, List

import rclpy
from async_state_recon.asr_manager import ASRManager
from async_state_recon.dummy_asr_manager import DummyASRManager
from async_state_recon.inference_client import InferenceClient
from exodapt_robot_interfaces.srv import StartReconciliation
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import (ChatCompletion,
                                               ChatCompletionMessage, Choice)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

# Timeouts (in seconds)
SERVER_STARTUP_TIMEOUT = 5.0  # Time allowed for uvicorn to bind the port
SPIN_TIMEOUT = 0.1  # Executor spin timeout


def _wait_for_port(host: str,
                   port: int,
                   timeout: float = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll until the HTTP server accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class MockInferenceClient(InferenceClient):
    """Mock inference client for unit testing without real servers.

    This mock client simulates inference server behavior with configurable
    latency and tracks call history for validation.
    """

    def __init__(self,
                 name: str,
                 model_name: str,
                 url: str,
                 latency: float = 0.05,
                 **kwargs):
        """Initialize mock client.

        Args:
            name: Client name identifier
            model_name: Model identifier
            url: Server URL (not used in mock)
            latency: Simulated inference delay in seconds
        """
        super().__init__(name, model_name, url)
        self.latency = latency
        self.call_count = 0
        self.call_history: List[str] = []
        self.last_prompt = None

    def run(self,
            prompt: str,
            max_tokens: int = 1024,
            stream: bool = False,
            **kwargs):
        """Simulate inference with configurable latency.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (ignored in mock)
            stream: If True, return streaming generator;
                    else return ChatCompletion
            **kwargs: Additional arguments (ignored)

        Returns:
            If stream=False: ChatCompletion object matching the real
                vLLM API response.
            If stream=True: Generator yielding ChatCompletion chunks
                with delta content.
        """
        self.call_count += 1
        self.last_prompt = prompt
        self.call_history.append(prompt)

        if stream:
            return self._stream_inference(prompt)
        else:
            time.sleep(self.latency)  # Simulate inference time
            content = (f"Mock response {self.call_count} for "
                       f"prompt: {prompt[:50]}..")
            return ChatCompletion(
                id=f'chatcmpl-{uuid.uuid4().hex}',
                choices=[
                    Choice(
                        finish_reason='stop',
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content=content,
                            refusal=None,
                            role='assistant',
                        ),
                    )
                ],
                created=int(time.time()),
                model=self.model_name,
                object='chat.completion',
                usage=CompletionUsage(
                    completion_tokens=len(content.split()),
                    prompt_tokens=len(prompt.split()),
                    total_tokens=len(content.split()) + len(prompt.split()),
                ),
            )

    def _stream_inference(self, prompt: str) -> Generator:
        """Generate streaming chunks simulating token-by-token response.

        Yields ChatCompletionChunk objects with delta content for each
        simulated token. The final chunk includes usage statistics.

        Args:
            prompt: Input prompt (for generating mock response)

        Yields:
            ChatCompletionChunk objects with delta.content for tokens,
            then a final chunk.
        """
        time.sleep(self.latency)
        content = (f"Mock response {self.call_count} for "
                   f"prompt: {prompt[:50]}..")
        tokens = content.split()

        # Yield individual token chunks with small inter-token delay
        for i, token in enumerate(tokens):
            chunk = ChatCompletionChunk(
                id=f'chatcmpl-{uuid.uuid4().hex}',
                choices=[
                    ChunkChoice(
                        finish_reason=None,
                        index=0,
                        delta=ChoiceDelta(content=token + ' ',
                                          role='assistant'),
                    )
                ],
                created=int(time.time()),
                model=self.model_name,
                object='chat.completion.chunk',
            )
            yield chunk
            time.sleep(self.latency / len(tokens))

        # Final chunk with finish_reason='stop' and usage stats
        final_chunk = ChatCompletionChunk(
            id=f'chatcmpl-{uuid.uuid4().hex}',
            choices=[
                ChunkChoice(
                    finish_reason='stop',
                    index=0,
                    delta=ChoiceDelta(content=None, role=None),
                )
            ],
            created=int(time.time()),
            model=self.model_name,
            object='chat.completion.chunk',
            usage=CompletionUsage(
                completion_tokens=len(tokens),
                prompt_tokens=len(prompt.split()),
                total_tokens=len(tokens) + len(prompt.split()),
            ),
        )
        yield final_chunk


class BaseASRManagerTest:
    """
    Base class for ASRManager tests with common infrastructure.

    This class provides shared setup/teardown methods, helper functions,
    and utilities for testing ASRManager with both mock and real inference
    clients.

    Key Features:
        - ROS 2 executor and node lifecycle management
    """

    def setup_method(self):
        """Initialize test state before each test method."""
        self.asr_manager = None
        self.state_publisher = None
        self.test_node = None

    def teardown_method(self):
        """Clean up resources after each test method."""
        if self.asr_manager:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy()

        if self.state_publisher:
            self.executor.remove_node(self.test_node)
            self.test_node.destroy_node()

    ####################
    #  Helper methods
    ####################

    def create_test_parameters(self,
                               r1_url: str = 'http://localhost:8001',
                               r2_url: str = 'http://localhost:8002',
                               client_type: str = 'vllm',
                               model_name: str = 'test-model',
                               catchup_thresh: int = 64,
                               http_host: str = '0.0.0.0',
                               http_port: int = 8000):
        """Create Parameter objects for ASRManager testing.

        Args:
            r1_url: URL for the first inference server.
            r2_url: URL for the second inference server.
            client_type: Type of inference client (e.g. 'vllm').
            model_name: Model name identifier.
            catchup_thresh: Integer threshold used for catch-up behavior.
            http_host: HTTP host to bind for any local test server.
            http_port: HTTP port to bind for any local test server.

        Returns:
            List[Parameter]: List of `rclpy.parameter.Parameter` objects
                representing the above configuration values.
        """
        return [
            Parameter('r1_url', Parameter.Type.STRING, r1_url),
            Parameter('r2_url', Parameter.Type.STRING, r2_url),
            Parameter('client_type', Parameter.Type.STRING, client_type),
            Parameter('model_name', Parameter.Type.STRING, model_name),
            Parameter('catchup_thresh', Parameter.Type.INTEGER,
                      catchup_thresh),
            Parameter('http_host', Parameter.Type.STRING, http_host),
            Parameter('http_port', Parameter.Type.INTEGER, http_port),
        ]

    def create_asr_manager(self, port: int, is_dummy: bool = False):
        """Instantiate the requested manager variant with test parameters."""
        params = self.create_test_parameters(http_port=port)
        if is_dummy:
            manager = DummyASRManager(parameter_overrides=params)
        else:
            manager = ASRManager(parameter_overrides=params)

        self.executor.add_node(manager)
        return manager

    def inject_mocks(self, is_dummy: bool = False):
        """Helper to replace real inference clients with mocks."""
        # Use the property since we moved model_name to the base class
        model_name = self.asr_manager.model_name

        mock_r1 = MockInferenceClient('R1', model_name,
                                      'http://localhost:8001')

        if is_dummy:
            self.asr_manager._r_primary = mock_r1
            return mock_r1, None

        mock_r2 = MockInferenceClient('R2', model_name,
                                      'http://localhost:8002')
        self.asr_manager._r1 = mock_r1
        self.asr_manager._r2 = mock_r2
        self.asr_manager._r_primary = self.asr_manager._r1
        self.asr_manager._r_secondary = self.asr_manager._r2
        return mock_r1, mock_r2

    @staticmethod
    def create_completion_request(
        model: str = 'default_model',
        prompt: str = 'Hello ' * 10,
        stream: bool = False,
        max_tokens: int = 128,
        temperature: float = 0.7,
        seed: int = 42,
        extra_body: dict = None,
    ) -> dict:
        """Build parameters for client.chat.completions.create() call.

        Args:
            model: Model identifier
            prompt: User prompt/content
            stream: Enable streaming response
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            extra_body: Additional parameters (chat_template_kwargs,
                asr_metadata)

        Returns:
            dict: Parameters ready to unpack into create() call
        """
        # Add default value
        if not extra_body:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
                "asr_metadata": {
                    "state_idx": 0,
                    "state_seq_ver": 0,
                    "static_char_len": len('Hello hello '),
                    "evicted_char_length": len('Hello ')
                }
            }
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "extra_body": extra_body
        }


class TestASRManagerHTTPServer(BaseASRManagerTest):
    """
    Unit tests for ASRManager HTTP server and OpenAI-compatible API.

    This test class validates the HTTP server infrastructure and API
    compatibility. It tests the core functionality for:
        - HTTP server startup and port binding on initialization
        - Graceful shutdown with proper stop event signaling
        - OpenAI-compatible /v1/chat/completions endpoint integration
        - Mock inference client injection for testing without real servers
        - ChatCompletion response format validation

    Uses mock inference clients for fast, deterministic testing without
    infrastructure dependencies.
    """

    @classmethod
    def _get_port(self) -> int:
        """Return a unique port for each test to avoid interference."""
        port = TestASRManagerHTTPServer._starting_port + \
            TestASRManagerHTTPServer._port_counter
        TestASRManagerHTTPServer._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for all tests in this class."""
        print('Setting up TestASRManagerHTTPServer class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 18100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after all tests complete."""
        print('Tearing down TestASRManagerHTTPServer class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    ###########
    #  Tests
    ###########

    def test_http_server_starts_on_init(self):
        """HTTP server thread is created and the port becomes reachable."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f"HTTP server did not bind port {port} within \
                {SERVER_STARTUP_TIMEOUT}s"
        assert self.asr_manager._http_server_thread is not None, \
            "_http_server_thread attribute must be set"
        assert self.asr_manager._http_server_thread.is_alive(), \
            "_http_server_thread thread must be alive"

    def test_http_server_graceful_shutdown(self):
        """Destroying the node sets the http_server_should_stop event."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f"HTTP server did not bind port {port} within \
                {SERVER_STARTUP_TIMEOUT}s"

        assert self.asr_manager._http_server_thread is not None, \
            "_http_server_thread attribute must be set"
        assert self.asr_manager._http_server_thread.is_alive(), \
            "_http_server_thread thread must be alive"

        # Trigger graceful shutdown and verify the stop event is set.
        self.executor.remove_node(self.asr_manager)
        self.asr_manager.destroy()

        assert self.asr_manager._http_stop_event.is_set(), \
            "HTTP server stop event must be set by destroy()"

        # Prevent teardown_method from double-destroying
        self.asr_manager = None

    def test_http_server_openai_client_compatibility(self):
        """HTTP endpoint accepts a query and returns a ChatCompletion object.

        Verifies the full round-trip:
          1. ASRManager starts with its HTTP server.
          2. Mock inference clients are injected in place of real vLLM clients.
          3. A valid ASR state JSON is embedded as the last message content.
          4. The OpenAI Python client POSTs to /v1/chat/completions.
          5. The response is a ChatCompletion with non-empty message content.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        # Submit via the OpenAI-compatible client
        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )
        params = self.create_completion_request()
        response = client.chat.completions.create(**params)

        assert type(
            response
        ) is ChatCompletion, 'Response not of type \'ChatCompletion\''  # noqa
        assert response.choices, 'Response must have at least one choice'
        content = response.choices[0].message.content
        assert content, 'Response message content must not be empty'
        assert mock_r1.call_count == 1, \
            f'Expected 1 inference call on r_primary, got {mock_r1.call_count}'
        assert mock_r2.call_count == 0, \
            f'Expected 0 inference call on r_secondary, got {mock_r2.call_count}'  # noqa

    def test_http_server_streaming_response(self):
        """HTTP endpoint correctly yields SSE chunks when stream=True.

        Verifies the full streaming round-trip:
          1. ASRManager starts with its HTTP server.
          2. Mock inference clients are injected to simulate streaming tokens.
          3. The OpenAI Python client POSTs with stream=True.
          4. The response is a generator yielding chunked deltas.
          5. The reconstructed text matches the expected mock output.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        # Submit via the OpenAI-compatible client
        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        params = self.create_completion_request(stream=True)
        response_stream = client.chat.completions.create(**params)

        collected_chunks = []
        collected_content = ""

        # Iterate over the Server-Sent Events (SSE) stream
        for chunk in response_stream:
            collected_chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        # Assertions
        assert len(collected_chunks) > 1, \
            f'Expected multiple chunks for a streaming response, got {len(collected_chunks)}'  # noqa
        assert len(collected_content) > 0, \
            'Streamed message content must not be empty'
        assert 'Mock response 1 for prompt:' in collected_content, \
            'Reconstructed content did not match expected mock output'

        assert mock_r1.call_count == 1, \
            f'Expected 1 inference call on r_primary, got {mock_r1.call_count}'
        assert mock_r2.call_count == 0, \
            f'Expected 0 inference call on r_secondary, got {mock_r2.call_count}'  # noqa


class TestASRManagerCase1(BaseASRManagerTest):
    """
    Unit tests for inference queries routed as "Case 1: Sequence continuation".
    """

    @classmethod
    def _get_port(self) -> int:
        """Return a unique port for each test to avoid interference."""
        port = TestASRManagerCase1._starting_port + \
            TestASRManagerCase1._port_counter
        TestASRManagerCase1._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for all tests in this class."""
        print('Setting up TestASRManagerCase1 class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 18100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after all tests complete."""
        print('Tearing down TestASRManagerCase1 class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    ###########
    #  Tests
    ###########

    def test_case1_initial_request_initializes_state(self):
        """First inference request routes to r_primary and sets initial state
        cursors.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        state_json_str = json.dumps({
            "sequence": "pre chunks dyn",
            "j_t": 10,  # "pre chunks "
            "k_t": 0,
            "j_epsilon_t": 0
        })

        self.asr_manager.run(state_json_str)

        # Verify Routing
        assert mock_r1.call_count == 1, "Initial request must route to r_primary"  # noqa
        assert mock_r2.call_count == 0, "r_secondary should not be called"

        # Verify Metrics
        assert self.asr_manager.stats.case1_continuations == 1
        assert self.asr_manager.stats.total_inference_requests == 1

        # Verify State Updates
        assert self.asr_manager._j == 10
        assert self.asr_manager._x_recon == "pre chunks"

    def test_case1_sequence_growth_updates_recon_state(self):
        """Subsequent request with same sequence version k updates x_recon."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Initial request
        state_json_str = json.dumps({
            "sequence": "pre chunks dyn",
            "j_t": 10,  # "pre chunks"
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(state_json_str)

        # 2. Sequence grows
        state_json_str = json.dumps({
            "sequence": "pre chunks chunks dyn",
            "j_t": 17,  # "pre chunks chunks"
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(state_json_str)

        # Verify Routing
        assert mock_r1.call_count == 2
        assert mock_r2.call_count == 0
        assert self.asr_manager.stats.case1_continuations == 2

        # Verify State Updates
        assert self.asr_manager._j == 17
        assert self.asr_manager._x_recon == "pre chunks chunks"

    def test_case1_static_context_unchanged(self):
        """Subsequent request with same static context bypasses update."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Initial request
        state_json_str = json.dumps({
            "sequence": "pre chunks dyn",
            "j_t": 10,  # "pre chunks"
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(state_json_str)

        initial_recon = self.asr_manager._x_recon

        # 2. Dynamic task swapped, but static context remains exactly the same
        state_json_str = json.dumps({
            "sequence": "pre chunks DYN",
            "j_t": 10,  # "pre chunks"
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(state_json_str)

        # Verify Routing
        assert mock_r1.call_count == 2
        assert mock_r2.call_count == 0
        assert self.asr_manager.stats.case1_continuations == 2

        # Verify State Updates (should remain identical to step 1)
        assert self.asr_manager._j == 10
        assert self.asr_manager._x_recon == initial_recon


class TestASRManagerCase2(BaseASRManagerTest):
    """
    Unit tests for inference queries routed as "Case 2: New sequence version —
    reconciliation phase (k_t > k)".
    """

    @classmethod
    def _get_port(self) -> int:
        """Return a unique port for each test to avoid interference."""
        port = TestASRManagerCase2._starting_port + \
            TestASRManagerCase2._port_counter
        TestASRManagerCase2._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for all tests in this class."""
        print('Setting up TestASRManagerCase2 class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 18100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after all tests complete."""
        print('Tearing down TestASRManagerCase2 class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    ###############################################
    #  Tests | Case 2a: Bridging (k_t > k_ready)
    ###############################################

    def test_case2a_bridge_state_and_route_to_primary(self):
        """When k_t > k_ready, query bridges state and routes to r_primary."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Establish initial state (k=0)
        init_state = json.dumps({
            "sequence": "pre chunk1 chunk2 dyn",
            "j_t": 18,  # "pre chunk1 chunk2 "
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(init_state)

        # Verify Routing
        assert mock_r1.call_count == 1
        assert mock_r2.call_count == 0
        assert self.asr_manager.stats.case1_continuations == 1
        assert self.asr_manager.stats.case2_bridge == 0

        # Verify static sequence is maintained
        assert self.asr_manager._j_epsilon is None
        assert self.asr_manager._x_recon == "pre chunk1 chunk2 "

        # Evicted state: "pre chunk2 chunk3dyn"

        # 2. Trigger Case 2a: New sequence version (k=1), secondary not ready
        #    (k_ready=0)
        evicted_state = json.dumps({
            "sequence": "pre chunk2 chunk3 dyn",
            "j_t": 18,  # "pre chunk2 chunk3 "
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(evicted_state)

        # Verify Routing
        assert mock_r1.call_count == 2
        assert mock_r2.call_count == 0
        assert self.asr_manager.stats.case1_continuations == 1
        assert self.asr_manager.stats.case2_bridge == 1

        # Verify bridge state updates
        assert self.asr_manager._x_recon == "pre chunk1 chunk2 chunk3 "

        # Ensure eviction cursor advanced to current j_t
        assert self.asr_manager._j_epsilon == 18

        # Verify the actual prompt sent to LLM was bridged correctly
        assert mock_r1.last_prompt == "pre chunk1 chunk2 chunk3 dyn"

    def test_case2a_multiple_bridging_accumulates_catchup_buffer(self):
        """Successive requests before r_secondary is ready correctly accumulate
        delta buffers."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Establish initial state (k=0)
        init_state = json.dumps({
            "sequence": "pre chunk1 chunk2 dyn",
            "j_t": 18,  # "pre chunk1 chunk2 "
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(init_state)

        # 2. First bridge request (k=1)
        evicted_state_1 = json.dumps({
            "sequence": "pre chunk2 chunk3 dyn",
            "j_t": 18,  # "pre chunk2 chunk3 "
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(evicted_state_1)

        # Assert intermediate catch-up buffer
        assert self.asr_manager._delta_x_epsilon == "chunk3 "
        assert self.asr_manager._j_epsilon == 18

        # 3. Second bridge request (k=1)
        evicted_state_2 = json.dumps({
            "sequence": "pre chunk2 chunk3 chunk4 dyn",
            "j_t": 25,  # "pre chunk2 chunk3 chunk4"
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(evicted_state_2)

        # Verify Routing
        assert mock_r1.call_count == 3
        assert mock_r2.call_count == 0
        assert self.asr_manager.stats.case2_bridge == 2

        # Verify Catch-up Buffer Accumulation
        # Second delta = sequence[10:13] = "C2 "
        # Total delta = "fix C1 " + "C2 " = "fix C1 C2 "
        assert self.asr_manager._delta_x_epsilon == "chunk3 chunk4 "
        assert self.asr_manager._j_epsilon == 25

    ############################################
    #  Tests | Case 2b: Swap (k_t == k_ready)
    ############################################

    def test_case2b_swap_resources_when_ready(self):
        """When k_t == k_ready, atomically swap resources and route to new
            primary resource."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Establish initial state (k=0)
        init_state = json.dumps({
            "sequence": "pre chunk1 chunk2 dyn",
            "j_t": 18,  # "pre chunk1 chunk2 "
            "k_t": 0,
            "j_epsilon_t": 0
        })
        self.asr_manager.run(init_state)

        # 2. First bridge request (k=1)
        evicted_state_1 = json.dumps({
            "sequence": "pre chunk2 chunk3 dyn",
            "j_t": 18,  # "pre chunk2 chunk3 "
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(evicted_state_1)

        # Manually simulate the background reconciliation thread finishing
        with self.asr_manager._state_lock:
            self.asr_manager._k_ready = 1
            self.asr_manager._is_reconciling = True

        # 2. Trigger Case 2b: k_t (1) == k_ready (1)
        evicted_state_2 = json.dumps({
            "sequence": "pre chunk2 chunk3 chunk4 dyn",
            "j_t": 25,  # "pre chunk2 chunk3 chunk4 "
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(evicted_state_2)

        # Verify Routing: It should have swapped to mock_r2
        assert mock_r1.call_count == 2  # From initial + bridge state
        assert mock_r2.call_count == 1  # From swapped state
        assert self.asr_manager.stats.case2_swap == 1
        assert self.asr_manager.stats.swap_count == 1

        # Verify Resource Pointer Swap
        assert self.asr_manager._r_primary.name == 'R2'
        assert self.asr_manager._r_secondary.name == 'R1'

        # Verify State Resets & Cursors
        assert not self.asr_manager._is_reconciling
        assert self.asr_manager._j == 25
        assert self.asr_manager._k == 1
        assert self.asr_manager._j_epsilon is None
        assert self.asr_manager._x_recon == "pre chunk2 chunk3 chunk4 "

        # Verify the prompt sent to the new primary resource is exactly the
        # requested sequence
        assert mock_r2.last_prompt == "pre chunk2 chunk3 chunk4 dyn"


class TestASRManagerCase3(BaseASRManagerTest):
    """
    Unit tests for inference queries routed as "Case 3: Straggler queries
    (k_t < k)".
    """

    @classmethod
    def _get_port(self) -> int:
        """Return a unique port for each test to avoid interference."""
        port = TestASRManagerCase3._starting_port + \
            TestASRManagerCase3._port_counter
        TestASRManagerCase3._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for all tests in this class."""
        print('Setting up TestASRManagerCase3 class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 18100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after all tests complete."""
        print('Tearing down TestASRManagerCase3 class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    ######################################################
    #  Tests | Case 3: Straggler Queries (k_t < k)
    ######################################################

    def test_case3_straggler_when_not_reconciling_routes_secondary(self):
        """When not reconciling, a straggler query routes to r_secondary.

        This tests the path where the secondary resource holds the intact
        KV cache for the old sequence, so it can handle delayed requests.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # Manually advance internal state to simulate we are on sequence k=1
        # and reconciliation has finished.
        with self.asr_manager._state_lock:
            self.asr_manager._k = 1
            self.asr_manager._is_reconciling = False

        # Create a delayed state request from an older sequence (k_t=0)
        straggler_state = json.dumps({
            "sequence": "pre chunk1 chunk2 dyn",
            "j_t": 18,  # "pre chunk1 chunk2 "
            "k_t": 0,
            "j_epsilon_t": 0
        })

        self.asr_manager.run(straggler_state)

        # Verify Routing: Must go to secondary resource
        assert mock_r1.call_count == 0, "r_primary should not be called"
        assert mock_r2.call_count == 1, "Straggler must route to r_secondary"

        # Verify prompt passed matches the exact straggler query
        assert mock_r2.last_prompt == "pre chunk1 chunk2 dyn"

    def test_case3_straggler_when_reconciling_routes_primary(self):
        """When reconciling, a straggler query routes to r_primary with bridged
        state.

        This tests the protection mechanism: if r_secondary is actively
        precomputing the KV cache for a new sequence, sending a straggler there
        would thrash the cache. It must dynamically bridge the task  using
        r_primary's current reconciliation state.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # Manually advance internal state to simulate we are on sequence k=1,
        # actively reconciling, and x_recon has been accumulating new chunks.
        with self.asr_manager._state_lock:
            self.asr_manager._k = 1
            self.asr_manager._is_reconciling = True
            # The extended state built during Case 2a bridging
            self.asr_manager._x_recon = "pre chunk2 chunk3 chunk4 "

        # Create a delayed state request from an older sequence (k_t=0)
        straggler_state = json.dumps({
            "sequence": "pre chunk1 chunk2 dyn",
            "j_t": 18,  # "pre chunk1 chunk2 "
            "k_t": 0,
            "j_epsilon_t": 0
        })

        self.asr_manager.run(straggler_state)

        # Verify Routing: Must go to primary resource to protect secondary
        assert mock_r1.call_count == 1, "Protected straggler must route to r_primary"  # noqa
        assert mock_r2.call_count == 0, "r_secondary should not be interrupted"

        # Verify Prompt Bridging (X'_t <- X_recon + X_dyn)
        expected_bridged_prompt = "pre chunk2 chunk3 chunk4 dyn"

        assert mock_r1.last_prompt == expected_bridged_prompt
        assert mock_r1.last_prompt == expected_bridged_prompt


class TestASRManagerReconciliation(BaseASRManagerTest):
    """
    Unit tests for the Asynchronous State Reconciliation background process
    and its triggering ROS 2 service.
    """

    @classmethod
    def _get_port(self) -> int:
        port = TestASRManagerReconciliation._starting_port + \
            TestASRManagerReconciliation._port_counter
        TestASRManagerReconciliation._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        print('Setting up TestASRManagerReconciliation class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 18100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        print('Tearing down TestASRManagerReconciliation class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    ######################################################
    #  Tests | Background Reconcile Daemon
    ######################################################

    def test_reconcile_basic_warmup_completes_and_authorizes_swap(self):
        """A simple reconciliation completes prefill and sets k_ready."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # Reconciliation call variables
        x_epsilon = "pre chunk2 "
        k_target = 1

        # Skip service callback ==> Trigger the background daemon directly
        self.asr_manager._start_reconciliation(x_epsilon, k_target)

        # Wait for the background thread to finish its work
        self.asr_manager._reconciliation_thread.join(timeout=2.0)

        # Assertions
        assert not self.asr_manager._reconciliation_thread.is_alive()
        assert mock_r2.call_count == 1, "Expected exactly 1 prefill call on r_secondary"  # noqa
        assert self.asr_manager._k_ready == k_target, "k_ready was not updated"
        assert self.asr_manager._delta_x_epsilon == "", "Catch-up buffer not cleared"  # noqa

    def test_reconcile_catchup_loop_drains_buffer(self):
        """Reconciliation loops if delta buffer grows beyond threshold during
        warmup."""
        port = self._get_port()
        # Set a very small catchup threshold so we can easily trigger the loop
        self.asr_manager = self.create_asr_manager(port)
        self.asr_manager.catchup_thresh = 10
        mock_r1, mock_r2 = self.inject_mocks()

        # Artificially slow down r_secondary so we have time to inject data
        mock_r2.latency = 0.2

        x_epsilon = "pre chunk2 "
        k_target = 1

        # 1. Start the slow background warmup
        self.asr_manager._start_reconciliation(x_epsilon, k_target)

        # 2. Yield briefly to ensure the background thread locks r_secondary and
        #    starts
        time.sleep(0.05)

        # 3. Simulate concurrent Case 2a inference requests filling the buffer
        # We inject a string larger than catchup_thresh (10 chars)
        injected_chunks = "chunk3 chunk4 chunk5 "
        with self.asr_manager._delta_x_epsilon_lock:
            self.asr_manager._delta_x_epsilon += injected_chunks

        # 4. Wait for the catch-up loop to process the injected data and finish
        self.asr_manager._reconciliation_thread.join(timeout=2.0)

        # Assertions
        assert not self.asr_manager._reconciliation_thread.is_alive()

        # It should take at least 2 calls: 1 for the initial warmup, 1 for the
        # catch-up loop
        assert mock_r2.call_count >= 2, "Catch-up loop did not cycle as expected"  # noqa
        assert self.asr_manager._k_ready == k_target, "Swap was not authorized"

        with self.asr_manager._delta_x_epsilon_lock:
            assert self.asr_manager._delta_x_epsilon == "", "Final drain failed to empty buffer"  # noqa

    ######################################################
    #  Tests | StartReconciliation ROS 2 Service
    ######################################################

    def test_service_trigger_and_full_reconciliation_lifecycle(self):
        """
        Verifies the full lifecycle of a service-triggered reconciliation:
        1. Service trigger (starts daemon, active state unchanged)
        2. Daemon completion (updates k_ready, leaves k unchanged)
        3. Next inference request (triggers Case 2b swap, finalizes state)
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # Artificially slow down r_secondary so we have time to inject data
        mock_r2.latency = 0.2

        # Setup initial active state (Sequence k=0)
        self.asr_manager._k = 0
        self.asr_manager._k_ready = 0
        self.asr_manager._j_epsilon = None
        self.asr_manager._x_recon = "pre chunks "

        request = StartReconciliation.Request()
        request.evicted_state = "pre chunk2 "
        request.evicted_state_seq_ver = 1
        response = StartReconciliation.Response()

        # ==========================================
        # PHASE 1: Trigger and verify active state
        # ==========================================
        result = self.asr_manager._start_reconciliation_callback(
            request, response)

        assert result.success is True
        assert self.asr_manager._k == 0  # Active sequence hasn't swapped
        assert self.asr_manager._k_ready == 0  # Secondary not ready yet
        assert self.asr_manager._is_reconciling is True
        assert self.asr_manager._reconciliation_thread.is_alive()

        # ==========================================
        # PHASE 2: Wait for daemon to complete
        # ==========================================
        self.asr_manager._reconciliation_thread.join(timeout=2.0)

        assert not self.asr_manager._reconciliation_thread.is_alive()
        assert mock_r2.call_count == 1, "r_secondary should have been warmed up"

        # The daemon authorizes the swap by updating k_ready
        assert self.asr_manager._k_ready == 1

        # Crucially, k and is_reconciling do NOT change yet!
        # They wait for the next inference request to execute the swap.
        assert self.asr_manager._k == 0
        assert self.asr_manager._is_reconciling is True

        # ==========================================
        # PHASE 3: The Atomic Swap (Case 2b)
        # ==========================================
        # An inference request arrives matching the newly ready sequence
        next_inference_state = json.dumps({
            "sequence": "pre chunk2 chunk 3 dyn",
            "j_t": 19,  # "pre chunk2 chunk 3 "
            "k_t": 1,
            "j_epsilon_t": 11
        })
        self.asr_manager.run(next_inference_state)

        # Now the state should be fully finalized
        assert self.asr_manager._k == 1
        assert self.asr_manager._is_reconciling is False
        assert self.asr_manager._r_primary.name == 'R2'
        assert self.asr_manager._r_secondary.name == 'R1'

    def test_service_rejects_stale_and_duplicate_triggers(self):
        """The service protects against overlapping or outdated sequence ver."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port)
        mock_r1, mock_r2 = self.inject_mocks()

        # 1. Test Stale Request Guard
        self.asr_manager._k = 5
        request = StartReconciliation.Request()
        request.evicted_state = "pre chunk "
        request.evicted_state_seq_ver = 3  # Older than current k

        response = StartReconciliation.Response()
        result = self.asr_manager._start_reconciliation_callback(
            request, response)

        assert result.success is False
        assert self.asr_manager._k == 5  # Unchanged
        assert self.asr_manager._reconciliation_thread is None

        # 2. Test Concurrent Lock Guard
        self.asr_manager._k = 5
        self.asr_manager._is_reconciling = True  # Simulate active daemon

        request.evicted_state_seq_ver = 6  # Valid new k
        result = self.asr_manager._start_reconciliation_callback(
            request, response)

        assert result.success is False
        assert self.asr_manager._k == 5  # Unchanged, trigger rejected


class TestDummyASRManager(BaseASRManagerTest):
    """Unit tests for the single-resource synchronous fallback manager."""

    @classmethod
    def _get_port(self) -> int:
        port = TestDummyASRManager._starting_port + TestDummyASRManager._port_counter
        TestDummyASRManager._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        rclpy.init()
        cls.executor = MultiThreadedExecutor()
        cls._starting_port = 19100
        cls._port_counter = 0

    @classmethod
    def teardown_class(cls):
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        self.asr_manager = None

    def teardown_method(self):
        if self.asr_manager is not None:
            self.executor.remove_node(self.asr_manager)
            self.asr_manager.destroy_node()
            self.asr_manager = None

    def test_dummy_manager_routes_sync_inference(self):
        """Verify inference requests are passed directly to the single client."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port, is_dummy=True)
        mock_r1, _ = self.inject_mocks(is_dummy=True)

        state_json_str = json.dumps({
            "sequence": "pre chunks dyn",
            "j_t": 10,
            "k_t": 0,
            "j_epsilon_t": 0
        })

        self.asr_manager.run(state_json_str)

        assert mock_r1.call_count == 1, "Request must route to primary mock"
        assert self.asr_manager.stats.total_inference_requests == 1
        assert mock_r1.last_prompt == "pre chunks dyn"

    def test_dummy_manager_acknowledges_reconciliation_safely(self):
        """Verify the dummy service immediately returns True without threads."""
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port, is_dummy=True)

        request = StartReconciliation.Request()
        request.evicted_state = "pre chunk2 "
        request.evicted_state_seq_ver = 1
        response = StartReconciliation.Response()

        result = self.asr_manager._dummy_reconciliation_callback(
            request, response)

        assert result.success is True, "Service must acknowledge eviction"
        assert not hasattr(self.asr_manager, '_reconciliation_thread'), \
            "Dummy should not possess or spawn reconciliation threads"

    def test_dummy_manager_http_server_openai_client_compatibility(self):
        """HTTP endpoint accepts a query and returns a ChatCompletion object.

        Verifies the full round-trip for the single-resource fallback:
          1. DummyASRManager starts with its HTTP server.
          2. A single mock inference client is injected.
          3. The OpenAI Python client POSTs to /v1/chat/completions.
          4. The response is a ChatCompletion with non-empty message content.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port, is_dummy=True)
        mock_r1, _ = self.inject_mocks(is_dummy=True)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        # Submit via the OpenAI-compatible client
        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )
        params = self.create_completion_request()
        response = client.chat.completions.create(**params)

        assert type(
            response
        ) is ChatCompletion, 'Response not of type \'ChatCompletion\''
        assert response.choices, 'Response must have at least one choice'
        content = response.choices[0].message.content
        assert content, 'Response message content must not be empty'

        # We only verify r_primary, as r_secondary doesn't exist in dummy mode
        assert mock_r1.call_count == 1, \
            f'Expected 1 inference call on r_primary, got {mock_r1.call_count}'

    def test_dummy_manager_http_server_streaming_response(self):
        """HTTP endpoint correctly yields SSE chunks when stream=True.

        Verifies the full streaming round-trip for the single-resource fallback:
          1. DummyASRManager starts with its HTTP server.
          2. A single mock inference client is injected to simulate streaming.
          3. The OpenAI Python client POSTs with stream=True.
          4. The response is a generator yielding chunked deltas.
          5. The reconstructed text matches the expected mock output.
        """
        port = self._get_port()
        self.asr_manager = self.create_asr_manager(port, is_dummy=True)
        mock_r1, _ = self.inject_mocks(is_dummy=True)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        # Submit via the OpenAI-compatible client
        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        params = self.create_completion_request(stream=True)
        response_stream = client.chat.completions.create(**params)

        collected_chunks = []
        collected_content = ""

        # Iterate over the Server-Sent Events (SSE) stream
        for chunk in response_stream:
            collected_chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        # Assertions
        assert len(collected_chunks) > 1, \
            f'Expected multiple chunks for a streaming response, got {len(collected_chunks)}'
        assert len(collected_content) > 0, \
            'Streamed message content must not be empty'
        assert 'Mock response 1 for prompt:' in collected_content, \
            'Reconstructed content did not match expected mock output'

        assert mock_r1.call_count == 1, \
            f'Expected 1 inference call on r_primary, got {mock_r1.call_count}'
