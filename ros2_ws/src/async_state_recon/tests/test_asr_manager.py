import socket
import time
import uuid
from typing import Generator, List

import rclpy
from async_state_recon.asr_manager import ASRManager
from async_state_recon.inference_client import InferenceClient
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
SERVER_STARTUP_TIMEOUT = 5.0   # Time allowed for uvicorn to bind the port
SPIN_TIMEOUT = 0.1             # Executor spin timeout


def _wait_for_port(
        host: str, port: int, timeout: float = SERVER_STARTUP_TIMEOUT) -> bool:
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

    def run(self, prompt: str, max_tokens: int = 1024,
            stream: bool = False, **kwargs):
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
                        delta=ChoiceDelta(
                            content=token + ' ', role='assistant'
                        ),
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
            Parameter('catchup_thresh', Parameter.Type.INTEGER, catchup_thresh),
            Parameter('http_host', Parameter.Type.STRING, http_host),
            Parameter('http_port', Parameter.Type.INTEGER, http_port),
        ]

    def create_asr_manager(self, port: int) -> ASRManager:
        """Instantiate ASRManager with test parameters and register it."""
        asr_manager = ASRManager(
            parameter_overrides=self.create_test_parameters(http_port=port))
        self.executor.add_node(asr_manager)
        return asr_manager

    def inject_mocks(self):
        """Helper to replace real inference clients with mocks."""
        model_name = self.asr_manager.get_model_name()
        mock_r1 = MockInferenceClient('R1', model_name, 'http://localhost:8001')
        mock_r2 = MockInferenceClient('R2', model_name, 'http://localhost:8002')
        self.asr_manager._r1 = mock_r1
        self.asr_manager._r2 = mock_r2
        self.asr_manager._r_primary = self.asr_manager._r1
        self.asr_manager._r_secondary = self.asr_manager._r2
        return mock_r1, mock_r2

    @staticmethod
    def create_completion_request(
        model: str = 'default_model',
        prompt: str = 'Hello hello hello',
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
            extra_body: Additional parameters (chat_template_kwargs, asr_metadata)

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

        assert type(response) is ChatCompletion, 'Response not of type \'ChatCompletion\''  # noqa
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
    Unit tests for inference queries routed as Case 1: Sequence continuation
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

    