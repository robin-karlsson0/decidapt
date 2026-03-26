import json
import os

import pytest
from async_state_recon.asr_manager import ASRManager
from async_state_recon.dummy_asr_manager import DummyASRManager
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
# Adjust this import path based on where your base class is located
from tests.unit.test_asr_manager import (SERVER_STARTUP_TIMEOUT,
                                         BaseASRManagerTest, _wait_for_port)

# Fetch and parse URL strings as JSON
RAW_URLS = os.environ.get("INFERENCE_SERVER_URLS", "[]")
try:
    INFERENCE_SERVER_URLS = json.loads(RAW_URLS)
except json.JSONDecodeError:
    # Fallback to an empty list if the JSON is malformed
    INFERENCE_SERVER_URLS = []

# Valid URLs check
NUM_VALID_URLS = len(INFERENCE_SERVER_URLS)

MODEL_NAME = os.environ.get("MODEL_NAME", "")

# Apply the integration marker to all tests in this file
pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    NUM_VALID_URLS == 0,
    reason="Need 1x INFERENCE_SERVER_URLS set in environment variables")
class TestDummyASRIntegration(BaseASRManagerTest):
    """
    End-to-end integration tests using a real vLLM inference server.

    Inherits ROS 2 lifecycle management from BaseASRManagerTest but connects
    the DummyASRManager to a real inference server instead of injecting mock
    clients.
    """

    @classmethod
    def _get_port(cls) -> int:
        """Return a unique port for the HTTP server to avoid conflicts."""
        # Use a different starting range than the unit tests to prevent port
        # collisions if tests are ever run in parallel.
        if not hasattr(cls, '_starting_port'):
            cls._starting_port = 19100
            cls._port_counter = 0

        port = cls._starting_port + cls._port_counter
        cls._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for integration tests."""
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        print('Setting up TestASRIntegration class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after integration tests complete."""
        import rclpy
        print('Tearing down TestASRIntegration class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def create_real_dummy_asr_manager(self, port: int):
        """Instantiate DummyASRManager pointing to the real vLLM URLs."""
        # Reuse base class parameter generator, but inject the real vLLM URL
        params = self.create_test_parameters(r1_url=INFERENCE_SERVER_URLS[0],
                                             model_name=MODEL_NAME,
                                             http_port=port)
        manager = DummyASRManager(parameter_overrides=params)
        self.executor.add_node(manager)
        return manager

    def test_dummy_real_inference_blocking(self):
        """Validate standard blocking inference against a real vLLM server."""
        print("Testing: test_real_inference_blocking")
        port = self._get_port()
        self.asr_manager = self.create_real_dummy_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        # NOTE: You may need to update 'model' to match the actual model
        # loaded in your real vLLM deployment (e.g., 'meta-llama/Llama-2-7b-chat-hf')
        params = self.create_completion_request(model=MODEL_NAME)
        response = client.chat.completions.create(**params)

        assert isinstance(
            response,
            ChatCompletion), "Response is not a ChatCompletion object"
        assert response.choices, "Response choices should not be empty"

        content = response.choices[0].message.content
        print(content)
        assert content, "Response message content must not be empty"
        # Notice we don't assert a specific string match since real model output varies

    def test_dummy_real_inference_streaming(self):
        """Validate streaming inference against a real vLLM server."""
        print("Testing: test_real_inference_streaming")
        port = self._get_port()
        self.asr_manager = self.create_real_dummy_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        params = self.create_completion_request(model=MODEL_NAME, stream=True)
        response_stream = client.chat.completions.create(**params)

        collected_chunks = []
        collected_content = ""

        for chunk in response_stream:
            collected_chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content
                print(collected_content)

        assert len(
            collected_chunks) > 0, "Expected chunks for a streaming response"
        assert len(collected_content
                   ) > 0, "Streamed message content must not be empty"


@pytest.mark.skipif(
    not NUM_VALID_URLS == 2,
    reason="Need 2x INFERENCE_SERVER_URLS set in environment variables")
class TestASRIntegration(BaseASRManagerTest):
    """
    End-to-end integration tests using a real vLLM inference server.

    Inherits ROS 2 lifecycle management from BaseASRManagerTest but connects 
    the ASRManager to a real inference server instead of injecting mock clients.
    """

    @classmethod
    def _get_port(cls) -> int:
        """Return a unique port for the HTTP server to avoid conflicts."""
        # Use a different starting range than the unit tests to prevent port
        # collisions if tests are ever run in parallel.
        if not hasattr(cls, '_starting_port'):
            cls._starting_port = 19200
            cls._port_counter = 0

        port = cls._starting_port + cls._port_counter
        cls._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        """Initialize ROS 2 context for integration tests."""
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        print('Setting up TestASRIntegration class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS 2 context after integration tests complete."""
        import rclpy
        print('Tearing down TestASRIntegration class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    def create_real_asr_manager(self, port: int):
        """Instantiate ASRManager pointing to the real vLLM URLs."""
        # Reuse base class parameter generator, but inject the real vLLM URL
        params = self.create_test_parameters(r1_url=INFERENCE_SERVER_URLS[0],
                                             r2_url=INFERENCE_SERVER_URLS[1],
                                             model_name=MODEL_NAME,
                                             http_port=port)
        manager = ASRManager(parameter_overrides=params)
        self.executor.add_node(manager)
        return manager

    def test_real_inference_blocking(self):
        """Validate standard blocking inference against a real vLLM server."""
        port = self._get_port()
        self.asr_manager = self.create_real_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        params = self.create_completion_request(model=MODEL_NAME)
        response = client.chat.completions.create(**params)

        assert isinstance(
            response,
            ChatCompletion), "Response is not a ChatCompletion object"
        assert response.choices, "Response choices should not be empty"

        content = response.choices[0].message.content
        assert content, "Response message content must not be empty"
        # Notice we don't assert a specific string match since real model output varies

    def test_real_inference_streaming(self):
        """Validate streaming inference against a real vLLM server."""
        port = self._get_port()
        self.asr_manager = self.create_real_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        client = OpenAI(
            base_url=f'http://127.0.0.1:{port}/v1',
            api_key='dummy-key',
        )

        params = self.create_completion_request(model=MODEL_NAME, stream=True)
        response_stream = client.chat.completions.create(**params)

        collected_chunks = []
        collected_content = ""

        for chunk in response_stream:
            collected_chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                collected_content += chunk.choices[0].delta.content

        assert len(
            collected_chunks) > 0, "Expected chunks for a streaming response"
        assert len(collected_content
                   ) > 0, "Streamed message content must not be empty"
