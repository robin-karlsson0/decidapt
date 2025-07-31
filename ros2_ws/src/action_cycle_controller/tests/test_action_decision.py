"""
Test suite for ActionDecisionActionServer ROS 2 node with multiple LLM 
inference backends.

This module provides comprehensive integration tests for the 
ActionDecisionActionServer, which integrates ROS 2 action server functionality 
with Large Language Model (LLM) inference capabilities. The tests validate both 
the ROS 2 action interface and the underlying LLM inference implementations.

Test Structure:
    - BaseActionDecisionActionServerTest: Common test infrastructure and shared 
      test logic
    - TestActionDecisionActionServerTGI: Integration tests using TGI inference 
      backend
    - TestActionDecisionActionServerVLLM: Integration tests using vLLM 
      inference backend

Test Coverage:
    1. Docker container management for LLM inference servers
    2. Server health checks and readiness validation
    3. Direct inference API calls to validate server functionality
    4. ROS 2 action server initialization and parameter handling
    5. End-to-end action execution with real LLM inference
    6. Action decision validation for robot control

Each test class manages its own Docker container for the respective inference
server, ensuring isolated and reproducible test environments. Tests validate
both the technical integration (API calls, ROS 2 messaging) and functional
behavior (action decision generation).

Requirements:
    - Docker with GPU support
    - HF_TOKEN environment variable for Hugging Face model access
    - Sufficient GPU memory for model loading (configured for 1.5B parameter
      models)
"""
import os
import subprocess
import time

import rclpy
import requests
from action_cycle_controller.action_decision import ActionDecisionActionServer
from exodapt_robot_interfaces.action import ActionDecision
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

# Test Configuration for ActionDecisionActionServer Tests

# Model Configuration
DEFAULT_MODEL = "Qwen/Qwen2-1.5B-Instruct"
TEST_MAX_TOKENS = 1  # Single token output for action decisions
TEST_TEMPERATURE = 0.0
TEST_SEED = 14

# Server Configuration
TGI_PORT = 8000
VLLM_PORT = 8001  # Use different port to avoid conflicts
TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:3.3.2"
VLLM_IMAGE = "vllm/vllm-openai:latest"

# Timeouts (in seconds)
SERVER_STARTUP_TIMEOUT = 300  # model loading
ACTION_EXECUTION_TIMEOUT = 30  # inference
GOAL_PROCESSING_TIMEOUT = 5  # goal acceptance

# Test Parameters
TEST_STATE = "Test robot state: robot is idle. Take action to continue be idle."
TEST_VALID_ACTIONS = ('a: [Idle action] A no-operation action that represents '
                      'a decision to remain idle.')

# Docker Configuration
DOCKER_SHARED_MEMORY = "64g"
DOCKER_GPU_CONFIG = '"device=0"'
MODEL_MAX_LENGTH = 1024

ACTION_DECISION_ACTION_SERVER_NAME = 'action_decision_action_server'


class BaseActionDecisionActionServerTest:
    """
    Base class for ActionDecisionActionServer tests with common functionality.

    This class provides shared infrastructure for testing ActionDecisionActionServer
    with different LLM inference backends. It includes common setup/teardown
    methods and the core test logic that can be reused across different
    inference server implementations.

    Key Features:
        - Common test setup and cleanup
        - Shared action client creation and management
        - Unified test execution flow for different backends
        - Result validation and action decision verification

    The base class implements the core test pattern:
        1. Initialize ActionDecisionActionServer with backend-specific parameters
        2. Create action client 
        3. Send test goal and verify acceptance
        4. Wait for action completion and validate result
        5. Verify action decision format and content
    """

    def setup_method(self):
        """Initialize test state before each test method execution."""
        self.action_decision_action_server = None

    def teardown_method(self):
        """Clean up resources after each test method execution."""
        if self.action_decision_action_server:
            self.executor.remove_node(self.action_decision_action_server)
            self.action_decision_action_server.destroy_node()

    def _test_action_decision_with_server(self, inference_server_type,
                                          server_url):
        """
        Execute comprehensive end-to-end test of ActionDecisionActionServer.

        This method implements the complete test workflow for validating
        ActionDecisionActionServer functionality with a real LLM inference backend.

        Args:
            inference_server_type (str): Backend type ('tgi' or 'vllm')
            server_url (str): URL of the running inference server

        Test Flow:
            1. Configure ActionDecisionActionServer with backend-specific parameters
            2. Create action client
            3. Send test goal with dummy state and valid actions
            4. Verify goal acceptance and action execution
            5. Validate generated action decision content and format
            6. Confirm single-token response for fast action decisions

        Assertions:
            - Action server availability and goal acceptance
            - Non-empty string response from LLM inference
            - Reasonable action decision format (single token expected)
        """
        test_params = [
            Parameter('inference_server_type', Parameter.Type.STRING,
                      inference_server_type),
            Parameter('inference_server_url', Parameter.Type.STRING,
                      server_url),
            Parameter('max_tokens', Parameter.Type.INTEGER, TEST_MAX_TOKENS),
            Parameter('llm_temp', Parameter.Type.DOUBLE, TEST_TEMPERATURE),
            Parameter('llm_seed', Parameter.Type.INTEGER, TEST_SEED),
        ]

        self.action_decision_action_server = ActionDecisionActionServer(
            parameter_overrides=test_params)
        self.executor.add_node(self.action_decision_action_server)

        # Create separate client node
        client_node = rclpy.create_node('action_decision_client_test_node')
        self.executor.add_node(client_node)

        action_decision_client = ActionClient(
            client_node,
            ActionDecision,
            ACTION_DECISION_ACTION_SERVER_NAME,
        )

        try:
            # Wait for the action server to be available
            assert action_decision_client.wait_for_server(timeout_sec=10), \
                f"Action server '{ACTION_DECISION_ACTION_SERVER_NAME}' not available"

            # Create test goal
            goal_msg = ActionDecision.Goal()
            goal_msg.state = TEST_STATE
            goal_msg.valid_actions = TEST_VALID_ACTIONS

            # Send the goal
            send_goal_future = action_decision_client.send_goal_async(goal_msg)

            # Spin the executor to process the goal request
            start_time = time.time()
            while (not send_goal_future.done()
                   and (time.time() - start_time) < GOAL_PROCESSING_TIMEOUT):
                self.executor.spin_once(timeout_sec=0.1)

            # Check if future completed
            assert send_goal_future.done(), \
                "send_goal_future did not complete within timeout"

            goal_handle = send_goal_future.result()
            assert goal_handle is not None, \
                "Goal handle is None - future may have failed"
            assert goal_handle.accepted, \
                "Goal was not accepted by action server"

            print("Goal accepted, waiting for result...")

            # Wait for result
            get_result_future = goal_handle.get_result_async()

            # Spin executor until result is ready
            start_time = time.time()
            while (not get_result_future.done()
                   and (time.time() - start_time) < ACTION_EXECUTION_TIMEOUT):
                self.executor.spin_once(timeout_sec=0.1)

            assert get_result_future.done(), \
                "get_result_future did not complete within timeout"

            # Verify result
            result_response = get_result_future.result()
            assert result_response is not None, "No result response received"

            result = result_response.result
            assert result is not None, "No result received"
            assert hasattr(result,
                           'pred_action'), "Result missing pred_action field"
            pred_action = result.pred_action
            assert isinstance(pred_action,
                              str), "pred_action should be a string"
            assert len(pred_action) > 0, "pred_action should not be empty"

            print(f'Received action decision: "{pred_action}"')

            # Verify it's a reasonable response (single token with max_tokens=1)
            # Note: This might be more than one character due to tokenization
            assert len(pred_action.strip()) <= 10, \
                "pred_action should be relatively short for single token"

        except Exception as e:
            print(f"Test failed with error: {e}")
            print(
                f"Future done: {send_goal_future.done() if 'send_goal_future' in locals() else 'N/A'}"
            )
            if 'send_goal_future' in locals() and send_goal_future.done():
                print(f"Future result: {send_goal_future.result()}")
            raise

        finally:
            # Clean up client node
            self.executor.remove_node(client_node)
            client_node.destroy_node()


class TestActionDecisionActionServerTGI(BaseActionDecisionActionServerTest):
    """
    Integration tests for ActionDecisionActionServer with TGI inference backend.

    This test class validates the complete integration between ActionDecisionActionServer
    and TGI (Text Generation Inference) serving infrastructure. It manages
    a dedicated Docker container running TGI with HTTP API and executes
    comprehensive tests covering both direct API access and ROS 2 action
    interface functionality.

    Test Infrastructure:
        - Automated Docker container lifecycle management
        - TGI server with HTTP inference API endpoints
        - Health monitoring and readiness validation
        - Proper resource cleanup and error handling

    Test Categories:
        1. Infrastructure Tests:
           - Server health and availability validation
           - Direct API inference calls using TGI HTTP format

        2. Integration Tests:
           - ActionDecisionActionServer initialization with TGI backend
           - End-to-end action execution with real action decision generation
           - Action decision validation for robot control

    Container Configuration:
        - Model: Qwen/Qwen2-1.5B-Instruct (configured for test efficiency)
        - API: TGI HTTP /generate endpoint
        - Port: 8000 (default TGI port mapped to host)
        - GPU: Requires CUDA-capable device for model inference

    Note: Tests require HF_TOKEN environment variable for model access
    and sufficient GPU memory for 1.5B parameter model loading.
    """

    # Class-level variables for Docker container management
    docker_container_id = None
    tgi_port = TGI_PORT
    tgi_url = f"http://localhost:{tgi_port}"

    @classmethod
    def setup_class(cls):
        """Initialize ROS2 and start TGI inference server for all tests."""
        print("Setting up TGI test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

        # Start real TGI inference server
        cls._start_tgi_server()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS2 and stop TGI inference server after all tests."""
        print("Tearing down TGI test class...")

        # Stop TGI server
        cls._stop_tgi_server()

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    @classmethod
    def _start_tgi_server(cls):
        """Start the TGI inference server using Docker."""
        print("Starting TGI inference server...")

        # Configuration
        volume = f"/home/{os.getenv('USER', 'root')}/.cache/huggingface"
        hf_token = os.getenv('HF_TOKEN', '')

        # Docker command
        docker_cmd = [
            "docker",
            "run",
            "-d",  # -d for detached mode
            "--gpus",
            DOCKER_GPU_CONFIG,
            "--shm-size",
            DOCKER_SHARED_MEMORY,
            "-p",
            f"{cls.tgi_port}:80",
            "-v",
            f"{volume}:/data",
            "-e",
            f"HF_TOKEN={hf_token}",
            TGI_IMAGE,
            "--model-id",
            DEFAULT_MODEL,
            "--max-input-length",
            str(MODEL_MAX_LENGTH),
            "--max-total-tokens",
            str(MODEL_MAX_LENGTH + 1),
            "--max-batch-prefill-tokens",
            str(MODEL_MAX_LENGTH),
        ]

        try:
            # Print the docker command for debugging
            print(f"Running Docker command: {' '.join(docker_cmd)}")

            # Start the container
            result = subprocess.run(docker_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            cls.docker_container_id = result.stdout.strip()
            print(f"Started TGI container: {cls.docker_container_id}")

            # Show any startup output
            if result.stderr:
                print(f"Docker startup stderr: {result.stderr}")

            # Print container logs immediately after starting
            print("Getting initial container logs...")
            logs_result = subprocess.run(
                ["docker", "logs", cls.docker_container_id],
                capture_output=True,
                text=True,
                timeout=10)
            if logs_result.stdout:
                print(f"Container stdout:\n{logs_result.stdout}")
            if logs_result.stderr:
                print(f"Container stderr:\n{logs_result.stderr}")

            # Wait for server to be ready
            cls._wait_for_tgi_server()

        except subprocess.CalledProcessError as e:
            print(f"Failed to start TGI server: {e}")
            print(f"Docker stdout: {e.stdout}")
            print(f"Docker stderr: {e.stderr}")

            # If container was created but failed, get its logs
            if cls.docker_container_id:
                print("Getting container logs for failed container...")
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", cls.docker_container_id],
                        capture_output=True,
                        text=True,
                        timeout=10)
                    if logs_result.stdout:
                        print(
                            f"Failed container stdout:\n{logs_result.stdout}")
                    if logs_result.stderr:
                        print(
                            f"Failed container stderr:\n{logs_result.stderr}")
                except Exception as log_error:
                    print(f"Could not get container logs: {log_error}")
            raise

    @classmethod
    def _wait_for_tgi_server(cls,
                             timeout=SERVER_STARTUP_TIMEOUT,
                             check_interval=5):
        """Wait for TGI server to be ready to accept requests."""
        print("Waiting for TGI server to be ready...")

        start_time = time.time()
        attempt = 0
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = time.time() - start_time
            print(f"Health check attempt {attempt} (elapsed: {elapsed:.1f}s)")

            try:
                # Try to make a health check request
                response = requests.get(f"{cls.tgi_url}/health", timeout=10)
                if response.status_code == 200:
                    print("TGI server is ready!")
                    return
                else:
                    print(
                        f"Health check failed with status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Health check request failed: {e}")

            # Show container logs during wait
            if cls.docker_container_id:
                try:
                    logs_result = subprocess.run([
                        "docker", "logs", "--tail", "10",
                        cls.docker_container_id
                    ],
                                                 capture_output=True,
                                                 text=True,
                                                 timeout=5)
                    if logs_result.stdout or logs_result.stderr:
                        print("Recent container logs:")
                        if logs_result.stdout:
                            print(f"stdout: {logs_result.stdout}")
                        if logs_result.stderr:
                            print(f"stderr: {logs_result.stderr}")
                except Exception as log_error:
                    print(f"Could not get container logs: {log_error}")

            print(f"TGI server not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)

        # If we get here, the server didn't start in time
        print("Timeout reached, getting final container logs...")
        if cls.docker_container_id:
            try:
                logs_result = subprocess.run(
                    ["docker", "logs", cls.docker_container_id],
                    capture_output=True,
                    text=True,
                    timeout=10)
                if logs_result.stdout:
                    print(f"Final container stdout:\n{logs_result.stdout}")
                if logs_result.stderr:
                    print(f"Final container stderr:\n{logs_result.stderr}")
            except Exception as log_error:
                print(f"Could not get final container logs: {log_error}")

        cls._stop_tgi_server()  # Clean up
        raise TimeoutError(
            f"TGI server failed to start within {timeout} seconds")

    @classmethod
    def _stop_tgi_server(cls):
        """Stop and remove the TGI Docker container."""
        if cls.docker_container_id:
            print(f"Stopping TGI container: {cls.docker_container_id}")
            try:
                # Stop the container
                subprocess.run(["docker", "stop", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                # Remove the container
                subprocess.run(["docker", "rm", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                print("TGI container stopped and removed")
            except subprocess.CalledProcessError as e:
                print(f"Error stopping TGI container: {e}")
            finally:
                cls.docker_container_id = None

    def test_tgi_server_health(self):
        """
        Validate TGI server accessibility and health status.

        Tests the basic connectivity to the TGI server by making a health
        check request to the /health endpoint. This ensures the server
        is running and responding before conducting more complex tests.
        """
        response = requests.get(f"{self.tgi_url}/health")
        assert response.status_code == 200

    def test_tgi_inference_call(self):
        """
        Test direct inference call to TGI server using HTTP API.

        Validates that the TGI server can successfully process inference
        requests by sending a generation request and verifying the response
        format. This test confirms the server's action decision generation
        capabilities independently of ROS 2 integration.

        Tests:
            - TGI HTTP /generate endpoint
            - Proper request/response format handling
            - Action decision generation with controlled parameters
        """
        # Example inference request
        payload = {
            "inputs": "Hello, how are you?",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": TEST_TEMPERATURE,
                "seed": TEST_SEED
            }
        }

        response = requests.post(f"{self.tgi_url}/generate", json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "generated_text" in result
        print(f"Generated text: {result['generated_text']}")

    def test_initialization_with_default_parameters(self):
        """
        Test ActionDecisionActionServer initialization with default configuration.

        Validates that the ActionDecisionActionServer can be successfully instantiated
        with default parameters and properly integrates with the ROS 2
        executor system. This test ensures basic node functionality before
        testing inference capabilities.

        Verifies:
            - Successful node creation and type validation
            - Correct action server name assignment
            - Proper integration with ROS 2 executor
        """
        self.action_decision_action_server = ActionDecisionActionServer()
        self.executor.add_node(self.action_decision_action_server)

        # Check action server is created
        assert isinstance(self.action_decision_action_server,
                          ActionDecisionActionServer)
        assert (self.action_decision_action_server.action_server_name ==
                ACTION_DECISION_ACTION_SERVER_NAME)

    def test_action_decision_with_real_tgi(self):
        """
        Test complete ActionDecisionActionServer integration with TGI backend.

        Executes the comprehensive end-to-end test using the shared test
        logic from BaseActionDecisionActionServerTest. This validates the complete
        workflow from ROS 2 action goal submission through TGI inference
        to action decision generation.

        Tests the full integration chain:
            - ROS 2 action server configuration with TGI parameters
            - Action goal processing and acceptance
            - TGI inference execution with real action decision generation
            - Result validation and action decision verification
        """
        self._test_action_decision_with_server('tgi', self.tgi_url)


class TestActionDecisionActionServerVLLM(BaseActionDecisionActionServerTest):
    """
    Integration tests for ActionDecisionActionServer with vLLM inference backend.

    This test class validates the complete integration between ActionDecisionActionServer
    and vLLM (Virtual Large Language Model) serving infrastructure. It manages
    a dedicated Docker container running vLLM with OpenAI-compatible API and
    executes comprehensive tests covering both direct API access and ROS 2
    action interface functionality.

    Test Infrastructure:
        - Automated Docker container lifecycle management
        - vLLM server with OpenAI-compatible API endpoints
        - Health monitoring and readiness validation
        - Proper resource cleanup and error handling

    Test Categories:
        1. Infrastructure Tests:
           - Server health and availability validation
           - Direct API inference calls using OpenAI format

        2. Integration Tests:
           - ActionDecisionActionServer initialization with vLLM backend
           - End-to-end action execution with real action decision generation
           - Action decision validation for robot control

    Container Configuration:
        - Model: Qwen/Qwen2-1.5B-Instruct (configured for test efficiency)
        - API: OpenAI-compatible chat completions endpoint
        - Port: 8001 (isolated from TGI to avoid conflicts)
        - GPU: Requires CUDA-capable device for model inference

    Note: Tests require HF_TOKEN environment variable for model access
    and sufficient GPU memory for 1.5B parameter model loading.
    """

    # Class-level variables for Docker container management
    docker_container_id = None
    vllm_port = VLLM_PORT
    vllm_url = f"http://localhost:{vllm_port}"

    @classmethod
    def setup_class(cls):
        """Initialize ROS2 and start vLLM inference server for all tests."""
        print("Setting up vLLM test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

        # Start real vLLM inference server
        cls._start_vllm_server()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS2 and stop vLLM inference server after all tests."""
        print("Tearing down vLLM test class...")

        # Stop vLLM server
        cls._stop_vllm_server()

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    @classmethod
    def _start_vllm_server(cls):
        """Start the vLLM inference server using Docker."""
        print("Starting vLLM inference server...")

        # Configuration
        volume = f"/home/{os.getenv('USER', 'root')}/.cache/huggingface"
        hf_token = os.getenv('HF_TOKEN', '')

        # Docker command for vLLM
        docker_cmd = [
            "docker",
            "run",
            "-d",  # -d for detached mode
            "--gpus",
            DOCKER_GPU_CONFIG,
            "--shm-size",
            DOCKER_SHARED_MEMORY,
            "-p",
            f"{cls.vllm_port}:8000",
            "-v",
            f"{volume}:/root/.cache/huggingface",
            "-e",
            f"HF_TOKEN={hf_token}",
            VLLM_IMAGE,
            "--model",
            DEFAULT_MODEL,
            "--max-model-len",
            str(MODEL_MAX_LENGTH),
        ]

        try:
            # Print the docker command for debugging
            print(f"Running Docker command: {' '.join(docker_cmd)}")

            # Start the container
            result = subprocess.run(docker_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            cls.docker_container_id = result.stdout.strip()
            print(f"Started vLLM container: {cls.docker_container_id}")

            # Show any startup output
            if result.stderr:
                print(f"Docker startup stderr: {result.stderr}")

            # Print container logs immediately after starting
            print("Getting initial container logs...")
            logs_result = subprocess.run(
                ["docker", "logs", cls.docker_container_id],
                capture_output=True,
                text=True,
                timeout=10)
            if logs_result.stdout:
                print(f"Container stdout:\n{logs_result.stdout}")
            if logs_result.stderr:
                print(f"Container stderr:\n{logs_result.stderr}")

            # Wait for server to be ready
            cls._wait_for_vllm_server()

        except subprocess.CalledProcessError as e:
            print(f"Failed to start vLLM server: {e}")
            print(f"Docker stdout: {e.stdout}")
            print(f"Docker stderr: {e.stderr}")

            # If container was created but failed, get its logs
            if cls.docker_container_id:
                print("Getting container logs for failed container...")
                try:
                    logs_result = subprocess.run(
                        ["docker", "logs", cls.docker_container_id],
                        capture_output=True,
                        text=True,
                        timeout=10)
                    if logs_result.stdout:
                        print(
                            f"Failed container stdout:\n{logs_result.stdout}")
                    if logs_result.stderr:
                        print(
                            f"Failed container stderr:\n{logs_result.stderr}")
                except Exception as log_error:
                    print(f"Could not get container logs: {log_error}")
            raise

    @classmethod
    def _get_container_status(cls):
        """Get container status information for debugging."""
        if not cls.docker_container_id:
            return "No container ID available"

        try:
            # Get container status
            status_result = subprocess.run([
                "docker", "ps", "-a", "--filter",
                f"id={cls.docker_container_id}", "--format",
                "table {{.Status}}"
            ],
                                           capture_output=True,
                                           text=True,
                                           timeout=5)
            return status_result.stdout.strip()
        except Exception as e:
            return f"Could not get container status: {e}"

    @classmethod
    def _wait_for_vllm_server(cls,
                              timeout=SERVER_STARTUP_TIMEOUT,
                              check_interval=5):
        """Wait for vLLM server to be ready to accept requests."""
        print("Waiting for vLLM server to be ready...")

        start_time = time.time()
        attempt = 0
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = time.time() - start_time
            container_status = cls._get_container_status()
            print(f"Health check attempt {attempt} (elapsed: {elapsed:.1f}s)")
            print(f"Container status: {container_status}")

            try:
                # Try to make a health check request
                response = requests.get(f"{cls.vllm_url}/health", timeout=10)
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    return
                else:
                    print(
                        f"Health check failed with status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Health check request failed: {e}")

            # Show container logs during wait
            if cls.docker_container_id:
                try:
                    logs_result = subprocess.run([
                        "docker", "logs", "--tail", "10",
                        cls.docker_container_id
                    ],
                                                 capture_output=True,
                                                 text=True,
                                                 timeout=5)
                    if logs_result.stdout or logs_result.stderr:
                        print("Recent container logs:")
                        if logs_result.stdout:
                            print(f"stdout: {logs_result.stdout}")
                        if logs_result.stderr:
                            print(f"stderr: {logs_result.stderr}")
                except Exception as log_error:
                    print(f"Could not get container logs: {log_error}")

            print(f"vLLM server not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)

        # If we get here, the server didn't start in time
        print("Timeout reached, getting final container logs...")
        if cls.docker_container_id:
            try:
                logs_result = subprocess.run(
                    ["docker", "logs", cls.docker_container_id],
                    capture_output=True,
                    text=True,
                    timeout=10)
                if logs_result.stdout:
                    print(f"Final container stdout:\n{logs_result.stdout}")
                if logs_result.stderr:
                    print(f"Final container stderr:\n{logs_result.stderr}")
            except Exception as log_error:
                print(f"Could not get final container logs: {log_error}")

        cls._stop_vllm_server()  # Clean up
        raise TimeoutError(
            f"vLLM server failed to start within {timeout} seconds")

    @classmethod
    def _stop_vllm_server(cls):
        """Stop and remove the vLLM Docker container."""
        if cls.docker_container_id:
            print(f"Stopping vLLM container: {cls.docker_container_id}")
            try:
                # Stop the container
                subprocess.run(["docker", "stop", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                # Remove the container
                subprocess.run(["docker", "rm", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                print("vLLM container stopped and removed")
            except subprocess.CalledProcessError as e:
                print(f"Error stopping vLLM container: {e}")
            finally:
                cls.docker_container_id = None

    def test_vllm_server_health(self):
        """
        Validate vLLM server accessibility and health status.

        Tests the basic connectivity to the vLLM server by making a health
        check request to the /health endpoint. This ensures the server
        is running and responding before conducting more complex tests.
        """
        response = requests.get(f"{self.vllm_url}/health")
        assert response.status_code == 200

    def test_vllm_inference_call(self):
        """
        Test direct inference call to vLLM server using OpenAI-compatible API.

        Validates that the vLLM server can successfully process inference
        requests by sending a chat completion request and verifying the
        response format. This test confirms the server's action decision
        generation capabilities independently of ROS 2 integration.

        Tests:
            - OpenAI-compatible chat completions endpoint
            - Proper request/response format handling
            - Action decision generation with controlled parameters
        """
        # Example inference request using OpenAI-compatible API
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{
                "role": "user",
                "content": "Hello, how are you?"
            }],
            "max_tokens": 10,
            "temperature": TEST_TEMPERATURE,
            "seed": TEST_SEED
        }

        response = requests.post(f"{self.vllm_url}/v1/chat/completions",
                                 json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        print(f"Generated text: {result['choices'][0]['message']['content']}")

    def test_initialization_with_default_parameters(self):
        """
        Test ActionDecisionActionServer initialization with default configuration.

        Validates that the ActionDecisionActionServer can be successfully instantiated
        with default parameters and properly integrates with the ROS 2
        executor system. This test ensures basic node functionality before
        testing inference capabilities.

        Verifies:
            - Successful node creation and type validation
            - Correct action server name assignment
            - Proper integration with ROS 2 executor
        """
        self.action_decision_action_server = ActionDecisionActionServer()
        self.executor.add_node(self.action_decision_action_server)

        # Check action server is created
        assert isinstance(self.action_decision_action_server,
                          ActionDecisionActionServer)
        assert (self.action_decision_action_server.action_server_name ==
                ACTION_DECISION_ACTION_SERVER_NAME)

    def test_action_decision_with_real_vllm(self):
        """
        Test complete ActionDecisionActionServer integration with vLLM backend.

        Executes the comprehensive end-to-end test using the shared test
        logic from BaseActionDecisionActionServerTest. This validates the complete
        workflow from ROS 2 action goal submission through vLLM inference
        to action decision generation.

        Tests the full integration chain:
            - ROS 2 action server configuration with vLLM parameters
            - Action goal processing and acceptance
            - vLLM inference execution with real action decision generation
            - Result validation and action decision verification
        """
        self._test_action_decision_with_server('vllm', self.vllm_url)
