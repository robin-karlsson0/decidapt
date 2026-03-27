import json
import os
import threading
import time

import pytest
from async_state_recon.asr_manager import ASRManager
from async_state_recon.dummy_asr_manager import DummyASRManager
from exodapt_robot_interfaces.srv import StartReconciliation
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
# Adjust this import path based on where your base class is located
from tests.unit.test_asr_manager import BaseASRManagerTest, _wait_for_port

# ---------------------------------------------------------------------------
# Context constants shared across the two-phase test
# ---------------------------------------------------------------------------

# A single repeatable sentence that provides ~45 chars per repetition
_SENTENCE = "The quick brown fox jumps over the lazy dog. "

# Stable system prompt / preamble  (~900 chars, 20 repetitions)
_PRE = _SENTENCE * 20

# Initial conversational context appended after PRE (~2700 chars)
_CHUNKS_INITIAL = _SENTENCE * 60

# Full initial static context presented on the first Case 1 requests
_INITIAL_STATIC = _PRE + _CHUNKS_INITIAL

# The per-request task instruction appended after j_t
_DYNAMIC = "Please summarize the key points of the above text."

# --- Phase 1 eviction: keep last ~30% of CHUNKS_INITIAL on r_secondary ---
_PHASE1_KEPT = _CHUNKS_INITIAL[int(len(_CHUNKS_INITIAL) * 0.70):]
# X_ε for phase 1:  PRE + last-30%-of-chunks
_X_EPSILON_1 = _PRE + _PHASE1_KEPT

# --- Phase 2 eviction: evict all chunks, keep only PRE ---
_X_EPSILON_2 = _PRE

# Bridge chunk appended once per bridge-thread iteration (~450 chars)
_BRIDGE_CHUNK = _SENTENCE * 10

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

        params = self.create_completion_request(model=MODEL_NAME)
        response = client.chat.completions.create(**params)

        assert isinstance(
            response,
            ChatCompletion), "Response is not a ChatCompletion object"
        assert response.choices, "Response choices should not be empty"

        content = response.choices[0].message.content
        print(content)
        assert content, "Response message content must not be empty"
        # Don't assert a specific string match since real model output varies

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
        # Don't assert a specific string match since real model output varies

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


@pytest.mark.skipif(
    not NUM_VALID_URLS == 2,
    reason="Need 2x INFERENCE_SERVER_URLS set in environment variables")
class TestASRTwoReconciliationPhases(BaseASRManagerTest):
    """
    End-to-end integration test demonstrating two complete Algorithm 1
    reconciliation cycles with real vLLM inference servers.

    Scenario
    --------
    Phase 0 — Baseline (Case 1):
        Several inference requests arrive on sequence k=0.  All are served on
        R1 (r_primary).  The static context grows to ~3600 chars.

    Phase 1 — First Reconciliation (k=0 → k=1):
        A StartReconciliation service call evicts 70% of the accumulated
        context.  R2 (r_secondary) starts precomputing the KV cache for the
        shorter evicted sequence X_ε_1.  Meanwhile a bridge thread fires
        Case 2a inference requests concurrently, each extending the static
        prefix with a new BRIDGE_CHUNK while R1 continues to serve them.
        Once R2 is ready (k_ready=1), a Case 2b request atomically swaps
        the resource pointers: R2 becomes r_primary, R1 becomes r_secondary.

    Phase 2 — Second Reconciliation (k=1 → k=2):
        Two Case 1 requests confirm inference works on the new primary (R2).
        A second StartReconciliation call evicts all chunks, leaving only PRE
        on r_secondary (R1).  A second bridge thread runs concurrently until
        k_ready=2.  A final Case 2b request completes the second swap, putting
        R1 back as r_primary.

    Assertions
    ----------
    * Every real inference call returns a non-empty ChatCompletion response.
    * ASRStats counters (case1_continuations, case2_bridge, case2_swap,
      swap_count) match expected values at each phase boundary.
    * Resource pointers (r_primary / r_secondary) correctly reflect each swap.
    * is_reconciling is False and k matches k_target after each Case 2b swap.
    """

    _RECONCILIATION_TIMEOUT = 300.0  # seconds to wait for prefill to finish
    _BRIDGE_INTERVAL = 2.0  # seconds between bridge-thread requests

    @classmethod
    def _get_port(cls) -> int:
        if not hasattr(cls, '_starting_port'):
            cls._starting_port = 19300
            cls._port_counter = 0
        port = cls._starting_port + cls._port_counter
        cls._port_counter += 1
        return port

    @classmethod
    def setup_class(cls):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        print('Setting up TestASRTwoReconciliationPhases class...')
        rclpy.init()
        cls.executor = MultiThreadedExecutor()

    @classmethod
    def teardown_class(cls):
        import rclpy
        print('Tearing down TestASRTwoReconciliationPhases class...')
        cls.executor.shutdown()
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def create_real_asr_manager(self, port: int) -> ASRManager:
        """Instantiate ASRManager with real vLLM URLs and production defaults.
        """
        params = self.create_test_parameters(
            r1_url=INFERENCE_SERVER_URLS[0],
            r2_url=INFERENCE_SERVER_URLS[1],
            model_name=MODEL_NAME,
            catchup_thresh=512,
            http_port=port,
        )
        manager = ASRManager(parameter_overrides=params)
        self.executor.add_node(manager)
        return manager

    @staticmethod
    def _make_state_json(sequence: str, j_t: int, k_t: int,
                         j_epsilon_t: int) -> str:
        return json.dumps({
            'sequence': sequence,
            'j_t': j_t,
            'k_t': k_t,
            'j_epsilon_t': j_epsilon_t,
        })

    def _wait_for_k_ready(self,
                          target: int,
                          timeout: float = _RECONCILIATION_TIMEOUT) -> bool:
        """Poll get_status() until k_ready reaches *target* or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.asr_manager.get_status()['k_ready'] >= target:
                return True
            time.sleep(0.5)
        return False

    def _run_bridge_thread(self, base_static: str, j_epsilon_t_val: int,
                           k_t_val: int) -> threading.Thread:
        """Spawn thread sending Case 2a bridge reqs until k_ready >= k_t_val.

        Each iteration appends one _BRIDGE_CHUNK to the growing static prefix
        so that _delta_x_epsilon accumulates across calls.  The thread exits
        naturally once the reconciliation background daemon has finished and
        k_ready matches the target sequence version.
        """
        manager = self.asr_manager

        def _bridge_fn():
            accumulated_static = base_static
            while True:
                # Exit condition: reconciliation finished, no more bridging req.
                if manager.get_status()['k_ready'] >= k_t_val:
                    print(
                        f'[BRIDGE] k_ready reached {k_t_val}, stopping bridge '
                        f'thread (k_t={k_t_val})')
                    break

                # Extend static prefix with one more bridge chunk
                accumulated_static += _BRIDGE_CHUNK
                j_t = len(accumulated_static)
                sequence = accumulated_static + _DYNAMIC

                state_json = self._make_state_json(
                    sequence=sequence,
                    j_t=j_t,
                    k_t=k_t_val,
                    j_epsilon_t=j_epsilon_t_val,
                )

                response = manager.run(state_json, max_tokens=32)
                content = response.choices[0].message.content
                print(f'[BRIDGE] k_t={k_t_val} j_t={j_t} '
                      f'response_len={len(content or "")}')

                time.sleep(self._BRIDGE_INTERVAL)

        t = threading.Thread(target=_bridge_fn,
                             daemon=True,
                             name=f'bridge-k{k_t_val}')
        t.start()
        return t

    # ------------------------------------------------------------------
    # The test
    # ------------------------------------------------------------------

    def test_two_reconciliation_phases(self):
        """Demonstrate two full reconciliation cycles with real vLLM servers.

        See class docstring for the complete scenario description.
        """
        port = self._get_port()
        self.asr_manager = self.create_real_asr_manager(port)

        assert _wait_for_port('127.0.0.1', port), \
            f'HTTP server did not start on port {port}'

        # ==============================================================
        # Phase 0: Baseline Case 1 — establish initial static context
        # ==============================================================
        print('\n=== Phase 0: Baseline Case 1 ===')

        # Three requests with incrementally growing static prefixes
        case1_states = [
            (_INITIAL_STATIC[:len(_PRE) + len(_CHUNKS_INITIAL) // 3],
             len(_PRE) + len(_CHUNKS_INITIAL) // 3),
            (_INITIAL_STATIC[:len(_PRE) + 2 * len(_CHUNKS_INITIAL) // 3],
             len(_PRE) + 2 * len(_CHUNKS_INITIAL) // 3),
            (_INITIAL_STATIC, len(_INITIAL_STATIC)),
        ]

        for static, j_t in case1_states:
            sequence = static + _DYNAMIC
            state_json = self._make_state_json(sequence=sequence,
                                               j_t=j_t,
                                               k_t=0,
                                               j_epsilon_t=0)
            response = self.asr_manager.run(state_json, max_tokens=32)
            assert response.choices, 'Phase 0: response must have choices'
            content = response.choices[0].message.content
            assert content, 'Phase 0: response content must not be empty'
            print(f'  Case 1 j_t={j_t}: OK')

        status = self.asr_manager.get_status()
        assert status['k'] == 0
        assert status['k_ready'] == 0
        assert not status['is_reconciling']
        assert self.asr_manager.stats.case1_continuations == 3

        # ==============================================================
        # Phase 1: First Reconciliation  (k=0 → k=1)
        # ==============================================================
        print('\n=== Phase 1: First Reconciliation (k=0 → k=1) ===')

        # Record bridge counter baseline before we start
        bridge_count_before_phase1 = self.asr_manager.stats.case2_bridge

        # Trigger reconciliation — mimics what StateManager calls via ROS 2 svc
        request = StartReconciliation.Request()
        request.evicted_state = _X_EPSILON_1
        request.evicted_state_seq_ver = 1
        result = self.asr_manager._start_reconciliation_callback(
            request, StartReconciliation.Response())

        assert result.success, 'Phase 1: service call must succeed'
        assert self.asr_manager._is_reconciling, \
            'Phase 1: is_reconciling must be True after service trigger'
        assert self.asr_manager._k == 0, \
            'Phase 1: active sequence k must not change on service trigger'

        print(f'  Reconciliation triggered: |X_ε_1|={len(_X_EPSILON_1)} chars')

        # Spawn bridge thread — runs concurrent Case 2a requests on r_primary
        # while R2 prefills X_ε_1 in the background
        bridge_thread_1 = self._run_bridge_thread(
            base_static=_INITIAL_STATIC,
            j_epsilon_t_val=len(_X_EPSILON_1),
            k_t_val=1,
        )

        # Wait for background daemon to finish prefill and authorize the swap
        assert self._wait_for_k_ready(1), \
            f'Phase 1: k_ready did not reach 1 within {self._RECONCILIATION_TIMEOUT}s'  # noqa

        bridge_thread_1.join(timeout=60.0)

        # Verify that the bridge thread actually ran at least one request
        assert self.asr_manager.stats.case2_bridge > bridge_count_before_phase1, 'Phase 1: expected at least one Case 2a bridge req during prefill'  # noqa

        status = self.asr_manager.get_status()
        assert status['k_ready'] == 1
        assert status['is_reconciling'], \
            'Phase 1: is_reconciling should still be True until Case 2b fires'

        print(
            f'  k_ready=1 confirmed. case2_bridge={self.asr_manager.stats.case2_bridge}'  # noqa
        )

        # ---- Case 2b: atomic swap ----
        # Build request using X_ε_1 + one bridge chunk as the new static
        # prefix so k_t=1 == k_ready=1 triggers the swap.
        case2b_static_1 = _X_EPSILON_1 + _BRIDGE_CHUNK
        case2b_state_1 = self._make_state_json(
            sequence=case2b_static_1 + _DYNAMIC,
            j_t=len(case2b_static_1),
            k_t=1,
            j_epsilon_t=len(_X_EPSILON_1),
        )
        response_2b_1 = self.asr_manager.run(case2b_state_1, max_tokens=32)
        assert response_2b_1.choices, 'Phase 1 Case 2b: resp must have choices'
        content_2b_1 = response_2b_1.choices[0].message.content
        assert content_2b_1, 'Phase 1 Case 2b: resp content must not be empty'

        # Post-swap assertions
        status = self.asr_manager.get_status()
        assert status['k'] == 1, \
            f"Phase 1: k must be 1 after swap, got {status['k']}"
        assert not status['is_reconciling'], \
            'Phase 1: is_reconciling must be False after swap'
        assert self.asr_manager._r_primary.name == 'R2', \
            'Phase 1: R2 must be r_primary after first swap'
        assert self.asr_manager._r_secondary.name == 'R1', \
            'Phase 1: R1 must be r_secondary after first swap'
        assert self.asr_manager.stats.case2_swap == 1
        assert self.asr_manager.stats.swap_count == 1

        print(
            f'  Swap 1 complete: r_primary={self.asr_manager._r_primary.name}')

        # ==============================================================
        # Phase 2: Second Reconciliation  (k=1 → k=2)
        # ==============================================================
        print('\n=== Phase 2: Second Reconciliation (k=1 → k=2) ===')

        # Record continuations baseline before phase 2 Case 1 requests
        continuations_before_phase2 = self.asr_manager.stats.case1_continuations

        # Two more Case 1 requests confirming inference on new primary (R2)
        phase2_static = case2b_static_1
        for i in range(2):
            phase2_static = phase2_static + _BRIDGE_CHUNK
            j_t = len(phase2_static)
            state_json = self._make_state_json(
                sequence=phase2_static + _DYNAMIC,
                j_t=j_t,
                k_t=1,
                j_epsilon_t=0,
            )
            response = self.asr_manager.run(state_json, max_tokens=32)
            assert response.choices, \
                f'Phase 2 Case 1 [{i}]: response must have choices'
            content = response.choices[0].message.content
            assert content, \
                f'Phase 2 Case 1 [{i}]: response content must not be empty'
            print(f'  Phase 2 Case 1 [{i}] j_t={j_t}: OK')

        assert self.asr_manager.stats.case1_continuations == \
            continuations_before_phase2 + 2, \
            'Phase 2: expected 2 additional Case 1 continuations on R2'

        bridge_count_before_phase2 = self.asr_manager.stats.case2_bridge

        # Trigger second reconciliation — evict all chunks, keep only PRE
        request2 = StartReconciliation.Request()
        request2.evicted_state = _X_EPSILON_2
        request2.evicted_state_seq_ver = 2
        result2 = self.asr_manager._start_reconciliation_callback(
            request2, StartReconciliation.Response())

        assert result2.success, 'Phase 2: service call must succeed'
        assert self.asr_manager._is_reconciling, \
            'Phase 2: is_reconciling must be True after service trigger'
        assert self.asr_manager._k == 1, \
            'Phase 2: active sequence k must not change on service trigger'

        print(f'  Reconciliation triggered: |X_ε_2|={len(_X_EPSILON_2)} chars')

        # Spawn second bridge thread
        bridge_thread_2 = self._run_bridge_thread(
            base_static=phase2_static,
            j_epsilon_t_val=len(_X_EPSILON_2),
            k_t_val=2,
        )

        assert self._wait_for_k_ready(2), \
            f'Phase 2: k_ready did not reach 2 within {self._RECONCILIATION_TIMEOUT}s'  # noqa

        bridge_thread_2.join(timeout=60.0)

        assert self.asr_manager.stats.case2_bridge > bridge_count_before_phase2, 'Phase 2: expected at least one Case 2a bridge req during prefill'  # noqa

        status = self.asr_manager.get_status()
        assert status['k_ready'] == 2
        assert status['is_reconciling'], \
            'Phase 2: is_reconciling should still be True until Case 2b fires'

        print(
            f'  k_ready=2 confirmed. case2_bridge={self.asr_manager.stats.case2_bridge}'  # noqa
        )

        # ---- Case 2b: second atomic swap ----
        case2b_static_2 = _X_EPSILON_2 + _BRIDGE_CHUNK
        case2b_state_2 = self._make_state_json(
            sequence=case2b_static_2 + _DYNAMIC,
            j_t=len(case2b_static_2),
            k_t=2,
            j_epsilon_t=len(_X_EPSILON_2),
        )
        response_2b_2 = self.asr_manager.run(case2b_state_2, max_tokens=32)
        assert response_2b_2.choices, 'Phase 2 Case 2b: resp must have choices'
        content_2b_2 = response_2b_2.choices[0].message.content
        assert content_2b_2, 'Phase 2 Case 2b: resp content must not be empty'

        # Post-swap assertions for the second swap
        status = self.asr_manager.get_status()
        assert status['k'] == 2, \
            f"Phase 2: k must be 2 after swap, got {status['k']}"
        assert not status['is_reconciling'], \
            'Phase 2: is_reconciling must be False after swap'
        assert self.asr_manager._r_primary.name == 'R1', \
            'Phase 2: R1 must be r_primary after second swap (pnt rotated back)'
        assert self.asr_manager._r_secondary.name == 'R2', \
            'Phase 2: R2 must be r_secondary after second swap'
        assert self.asr_manager.stats.case2_swap == 2
        assert self.asr_manager.stats.swap_count == 2

        print(
            f'  Swap 2 complete: r_primary={self.asr_manager._r_primary.name}')
        print('\n=== Two-phase reconciliation test PASSED ===')
        print(f'  Final stats: {self.asr_manager.get_status()["stats"]}')
