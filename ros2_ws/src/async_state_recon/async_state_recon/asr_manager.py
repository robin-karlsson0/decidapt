"""
asr_manager_node.py — Production ASR Manager ROS 2 Node.

Implements Algorithm 1 (Asynchronous State Reconciliation) from the CSR paper
as a production-grade ROS 2 node with thread-safe dual-resource KV cache
management and zero-latency inference routing.

Algorithm Overview
------------------
State Variables (per Algorithm 1):
  j          : Char length of current active seq static state (pre + chunks)
  k          : Current active sequence version
  k_ready    : Highest sequence version fully precomputed on r_secondary
  j_ε        : Evicted sequence static state char len (running eviction cursor)
  X_recon    : Reconciliation state — extends current seq w. new evicted chunks
  ΔX_ε       : Catch-up buffer for chunks that arrive during r_secondary warmup
  r_primary  : Resource actively serving inference (KV cache = current sequence)
  r_secondary: Resource precomputing KV cache for the new evicted sequence

Three Inference Routing Cases:
  Case 1 (k_t == k)       : Sequence continuation → serve on r_primary
  Case 2 (k_t > k)        : New sequence version (reconciliation in progress)
    └─ k_t > k_ready      : r_secondary warming up → bridge X_recon, run on r_primary
    └─ k_t == k_ready     : r_secondary ready → atomic swap, run on new primary
  Case 3 (k_t < k)        : Straggler (delayed request for old sequence)
    └─ is_reconciling     : Protect r_secondary → reconstruct and run on r_primary
    └─ else               : r_secondary holds intact old KV cache → route there

State Topic Message Format  (JSON in std_msgs/String on /asr_state):
  {
    "sequence"    : str,   # Full X_t = pre + static_chunks [+ dyn suffix]
    "j_t"         : int,   # len(pre + static_chunks)
    "k_t"         : int,   # Sequence version of this message
    "j_epsilon_t" : int    # len(evicted_prefix) at eviction time; 0 if k_t == k
  }

Inference Request Format  (JSON passed to run() / HTTP POST):
  {
    "sequence"    : str,   # Full X_t including dynamic suffix
    "j_t"         : int,   # Boundary between static and dynamic portions
    "k_t"         : int,   # Sequence version of this request
    "j_epsilon_t" : int    # Evicted static len (populated on first k_t > k call)
  }

ROS 2 Reuse:
  - InferenceClient / create_inference_client  (from inference_server)
  - StreamWrapper                              (from llm_manager)
  - MultiThreadedExecutor + QoS profile setup  (from llm_manager)
  - Optional OpenAI-compatible HTTP server     (mirrored from llm_manager)

Experimental ASRManager Reuse:
  - evict()         — suffix-based chunk eviction (Eq. 4)
  - _prefill()      — 1-token generation to warm KV cache
  - _count_tokens() — tokenizer-backed token counting
"""  # noqa

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import rclpy
import uvicorn
from exodapt_robot_interfaces.srv import StartReconciliation
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from starlette.requests import Request
from std_msgs.msg import String

from .inference_client import (ClientType, InferenceClient,
                               create_inference_client)


class StreamWrapper:
    """Wrapper that ensures client is released after stream consumption.
    
    This wrapper is crucial for streaming responses. Without it, the context
    manager would release the client immediately after returning the stream
    iterator, even though the stream is still being actively consumed.
    
    The wrapper:
    - Keeps the client locked (active_requests > 0) during streaming
    - Releases the client only after stream is fully consumed
    - Handles cleanup via __del__ if stream is abandoned mid-consumption
    
    Args:
        stream_iterator: The raw stream iterator from the inference client
        client_idx: Index of the client being used
        manager: Reference to the LLMManager instance
    """

    def __init__(self, stream_iterator, client_idx, manager):
        self._stream = stream_iterator
        self._client_idx = client_idx
        self._manager = manager
        self._consumed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._stream)
        except StopIteration:
            # Stream is finished, release the client
            if not self._consumed:
                self._release_client()
            raise
        except Exception:
            # Any other exception during streaming - release client and re-raise
            if not self._consumed:
                self._release_client()
            raise

    def _release_client(self):
        """Release the client after stream consumption."""
        if not self._consumed:
            self._consumed = True
            with self._manager.clients_lock:
                self._manager.inf_clients[
                    self._client_idx]['active_requests'] -= 1
            self._manager.get_logger().info(
                f"Stream completed, released client {self._client_idx}")

    def __del__(self):
        """Ensure client is released even if stream is not fully consumed."""
        if not self._consumed:
            self._manager.get_logger().info(
                f"Stream not fully consumed, releasing client {self._client_idx}"  # noqa
            )
            self._release_client()

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateMetadata:
    """Immutable parsed metadata extracted from an X_t state message.

    Mirrors the (j_t, k_t, j_{ε,t}) triple unpacked at the top of
    Algorithm 1 – Inference(X_t).
    """
    sequence: str       # Full X_t content
    j_t: int            # len(static prefix)  — boundary between static and dyn
    k_t: int            # Sequence version of this message
    j_epsilon_t: int    # len(evicted static) at eviction time; 0 otherwise

    @property
    def static_prefix(self) -> str:
        """X_t[:j_t] — pre + static_chunks, never includes the dyn suffix."""
        return self.sequence[:self.j_t]

    @property
    def dynamic_suffix(self) -> str:
        """X_t[j_t:] — the task instruction / query appended by the caller."""
        return self.sequence[self.j_t:]


@dataclass
class ASRStats:
    """Lightweight metrics for observability and paper-experiment logging."""
    total_inference_requests: int = 0
    case1_continuations: int = 0
    case2_bridge: int = 0       # k_t > k_ready — bridged on r_primary
    case2_swap: int = 0         # k_t == k_ready — resource swap executed
    case3_straggler_primary: int = 0    # straggler routed to r_primary
    case3_straggler_secondary: int = 0  # straggler routed to r_secondary
    reconciliation_starts: int = 0
    swap_count: int = 0
    total_warmup_s: float = 0.0
    total_catchup_iterations: int = 0


# ---------------------------------------------------------------------------
# ASRManager
# ---------------------------------------------------------------------------

class ASRManager(Node):
    """Production ROS 2 node — Asynchronous State Reconciliation (Algorithm 1).

    Manages two inference resources (r_primary, r_secondary) to achieve
    zero-latency LLM serving.  When the upstream StateManager evicts context,
    it publishes a state message with an incremented sequence version (k+1).
    This node detects the eviction, starts asynchronous KV-cache warmup of
    the new (shorter) sequence on r_secondary, and continues serving all
    inference requests on r_primary uninterrupted.  Once r_secondary is
    sufficiently caught up, an atomic pointer swap makes it the new primary.

    Parameters
    ----------
    r1_url : str
        URL of the primary vLLM server (default: 'http://localhost:8001')
    r2_url : str
        URL of the secondary vLLM server (default: 'http://localhost:8002')
    client_type : str
        Inference backend — only 'vllm' currently supported (default: 'vllm')
    model_name : str
        Model identifier passed to both InferenceClients
    catchup_thresh : int
        N_catch-up from Algorithm 1 — max |ΔX_ε| (chars) before the catch-up
        loop exits and the swap is authorised (default: 512)
    eviction_ratio : float
        Fraction of chunks to retain after eviction, 0 < ratio < 1
        (default: 0.3, i.e. keep the most recent 30 %)
    state_topic : str
        ROS 2 topic for state update messages (default: '/asr_state')
    enable_http_server : bool
        Expose an OpenAI-compatible HTTP endpoint (default: False)
    http_host : str
        Bind address for the HTTP server (default: '0.0.0.0')
    http_port : int
        Port for the HTTP server (default: 8000)

    Notes
    -----
    Thread-safety model
    ~~~~~~~~~~~~~~~~~~~
    _state_lock (RLock)
        Guards all Algorithm 1 cursors: j, k, k_ready, j_epsilon, x_recon,
        is_reconciling, and the r_primary / r_secondary pointers.
        Held by every call to run() / _route_inference().

    _delta_x_epsilon_lock (Lock)
        Guards only _delta_x_epsilon (the catch-up buffer ΔX_ε).
        May be acquired *while* _state_lock is held (in _route_inference)
        but is NEVER the outer lock — no inverse nesting occurs.

    Lock-ordering rule:  _state_lock > _delta_x_epsilon_lock.
    The reconciliation thread never acquires _state_lock while holding
    _delta_x_epsilon_lock, so deadlock is impossible.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__('asr_manager', **kwargs)

        # ------------------------------------------------------------------
        # ROS 2 parameter declarations
        # ------------------------------------------------------------------
        self.declare_parameter('r1_url', 'http://localhost:8001')
        self.declare_parameter('r2_url', 'http://localhost:8002')
        self.declare_parameter('client_type', 'vllm')
        self.declare_parameter('model_name', 'default_model')
        self.declare_parameter('catchup_thresh', 64)
        self.declare_parameter('http_host', '0.0.0.0')
        self.declare_parameter('http_port', 8000)
        self.declare_parameter('recon_srv_name', 'start_reconciliation')

        r1_url = self.get_parameter('r1_url').value
        r2_url = self.get_parameter('r2_url').value
        client_type_str = self.get_parameter('client_type').value
        self.model_name = self.get_parameter('model_name').value
        self.catchup_thresh: int = self.get_parameter('catchup_thresh').value
        self.http_host: str = self.get_parameter('http_host').value
        self.http_port: int = self.get_parameter('http_port').value
        self.recon_srv_name: str = self.get_parameter('recon_srv_name').value

        # ------------------------------------------------------------------
        # Inference resources
        # ------------------------------------------------------------------
        if client_type_str == 'vllm':
            client_type = ClientType.VLLM
        else:
            raise ValueError(f"Unsupported client_type: {client_type_str!r}")

        self._r1: InferenceClient = create_inference_client(
            client_type, model_name=self.model_name, url=r1_url, name='R1',
        )
        self._r2: InferenceClient = create_inference_client(
            client_type, model_name=self.model_name, url=r2_url, name='R2',
        )
        # Algorithm 1 resource pointers — swapped atomically on reconciliation
        self._r_primary: InferenceClient = self._r1
        self._r_secondary: InferenceClient = self._r2

        # ------------------------------------------------------------------
        # Algorithm 1 state variables
        # ------------------------------------------------------------------
        self._state_lock = threading.RLock()

        # j, k, j_ε are None until the first state update is received
        self._j: int = 0
        self._k: int = 0
        self._k_ready: int = 0
        self._j_epsilon: Optional[int] = None
        self._x_recon: Optional[str] = None

        # ΔX_ε — catch-up buffer, guarded by its own mutex (L_{ΔXε})
        self._delta_x_epsilon: str = ''
        self._delta_x_epsilon_lock = threading.Lock()

        self._is_reconciling: bool = False
        self._reconciliation_thread: Optional[threading.Thread] = None

        # ------------------------------------------------------------------
        # Observability
        # ------------------------------------------------------------------
        self.stats = ASRStats()

        # ------------------------------------------------------------------
        # ROS 2 service
        # ------------------------------------------------------------------
        self._recon_callback_group = MutuallyExclusiveCallbackGroup()
        self._recon_service = self.create_service(
            StartReconciliation,
            self.recon_srv_name,
            self._start_reconciliation_callback,
            callback_group=self._recon_callback_group
        )

        # ------------------------------------------------------------------
        # Optional HTTP server  (OpenAI-compatible, mirrors LLMManager)
        # ------------------------------------------------------------------
        self._http_server_thread: Optional[threading.Thread] = None
        self._http_stop_event = threading.Event()
        self._start_http_server()

        self.get_logger().info(
            f'ASRManager ready | '
            f'r_primary={self._r_primary.name} '
            f'r_secondary={self._r_secondary.name} | '
            f'catchup_thresh={self.catchup_thresh} '
        )

    # ==========================================================================
    # Public API
    # ==========================================================================

    def run(
        self,
        state_json_str: str,
        max_tokens: int = 512,
        temp: float = 0.7,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Any:
        """Main inference entry point — implements Algorithm 1, Inference(X_t).

        Parses X_t from *state_json_str*, determines the routing case, and
        dispatches to the appropriate resource.  The _state_lock is held for the
        entire routing decision so that a concurrent resource swap cannot
        interleave between the case check and the LLM call.

        Args:
            state_json_str : JSON-encoded inference request (see module docstring).
            max_tokens: Maximum tokens to generate.
            temp      : Sampling temperature.
            seed      : Optional RNG seed for reproducibility.
            stream    : If True return a StreamWrapper iterator; else blocking.

        Returns:
            Non-streaming: ChatCompletion response object.
            Streaming    : StreamWrapper yielding delta chunks.

        Raises:
            ValueError         : Malformed or missing keys in state_json_str.
            json.JSONDecodeError: state_json_str is not valid JSON.
        """  # noqa
        meta = self._parse_state(state_json_str)
        self.stats.total_inference_requests += 1

        with self._state_lock:
            return self._route_inference(meta, max_tokens, temp, seed, stream)

    def __call__(self, state_json_str: str, **kwargs) -> Any:
        """Callable alias for run() — enables manager(state_json_str, ...)."""
        return self.run(state_json_str, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name

    def _start_reconciliation_callback(
        self,
        request: StartReconciliation.Request,
        response: StartReconciliation.Response
    ) -> StartReconciliation.Response:
        """
        ROS 2 service callback to trigger Reconcile(X_eps, k_target).

        Extracts X_eps and the target sequence version from the request,
        authorizes the new sequence version (k_t), updates cursors, and
        spawns the background prefill loop.
        """
        k_target = request.evicted_state_seq_ver

        self.get_logger().info(
            f'[SERVICE] StartReconciliation called: k_target={k_target}'
        )

        with self._state_lock:
            # 1. Guard against overlapping reconciliation requests
            if self._is_reconciling:
                self.get_logger().warn(
                    f'[SERVICE] Reconciliation already in progress. '
                    f'Rejecting trigger for k_target={k_target}.'
                )
                response.success = False
                return response

            # 2. Guard against stale or out-of-order service calls
            if k_target <= self._k:
                self.get_logger().warn(
                    f'[SERVICE] Stale reconciliation request: k_target={k_target} '  # noqa
                    f'<= current k={self._k}. Rejecting.'
                )
                response.success = False
                return response

        # 5. Spawn the background Reconcile daemon on r_secondary
        # (This handles the L_{ΔXε} lock and sets is_reconciling = True)
        self._start_reconciliation(request.evicted_state, k_target)

        response.success = True
        return response

    # ==========================================================================
    # Algorithm 1 — Inference procedure
    # ==========================================================================

    def _route_inference(
        self,
        meta: StateMetadata,
        max_tokens: int,
        temp: float,
        seed: Optional[int],
        stream: bool,
    ) -> Any:
        """Core routing logic — Algorithm 1, Inference(X_t).

        MUST be called with self._state_lock held.

        Implements all three cases from the pseudocode verbatim:
          Case 1 — sequence continuation
          Case 2 — reconciliation (k_t > k), with bridge and swap sub-cases
          Case 3 — straggler queries (k_t < k)
        """
        j_t = meta.j_t
        k_t = meta.k_t
        j_epsilon_t = meta.j_epsilon_t
        X_t = meta.sequence

        # Algorithm 1, line 5-7: initialise j_ε on first reconciliation call
        if self._j_epsilon is None and k_t > self._k_ready:
            self._j_epsilon = j_epsilon_t

        # ------------------------------------------------------------------
        # Case 1: Sequence continuation  (k_t == k)
        # ------------------------------------------------------------------
        if k_t == self._k:
            # Update reconciliation snapshot whenever static portion grows
            if j_t > self._j:
                self._x_recon = X_t[:j_t]
                self._j = j_t

            self.stats.case1_continuations += 1
            self.get_logger().debug(
                f'[INFER-C1] k={k_t} j={j_t} -> {self._r_primary.name}'
            )
            return self._llm(
                X_t, self._r_primary, max_tokens, temp, seed, stream)

        # ------------------------------------------------------------------
        # Case 2: New sequence version — reconciliation phase  (k_t > k)
        # ------------------------------------------------------------------
        if k_t > self._k:

            if k_t > self._k_ready:
                # ---------------------------------------------------------
                # Case 2a: r_secondary still warming up — bridge state and
                #          run on r_primary (Algorithm 1, lines 22–32)
                # ---------------------------------------------------------
                # Extract new chunks from the evicted state since last cursor
                delta_x_t = X_t[self._j_epsilon:j_t]

                # Extend reconciliation state with new evicted chunks
                self._x_recon += delta_x_t

                # Append to catch-up buffer under its dedicated mutex
                with self._delta_x_epsilon_lock:
                    self._delta_x_epsilon += delta_x_t

                # Construct bridged query: extended static state + current dyn
                x_dyn = X_t[j_t:]
                x_prime = self._x_recon + x_dyn

                # Advance eviction cursor
                self._j_epsilon = j_t

                self.stats.case2_bridge += 1
                self.get_logger().debug(
                    f'[INFER-C2a] bridge k_t={k_t} k_ready={self._k_ready} '
                    f'delta={len(delta_x_t)}c -> {self._r_primary.name}'
                )
                return self._llm(
                    x_prime, self._r_primary, max_tokens, temp, seed, stream)

            elif k_t == self._k_ready:
                # ---------------------------------------------------------
                # Case 2b: r_secondary finished precomputation — swap
                #          (Algorithm 1, lines 33–39)
                # ---------------------------------------------------------
                self._r_primary, self._r_secondary = (
                    self._r_secondary, self._r_primary
                )
                self._is_reconciling = False
                self._x_recon = X_t[:j_t]      # Snapshot for future stragglers
                self._j = j_t
                self._k = k_t
                self._j_epsilon = None           # Reset eviction cursor

                self.stats.case2_swap += 1
                self.stats.swap_count += 1
                self.get_logger().info(
                    f'[INFER-C2b] SWAP k={k_t} '
                    f'r_primary now={self._r_primary.name}'
                )
                return self._llm(
                    X_t, self._r_primary, max_tokens, temp, seed, stream)

        # ------------------------------------------------------------------
        # Case 3: Straggler queries  (k_t < k)
        # ------------------------------------------------------------------
        if k_t < self._k:
            if self._is_reconciling:
                # ---------------------------------------------------------
                # Case 3a: Protect r_secondary from significantly delayed
                #          requests during reconciliation  (lines 42–46)
                # ---------------------------------------------------------
                x_dyn = X_t[j_t:]
                x_prime = (self._x_recon or '') + x_dyn

                self.stats.case3_straggler_primary += 1
                self.get_logger().debug(
                    f'[INFER-C3a] straggler(recon) k_t={k_t} k={self._k} '
                    f'-> {self._r_primary.name}'
                )
                return self._llm(
                    x_prime, self._r_primary, max_tokens, temp, seed, stream)
            else:
                # ---------------------------------------------------------
                # Case 3b: r_secondary holds the intact KV cache for the
                #          old sequence  (line 47)
                # ---------------------------------------------------------
                self.stats.case3_straggler_secondary += 1
                self.get_logger().debug(
                    f'[INFER-C3b] straggler k_t={k_t} k={self._k} '
                    f'-> {self._r_secondary.name}'
                )
                return self._llm(
                    X_t, self._r_secondary, max_tokens, temp, seed, stream)

        # Should never be reached under correct operation
        self.get_logger().error(
            f'[INFER] Unhandled routing case: k_t={k_t} k={self._k} '
            f'k_ready={self._k_ready}. Falling back to r_primary.'
        )
        return self._llm(X_t, self._r_primary, max_tokens, temp, seed, stream)

    # ==========================================================================
    # Algorithm 1 — Reconcile procedure
    # ==========================================================================

    def _start_reconciliation(self, x_epsilon: str, k_target: int) -> None:
        """Trigger asynchronous reconciliation for the new evicted sequence.

        Called from _state_update_callback when k_t > k is detected (eviction
        event).  Spawns a daemon thread that runs _reconcile() in the background
        while r_primary continues serving inference requests uninterrupted.

        Guards against duplicate triggers: if reconciliation is already running
        (e.g. rapid consecutive evictions), the call is a no-op with a warning.

        Args:
            x_epsilon : Initial evicted state X_ε = sequence[:j_epsilon_t].
            k_target  : New sequence version (k+1) to precompute on r_secondary.
        """
        with self._state_lock:
            if self._is_reconciling:
                self.get_logger().warn(
                    f'[RECON] Already reconciling; ignoring duplicate trigger '
                    f'for k_target={k_target}.'
                )
                return
            # Set flag *before* spawning the thread to eliminate the race
            # window between thread creation and the first is_reconciling check
            # in _route_inference (Case 3).
            self._is_reconciling = True
            self.stats.reconciliation_starts += 1

        self._reconciliation_thread = threading.Thread(
            target=self._reconcile,
            args=(x_epsilon, k_target),
            daemon=True,
            name=f'asr-reconcile-k{k_target}',
        )
        self._reconciliation_thread.start()

        self.get_logger().info(
            f'[RECON] Started reconciliation thread: '
            f'k_target={k_target} |X_ε|={len(x_epsilon)}c '
            f'r_secondary={self._r_secondary.name}'
        )

    def _reconcile(self, x_epsilon: str, k_target: int) -> None:
        """Background implementation of Algorithm 1 – Reconcile(X_ε, k_target).

        Runs in a daemon thread spawned by _start_reconciliation().

        Procedure (mirrors Algorithm 1 exactly):
          1. Lock r_secondary from straggler access  (is_reconciling already True)
          2. Clear ΔX_ε catch-up buffer  (fresh start)
          3. Warmup : full KV-cache prefill of X_ε on r_secondary (slow, ~seconds)
          4. Catch-up loop : iteratively prefill accumulated ΔX_ε until
             |ΔX_ε| ≤ N_catch-up  (fast — incremental prefill only)
          5. Final drain : prefill any remaining ΔX_ε below the threshold
          6. Authorise swap : set k_ready = k_target

        The Case 2b handler in _route_inference performs the actual pointer swap
        on the next inference request that arrives with k_t == k_ready.

        NOTE: The pseudocode comment "Remainder is computed at next task
              inference" refers to the final drain in step 5 — any delta
              that arrives between step 5 and the swap is handled by the
              bridging logic in Case 2a.

        Args:
            x_epsilon : Initial evicted state (the shorter post-eviction context).
            k_target  : The sequence version being precomputed.
        """  # noqa
        try:
            # ---- Step 1: Clear catch-up buffer ----
            # Algorithm 1 line: "Lock(L_{ΔXε}); ΔX_ε ← ∅; Unlock(L_{ΔXε})"
            with self._delta_x_epsilon_lock:
                self._delta_x_epsilon = ''

            # ---- Step 2: Warmup — full prefill of evicted state ----
            # Algorithm 1 line: "LLM(X_ε, r_secondary)"
            self.get_logger().info(
                f'[RECON] Warmup: prefilling |X_ε|={len(x_epsilon)}c '
                f'on {self._r_secondary.name}'
            )
            t_warmup = time.monotonic()
            self._prefill(x_epsilon, self._r_secondary)
            warmup_s = time.monotonic() - t_warmup
            self.stats.total_warmup_s += warmup_s
            self.get_logger().info(f'[RECON] Warmup done in {warmup_s:.3f}s')

            # Local mirror of the state fully precomputed on r_secondary
            x_epsilon_local = x_epsilon

            # ---- Step 3: Catch-up loop ----
            # Algorithm 1: "while |ΔX_ε| > N_catch-up do ..."
            catchup_iter = 0
            while True:
                with self._delta_x_epsilon_lock:
                    pending_len = len(self._delta_x_epsilon)

                if pending_len <= self.catchup_thresh:
                    break

                catchup_iter += 1
                self.stats.total_catchup_iterations += 1

                # Atomically snapshot and clear the buffer so new chunks can
                # accumulate while we prefill this batch.
                # Algorithm 1: "ΔX'_ε ← ΔX_ε; ΔX_ε ← ∅"
                with self._delta_x_epsilon_lock:
                    delta_snapshot = self._delta_x_epsilon
                    self._delta_x_epsilon = ''

                x_epsilon_local = x_epsilon_local + delta_snapshot

                t0 = time.monotonic()
                self.get_logger().info(
                    f'[RECON] Catch-up iter={catchup_iter}: '
                    f'delta={len(delta_snapshot)}c '
                    f'total_x_eps={len(x_epsilon_local)}c'
                )
                self._prefill(x_epsilon_local, self._r_secondary)
                self.get_logger().debug(
                    f'[RECON] Catch-up iter={catchup_iter} done '
                    f'in {time.monotonic() - t0:.3f}s'
                )

            # --- Step 4: Final drain — flush any remainder below threshold ---
            # Algorithm 1 note: "Remainder computed at next task inference"
            # We eagerly drain it here so the swap is as fresh as possible.
            with self._delta_x_epsilon_lock:
                remainder = self._delta_x_epsilon
                self._delta_x_epsilon = ''

            if remainder:
                x_epsilon_local = x_epsilon_local + remainder
                self.get_logger().info(
                    f'[RECON] Final drain: {len(remainder)}c remaining'
                )
                t0 = time.monotonic()
                self._prefill(x_epsilon_local, self._r_secondary)
                self.get_logger().debug(
                    f'[RECON] Final drain done in {time.monotonic() - t0:.3f}s'
                )

            # ---- Step 5: Authorise resource swap ----
            # Algorithm 1: "k_ready ← k_target"
            # The actual pointer swap happens in _route_inference Case 2b on
            # the next inference request with k_t == k_ready.
            with self._state_lock:
                self._k_ready = k_target

            self.get_logger().info(
                f'[RECON] Complete — k_ready={k_target} '
                f'(swap authorised for next Case-2b inference) '
                f'warmup={warmup_s:.2f}s iters={catchup_iter}'
            )

        except Exception as exc:
            self.get_logger().error(
                f'[RECON] Reconciliation failed for k_target={k_target}: {exc}',
                exc_info=True,
            )
            # Release the reconciliation lock so future evictions can proceed
            with self._state_lock:
                self._is_reconciling = False
            raise

    # ==========================================================================
    # ROS 2 state subscription callback
    # ==========================================================================

    def _state_update_callback(self, msg: String) -> None:
        """ROS 2 callback for the /asr_state topic.

        Serves two purposes:
          1. Maintain the j / k cursors for the inference routing logic.
          2. Detect eviction events (k_t > k) and trigger reconciliation.

        Routing:
          k_t == k  : Static portion grew — update j cursor and X_recon snapshot.
          k_t > k   : Eviction event — reconstruct X_ε and start reconciliation.
          k_t < k   : Stale / out-of-order update — log and discard.

        Args:
            msg: std_msgs/String carrying a JSON-encoded StateMetadata dict.
        """  # noqa
        try:
            meta = self._parse_state(msg.data)
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f'[STATE] Malformed state message: {exc}')
            return

        with self._state_lock:
            current_k = self._k

        # ---- First-ever state message ----
        if current_k is None:
            with self._state_lock:
                self._k = meta.k_t
                self._j = meta.j_t
                self._j_epsilon = meta.j_epsilon_t or None
                self._x_recon = meta.static_prefix
            self.get_logger().info(
                f'[STATE] Initialised: k={meta.k_t} j={meta.j_t}'
            )
            return

        # ---- Sequence continuation ----
        if meta.k_t == current_k:
            with self._state_lock:
                if meta.j_t > (self._j or 0):
                    self._x_recon = meta.static_prefix
                    self._j = meta.j_t
            self.get_logger().debug(
                f'[STATE] Continuation: k={meta.k_t} j={meta.j_t}'
            )
            return

        # ---- Eviction event: new sequence version ----
        if meta.k_t > current_k:
            # X_eps is the initial post-eviction state: everything up to
            # j_epsilon_t
            x_epsilon = meta.sequence[:meta.j_epsilon_t]
            self.get_logger().info(
                f'[STATE] Eviction k {current_k}→{meta.k_t} '
                f'j_ε={meta.j_epsilon_t} |X_ε|={len(x_epsilon)}c'
            )
            # Update k cursor before starting reconciliation so that concurrent
            # Case-2 inference calls see the new k immediately.
            with self._state_lock:
                self._k = meta.k_t
                self._j = meta.j_t
                self._j_epsilon = meta.j_epsilon_t
                if meta.j_t > 0:
                    self._x_recon = meta.static_prefix

            self._start_reconciliation(x_epsilon, meta.k_t)
            return

        # ---- Stale / out-of-order update ----
        self.get_logger().warn(
            f'[STATE] Stale update: k_t={meta.k_t} < current k={current_k}; '
            f'discarded.'
        )

    # ==========================================================================
    # Helpers — reused from experimental ASRManager
    # ==========================================================================

    def evict(self, chunks: str) -> str:
        """Keep only the most recent suffix of *chunks* (Equation 4, CSR paper).

        Reused verbatim from experimental ASRManager.evict().
        Called by the upstream StateManager when the context window fills;
        exposed here so callers can use the same eviction policy in-process.

        Args:
            chunks: Full chunk history as a string.

        Returns:
            Trailing suffix of length floor(len(chunks) * eviction_ratio).
        """
        keep_len = int(len(chunks) * self.eviction_ratio)
        evicted = chunks[-keep_len:] if keep_len > 0 else ''
        self.get_logger().info(
            f'[EVICT] {len(chunks)}c -> {len(evicted)}c '
            f'(ratio={self.eviction_ratio:.2f})'
        )
        return evicted

    def _prefill(self, prompt: str, client: InferenceClient) -> None:
        """Warm the KV cache of *client* without generating real output.

        Forces a full prefill pass by requesting exactly one token.
        Reused from experimental ASRManager.prefill().

        Args:
            prompt: The full prompt string whose KV cache is to be populated.
            client: Target InferenceClient resource.
        """
        client.run(prompt, max_tokens=1, stream=False)

    def _count_tokens(self, text: str) -> int:
        """Return the token count of *text* using the node's tokenizer.

        Falls back to character count if no tokenizer is set.
        Reused from experimental ASRManager._count_tokens().

        Args:
            text: Input string.

        Returns:
            Token count (int).
        """
        tokenizer = getattr(self, 'tokenizer', None)
        if not text:
            return 0
        if tokenizer is None:
            return len(text)
        return len(tokenizer.encode(text, add_special_tokens=False))

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    @staticmethod
    def _parse_state(state_json_str: str) -> StateMetadata:
        """Parse and validate a JSON state string into a StateMetadata object.

        Expected keys: 'sequence', 'j_t', 'k_t', 'j_epsilon_t'.

        Raises:
            ValueError         : One or more required keys are absent.
            json.JSONDecodeError: state_json_str is not valid JSON.
        """
        data = json.loads(state_json_str)
        required = {'sequence', 'j_t', 'k_t', 'j_epsilon_t'}
        missing = required - data.keys()
        if missing:
            raise ValueError(
                f'State message missing required keys: {missing}. '
                f'Got: {set(data.keys())}'
            )
        return StateMetadata(
            sequence=str(data['sequence']),
            j_t=int(data['j_t']),
            k_t=int(data['k_t']),
            j_epsilon_t=int(data['j_epsilon_t']),
        )

    def _llm(
        self,
        prompt: str,
        client: InferenceClient,
        max_tokens: int,
        temp: float,
        seed: Optional[int],
        stream: bool,
    ) -> Any:
        """Dispatch an inference call to a specific resource with timing.

        Args:
            prompt   : Full prompt string (pre + chunks + dyn or bridged equiv).
            client   : Target InferenceClient (r_primary or r_secondary).
            max_tokens, temp, seed, stream: Standard inference parameters.

        Returns:
            ChatCompletion response, or StreamWrapper if stream=True.
        """
        t0 = time.monotonic()
        response = client.run(
            prompt,
            max_tokens=max_tokens,
            temp=temp,
            seed=seed,
            stream=stream,
        )
        if not stream:
            self.get_logger().debug(
                f'[LLM] {client.name} {time.monotonic() - t0:.3f}s '
                f'prompt={len(prompt)}c'
            )
        return response

    def get_status(self) -> dict:
        """Return a snapshot of all Algorithm 1 state variables for debugging.

        Returns:
            dict with keys: j, k, k_ready, j_epsilon, is_reconciling,
            r_primary, r_secondary, x_recon_len, delta_x_epsilon_len, stats.
        """
        with self._state_lock:
            with self._delta_x_epsilon_lock:
                delta_len = len(self._delta_x_epsilon)
            return {
                'j': self._j,
                'k': self._k,
                'k_ready': self._k_ready,
                'j_epsilon': self._j_epsilon,
                'is_reconciling': self._is_reconciling,
                'r_primary': self._r_primary.name,
                'r_secondary': self._r_secondary.name,
                'x_recon_len': len(self._x_recon) if self._x_recon else 0,
                'delta_x_epsilon_len': delta_len,
                'stats': {
                    'total_requests': self.stats.total_inference_requests,
                    'case1': self.stats.case1_continuations,
                    'case2_bridge': self.stats.case2_bridge,
                    'case2_swap': self.stats.case2_swap,
                    'case3_primary': self.stats.case3_straggler_primary,
                    'case3_secondary': self.stats.case3_straggler_secondary,
                    'reconciliation_starts': self.stats.reconciliation_starts,
                    'swap_count': self.stats.swap_count,
                    'total_warmup_s': round(self.stats.total_warmup_s, 3),
                    'total_catchup_iters': self.stats.total_catchup_iterations,
                },
            }

    # ==========================================================================
    # Optional HTTP server — mirrors LLMManager._start_http_server()
    # ==========================================================================

    def _start_http_server(self) -> None:
        """Start an OpenAI-compatible FastAPI server in a background daemon thread.

        Exposes:
          POST /v1/chat/completions — chat completions (streaming + non-streaming)
          GET  /v1/models           — model listing
          GET  /health              — Algorithm 1 state + stats snapshot
          GET  /                    — API index

        The last message in the 'messages' array must carry the JSON state_json_str
        as its 'content' field (see module-level docstring for format).

        Design mirrors LLMManager._start_http_server() for drop-in compatibility.
        """  # noqa
        # try:
        #     from fastapi import FastAPI
        #     from fastapi.responses import JSONResponse, StreamingResponse
        #     from starlette.requests import Request
        # except ImportError:
        #     self.get_logger().error(
        #         '[HTTP] FastAPI/uvicorn not installed. '
        #         'Run: pip install fastapi uvicorn'
        #     )
        #     return

        app = FastAPI(
            title='ASRManager OpenAI-Compatible API',
            description=(
                'Zero-latency LLM serving via Asynchronous State Reconciliation'
            ),
            version='1.0.0',
        )

        @app.post('/v1/chat/completions')
        async def chat_completions(request: Request):
            """OpenAI-compatible completions."""
            try:
                body = await request.json()
                messages = body.get('messages', [])
                sequence = messages[-1]['content'] if messages else ''
                max_tokens = body.get('max_tokens', 1024)
                temperature = body.get('temperature', 0.7)
                seed = body.get('seed', None)
                do_stream = body.get('stream', False)

                # Reconstruct the StateMetadata JSON from
                # extra_body["asr_metadata"]
                # The upstream client (execute_callback_vllm) tunnels ASR state
                # fields here rather than encoding them into the message content
                asr_meta = body.get('asr_metadata', {})
                state_json_str = json.dumps({
                    'sequence': sequence,
                    'j_t': asr_meta.get('static_char_len', len(sequence)),
                    'k_t': asr_meta.get('state_seq_ver', 0),
                    'j_epsilon_t': asr_meta.get('evicted_char_length', 0),
                })

                if do_stream:
                    return StreamingResponse(
                        self._generate_sse(
                            state_json_str, max_tokens, temperature, seed),
                        media_type='text/event-stream',
                    )

                result = self.run(
                    state_json_str, max_tokens=max_tokens,
                    temp=temperature, seed=seed, stream=False,
                )
                return JSONResponse({
                    'id': f'chatcmpl-{int(time.time() * 1000)}',
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': self.model_name,
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': result.choices[0].message.content,
                        },
                        'finish_reason': 'stop',
                    }],
                    'usage': {
                        'prompt_tokens': result.usage.prompt_tokens,
                        'completion_tokens': result.usage.completion_tokens,
                        'total_tokens': result.usage.total_tokens,
                    },
                })
            except Exception as exc:
                self.get_logger().error(f'[HTTP] chat_completions: {exc}')
                return JSONResponse(
                    status_code=500, content={'error': str(exc)})

        @app.get('/v1/models')
        async def list_models():
            return {
                'object': 'list',
                'data': [{
                    'id': self.model_name,
                    'object': 'model',
                    'created': int(time.time()),
                    'owned_by': 'asr-manager',
                }],
            }

        @app.get('/health')
        async def health():
            return {'status': 'healthy', **self.get_status()}

        @app.get('/')
        async def root():
            return {
                'name': 'ASRManager API',
                'endpoints': {
                    'chat': '/v1/chat/completions',
                    'models': '/v1/models',
                    'health': '/health',
                },
            }

        def _run_server():
            config = uvicorn.Config(
                app,
                host=self.http_host,
                port=self.http_port,
                log_level='warning',
            )
            server = uvicorn.Server(config)
            asyncio.run(server.serve())

        self._http_server_thread = threading.Thread(
            target=_run_server, daemon=True, name='asr-http'
        )
        self._http_server_thread.start()
        self.get_logger().info(
            f'[HTTP] Server at http://{self.http_host}:{self.http_port}'
        )

    async def _generate_sse(
        self,
        state_json_str: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
    ):
        """Yield OpenAI-compatible SSE chunks for streaming HTTP responses.

        Mirrors LLMManager._generate_streaming_response().
        """
        try:
            stream = self.run(
                state_json_str, max_tokens=max_tokens,
                temp=temperature, seed=seed, stream=True,
            )
            for chunk in stream:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    final = {
                        'id': f'chatcmpl-{int(time.time() * 1000)}',
                        'object': 'chat.completion.chunk',
                        'choices': [
                            {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
                        ],
                    }
                    yield f'data: {json.dumps(final)}\n\n'
                    yield 'data: [DONE]\n\n'
                    continue
                content = chunk.choices[0].delta.content
                if content is not None:
                    payload = {
                        'id': f'chatcmpl-{int(time.time() * 1000)}',
                        'object': 'chat.completion.chunk',
                        'choices': [{
                            'index': 0,
                            'delta': {'content': content},
                            'finish_reason': None,
                        }],
                    }
                    yield f'data: {json.dumps(payload)}\n\n'
        except Exception as exc:
            self.get_logger().error(f'[HTTP] SSE error: {exc}')
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    def destroy(self) -> None:
        """Graceful shutdown: drain reconciliation thread then destroy node."""
        if self._reconciliation_thread is not None and \
                self._reconciliation_thread.is_alive():
            self.get_logger().info(
                '[DESTROY] Waiting for reconciliation thread...')
            self._reconciliation_thread.join(timeout=30.0)
            if self._reconciliation_thread.is_alive():
                self.get_logger().warn(
                    '[DESTROY] Reconciliation thread did not finish.')

        if self._http_server_thread is not None:
            self._http_stop_event.set()
            self._http_server_thread.join(timeout=5.0)

        super().destroy_node()
        self.get_logger().info('[DESTROY] ASRManager shut down.')


# =============================================================================
# ROS 2 entry point
# =============================================================================

def main(args=None) -> None:
    """Launch the ASRManager ROS 2 node with a MultiThreadedExecutor.

    The MultiThreadedExecutor is required so that the state subscription
    callback and any concurrent inference calls can run simultaneously
    without blocking one another.
    """
    rclpy.init(args=args)
    node = ASRManager()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
