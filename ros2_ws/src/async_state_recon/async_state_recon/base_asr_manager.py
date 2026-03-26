import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from rclpy.node import Node
from starlette.requests import Request


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
    sequence: str  # Full X_t content
    j_t: int  # len(static prefix)  — boundary between static and dyn
    k_t: int  # Sequence version of this message
    j_epsilon_t: int  # len(evicted static) at eviction time; 0 otherwise

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
    case2_bridge: int = 0  # k_t > k_ready — bridged on r_primary
    case2_swap: int = 0  # k_t == k_ready — resource swap executed
    case3_straggler_primary: int = 0  # straggler routed to r_primary
    case3_straggler_secondary: int = 0  # straggler routed to r_secondary
    reconciliation_starts: int = 0
    swap_count: int = 0
    total_warmup_s: float = 0.0
    total_catchup_iterations: int = 0


class BaseASRManager(Node):
    """Abstract base class handling common infrastructure for ASR Managers."""

    def __init__(self, **kwargs) -> None:
        super().__init__('asr_manager', **kwargs)

        # 1. Declare and read shared parameters
        self.declare_parameter('client_type', 'vllm')
        self.declare_parameter('model_name', 'default_model')
        self.declare_parameter('http_host', '0.0.0.0')
        self.declare_parameter('http_port', 8000)
        self.declare_parameter('recon_srv_name', 'start_reconciliation')

        self.client_type_str = self.get_parameter('client_type').value
        self.model_name = self.get_parameter('model_name').value
        self.http_host = self.get_parameter('http_host').value
        self.http_port = self.get_parameter('http_port').value
        self.recon_srv_name = self.get_parameter('recon_srv_name').value

        self.stats = ASRStats()

        # 2. Setup HTTP Server
        self._http_server_thread: Optional[threading.Thread] = None
        self._http_stop_event = threading.Event()
        self._start_http_server()

    # --- Abstract Methods (To be implemented by subclasses) ---

    def run(self,
            state_json_str: str,
            max_tokens: int = 512,
            temp: float = 0.7,
            seed: Optional[int] = None,
            stream: bool = False) -> Any:
        raise NotImplementedError("Subclasses must implement run()")

    def get_status(self) -> dict:
        raise NotImplementedError("Subclasses must implement get_status()")

    # --- Shared Utility Methods ---

    def __call__(self, state_json_str: str, **kwargs) -> Any:
        """Callable alias for run() — enables manager(state_json_str, ...)."""
        return self.run(state_json_str, **kwargs)

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
                f'Got: {set(data.keys())}')
        return StateMetadata(
            sequence=str(data['sequence']),
            j_t=int(data['j_t']),
            k_t=int(data['k_t']),
            j_epsilon_t=int(data['j_epsilon_t']),
        )

    # --- Shared HTTP Server Logic ---

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
                    'sequence':
                    sequence,
                    'j_t':
                    asr_meta.get('static_char_len', len(sequence)),
                    'k_t':
                    asr_meta.get('state_seq_ver', 0),
                    'j_epsilon_t':
                    asr_meta.get('evicted_char_length', 0),
                })

                if do_stream:
                    return StreamingResponse(
                        self._generate_sse(state_json_str, max_tokens,
                                           temperature, seed),
                        media_type='text/event-stream',
                    )

                result = self.run(
                    state_json_str,
                    max_tokens=max_tokens,
                    temp=temperature,
                    seed=seed,
                    stream=False,
                )
                return JSONResponse({
                    'id':
                    f'chatcmpl-{int(time.time() * 1000)}',
                    'object':
                    'chat.completion',
                    'created':
                    int(time.time()),
                    'model':
                    self.model_name,
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
                return JSONResponse(status_code=500,
                                    content={'error': str(exc)})

        @app.get('/v1/models')
        async def list_models():
            return {
                'object':
                'list',
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

        self._http_server_thread = threading.Thread(target=_run_server,
                                                    daemon=True,
                                                    name='asr-http')
        self._http_server_thread.start()
        self.get_logger().info(
            f'[HTTP] Server at http://{self.http_host}:{self.http_port}')

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
                state_json_str,
                max_tokens=max_tokens,
                temp=temperature,
                seed=seed,
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    final = {
                        'id':
                        f'chatcmpl-{int(time.time() * 1000)}',
                        'object':
                        'chat.completion.chunk',
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop'
                        }],
                    }
                    yield f'data: {json.dumps(final)}\n\n'
                    yield 'data: [DONE]\n\n'
                    continue
                content = chunk.choices[0].delta.content
                if content is not None:
                    payload = {
                        'id':
                        f'chatcmpl-{int(time.time() * 1000)}',
                        'object':
                        'chat.completion.chunk',
                        'choices': [{
                            'index': 0,
                            'delta': {
                                'content': content
                            },
                            'finish_reason': None,
                        }],
                    }
                    yield f'data: {json.dumps(payload)}\n\n'
        except Exception as exc:
            self.get_logger().error(f'[HTTP] SSE error: {exc}')
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'

    def destroy(self) -> None:
        """Base teardown for shared components."""
        if self._http_server_thread is not None:
            self._http_stop_event.set()
            self._http_server_thread.join(timeout=5.0)
        super().destroy_node()
