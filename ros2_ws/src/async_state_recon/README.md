# async_state_recon

ROS 2 Python package for Asynchronous State Reconciliation (ASR) of LLM
context state.

It runs an `asr_manager` node that:
- serves inference requests,
- accepts reconciliation triggers from upstream state eviction events,
- optionally exposes an OpenAI-compatible HTTP endpoint.

## Dependencies

- ROS 2 (`rclpy`)
- `exodapt_robot_interfaces`
- reachable vLLM endpoint(s)

## Build and source

From your ROS 2 workspace root (`ros2_ws`):

```bash
colcon build --packages-select async_state_recon
source install/setup.bash
```

## Run

### Option 1: launch file (recommended)

Single-resource mode (uses `DummyASRManager`):

```bash
ros2 launch async_state_recon asr_manager_launch.xml \
	r1_url:=http://localhost:8001 \
	r2_url:=""
```

Dual-resource ASR mode (uses `ASRManager`):

```bash
ros2 launch async_state_recon asr_manager_launch.xml \
	r1_url:=http://localhost:8001 \
	r2_url:=http://localhost:8002
```

Common optional launch args:
- `client_type` (default: `vllm`)
- `model_name`
- `catchup_thresh` (default: `512` in launch file)
- `enable_http_server` (default: `false`)
- `http_host` (default: `0.0.0.0`)
- `http_port` (default: `8000`)

### Option 2: direct executable

```bash
ros2 run async_state_recon asr_manager --ros-args -p r1_url:=http://localhost:8001 -p r2_url:=http://localhost:8002
```

## Reconciliation trigger service

The node provides `start_reconciliation` (`exodapt_robot_interfaces/srv/StartReconciliation`).

Example call:

```bash
ros2 service call /start_reconciliation exodapt_robot_interfaces/srv/StartReconciliation "{evicted_state: '...', evicted_state_seq_ver: 1}"
```

## HTTP API

The node exposes:
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

Chat request example:

```bash
curl -s http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "your-model",
		"messages": [{"role": "user", "content": "<full sequence string>"}],
		"max_tokens": 128,
		"temperature": 0.7,
		"asr_metadata": {
			"static_char_len": 1200,
			"state_seq_ver": 3,
			"evicted_char_length": 400
		}
	}'
```

`asr_metadata` maps to internal ASR fields:
- `static_char_len` -> `j_t`
- `state_seq_ver` -> `k_t`
- `evicted_char_length` -> `j_epsilon_t`

## Quick health check

```bash
curl -s http://localhost:8000/health
```
