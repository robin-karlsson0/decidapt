#!/usr/bin/env python3
"""
asr_manager_perf_bench.py — Performance benchmark for the production ASRManager.

Adapts the ASR TTFT Experiment notebook to the production ROS 2 ASRManager.

Compares TTFT (Time-To-First-Token) between:
  - Baseline: Direct vLLM inference with synchronous manual eviction
  - ASRManager: Production ROS 2 node with asynchronous state reconciliation

Key differences from the notebook (experimental) ASRManager:
  - Synchronous API: node.run() is blocking, no asyncio required.
  - State format: JSON with 'sequence', 'j_t', 'k_t', 'j_epsilon_t' keys.
    Position fields (j_t, j_epsilon_t) are character lengths, not token counts.
  - Reconciliation: triggered via
    ``node._start_reconciliation(x_epsilon, k_target)``.
  - Stats: ASRStats dataclass attributes (not a dict),
    e.g. ``node.stats.swap_count``.
  - Reconciliation state: ``node._is_reconciling``,
    ``node._reconciliation_thread``.
  - Resource pointers: ``node._r_primary.name``,
    ``node._r_secondary.name``.

Usage (after sourcing ros2_ws/install/setup.bash):
    python -m async_state_recon.asr_manager_perf_bench \\
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --r1-url http://localhost:8000 \\
        --r2-url http://localhost:8001 \\
        --num-runs 5

Algorithm 1 state format passed to node.run():
    {
        "sequence"    : str,   # Full X_t = pre + chunks + dyn
        "j_t"         : int,   # len(pre + chunks)  [char length]
        "k_t"         : int,   # Sequence version (incremented on eviction)
        "j_epsilon_t" : int    # len(x_epsilon) at eviction time; 0 otherwise
    }
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib  # noqa: E402

matplotlib.use('Agg')  # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

try:
    from async_state_recon.asr_manager import ASRManager
    from async_state_recon.base_asr_manager import ASRStats
    from async_state_recon.inference_client import (ClientType,
                                                    InferenceClient,
                                                    create_inference_client)
except ImportError:
    from asr_manager import ASRManager
    from base_asr_manager import ASRStats
    from inference_client import (ClientType, InferenceClient,
                                  create_inference_client)

# =============================================================================
# Default configuration (mirrors the notebook)
# =============================================================================

# ASRManager parameters
MEM_THRESH: int = 90_000  # 90_000  # tokens — triggers KV cache eviction
CATCHUP_THRESH: int = 100  # chars — catch-up buffer threshold (Alg. 1)
EVICTION_RATIO: float = 0.5  # fraction of chars to keep after eviction

# Experiment parameters
CHUNK_INCREMENT: int = 100  # tokens added per query
PRE_TOKENS: int = 5_000  # static prefix size (tokens)
DYN_TOKENS: int = 1  # dynamic suffix size (tokens)
NUM_RUNS: int = 1  # 5  # statistical sample repetitions
MAX_TOKENS_OUTPUT: int = 1  # generate only 1 token — isolates TTFT
NUM_EVICTIONS: int = 2  # eviction cycles to observe per run
QUERIES_POST_FINAL_EVICTION: int = int(5_000 / CHUNK_INCREMENT)

# Model
MODEL_NAME: str = 'Qwen/Qwen3.5-35B-A3B'

# Servers
NAME_1: str = 'R1'
NAME_2: str = 'R2'
SERVER_1_URL: str = 'http://localhost:8000'
SERVER_2_URL: str = 'http://localhost:8001'

# Internal HTTP port for the ASRManager node (must not collide with vLLM ports)
ASR_HTTP_PORT: int = 19876

# Output
RESULTS_DIR: Path = Path('asr_exp/results')

# =============================================================================
# Text / token utilities  (identical to notebook helpers)
# =============================================================================


def generate_chunk_with_exact_tokens(
        tokenizer,
        target_tokens: int,
        base_text: str = None,  # unused; kept for API compatibility
) -> str:
    """Generate random text with exactly *target_tokens* tokens.

    Random words prevent KV cache reuse between chunks.
    """
    if target_tokens == 0:
        return ''

    random_words = []
    for _ in range(target_tokens * 3):
        word_length = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        random_words.append(word)

    random_text = ' '.join(random_words)
    tokens = tokenizer.encode(random_text, add_special_tokens=False)
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def count_tokens(tokenizer, text: str) -> int:
    """Return the number of tokens in *text*."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def evict_chunks(chunks: str, eviction_ratio: float, tokenizer) -> str:
    """Keep only the most recent *eviction_ratio* fraction of *chunks*.

    Eviction is character-based (i.e. keeps the last ``eviction_ratio``
    fraction of the string), which mirrors the notebook implementation and
    is consistent with the production ASRManager's char-length positions.
    """
    keep_len = int(len(chunks) * eviction_ratio)
    evicted = chunks[-keep_len:] if keep_len > 0 else ''

    original_tokens = count_tokens(tokenizer, chunks)
    evicted_tokens = count_tokens(tokenizer, evicted)
    print(f'    [EVICTION] {original_tokens:,} tokens '
          f'→ {evicted_tokens:,} tokens '
          f'(kept {eviction_ratio:.0%})')
    return evicted


# =============================================================================
# Production ASRManager utilities
# =============================================================================


def build_state_json(
    sequence: str,
    j_t: int,
    k_t: int,
    j_epsilon_t: int,
) -> str:
    """Serialise an inference request for the production ASRManager.

    Args:
        sequence    : Full X_t string (pre + chunks + dyn).
        j_t         : Character length of the static portion (pre + chunks).
        k_t         : Sequence version counter (incremented on each eviction).
        j_epsilon_t : Character length of x_epsilon at eviction time; 0 if
                      no eviction has occurred for the current k_t.
    """
    return json.dumps({
        'sequence': sequence,
        'j_t': j_t,
        'k_t': k_t,
        'j_epsilon_t': j_epsilon_t,
    })


def reset_asr_node_state(node: ASRManager) -> None:
    """Reset all Algorithm 1 cursors for a fresh benchmark run.

    Waits for any in-progress reconciliation thread to finish, then resets
    all internal state variables and resource pointers to their initial values.
    This allows the same ROS 2 node to be reused across multiple runs without
    the overhead of creating a new node per run.
    """
    # Drain any running reconciliation before touching shared state.
    if node._reconciliation_thread and node._reconciliation_thread.is_alive():
        print('  [RESET] Waiting for reconciliation thread to finish...')
        node._reconciliation_thread.join(timeout=60.0)

    with node._state_lock:
        node._j = 0
        node._k = 0
        node._k_ready = 0
        node._j_epsilon = None
        node._x_recon = None
        node._is_reconciling = False
        node._reconciliation_thread = None
        node._r_primary = node._r1
        node._r_secondary = node._r2

        with node._delta_x_epsilon_lock:
            node._delta_x_epsilon = ''

    node.stats = ASRStats()


# =============================================================================
# Baseline experiment
# =============================================================================


def run_baseline_experiment(
    run_id: int,
    inference_client: InferenceClient,
    tokenizer,
    pre_tokens: int,
    dyn_tokens: int,
    chunk_increment: int,
    mem_thresh: int,
    eviction_ratio: float,
    num_evictions: int,
    queries_post_final_eviction: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Run the baseline experiment: direct inference with manual eviction.

    Replicates the notebook baseline exactly, adapted to the synchronous
    (non-async) inference client API of the production stack.

    Eviction is triggered when the static portion of the context
    (pre + chunks) first exceeds *mem_thresh* tokens by more than
    4 × chunk_increment tokens (a small buffer to avoid boundary jitter).

    Args:
        run_id:                       Experiment repetition index (1-based).
        inference_client:             vLLM InferenceClient instance.
        tokenizer:                    HuggingFace tokenizer.
        pre_tokens:                   Static prefix size in tokens.
        dyn_tokens:                   Dynamic suffix size in tokens.
        chunk_increment:              Tokens appended per query.
        mem_thresh:                   Eviction threshold in tokens.
        eviction_ratio:               Fraction of chars to keep after eviction.
        num_evictions:                Number of eviction cycles before stopping.
        queries_post_final_eviction:  Extra queries after the last eviction.
        num_runs:                     Total runs (for progress display only).

    Returns:
        Result dict with keys: run_id, experiment, config, queries,
        eviction_at_queries, eviction_count.
    """
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'BASELINE EXPERIMENT — Run {run_id}/{num_runs}')
    print(f'{sep}\n')

    pre = generate_chunk_with_exact_tokens(tokenizer, pre_tokens)
    dyn = generate_chunk_with_exact_tokens(tokenizer, dyn_tokens)

    chunks = ''
    query_count = 0
    queries: List[Dict[str, Any]] = []
    eviction_count = 0
    eviction_at_queries: List[int] = []
    cumulative_tokens = 0

    while True:
        query_count += 1

        # Stop after finishing the required post-eviction observation window.
        if eviction_count >= num_evictions:
            last = eviction_at_queries[-1]
            if query_count > last + queries_post_final_eviction:
                break

        # Eviction check — identical to notebook: trigger slightly before the
        # hard threshold to give a reproducible eviction point.
        static_tokens = count_tokens(tokenizer, pre + chunks)
        if (static_tokens - 4 * chunk_increment >= mem_thresh
                and eviction_count < num_evictions):
            eviction_count += 1
            print(f'  Query {query_count}: '
                  f'EVICTION #{eviction_count} TRIGGERED '
                  f'(static={static_tokens:,} tokens)')
            chunks = evict_chunks(chunks, eviction_ratio, tokenizer)
            eviction_at_queries.append(query_count)

        # Append a new random chunk.
        chunks += generate_chunk_with_exact_tokens(tokenizer, chunk_increment)
        chunk_tokens_after = count_tokens(tokenizer, chunks)
        cumulative_tokens += chunk_increment

        prompt = pre + chunks + dyn
        total_tokens = count_tokens(tokenizer, prompt)

        ttft_start = time.time()
        inference_client.run(
            prompt,
            max_tokens=MAX_TOKENS_OUTPUT,
            temp=0.7,
            seed=42,
            stream=False,
        )
        ttft = time.time() - ttft_start

        if eviction_count == 0:
            evict_marker = '[PRE-EVICT ] '
        else:
            evict_marker = f'[POST-EV#{eviction_count}] '
        print(f'  Query {query_count:3d} {evict_marker}| '
              f'Chunks: {chunk_tokens_after:6,} tokens | '
              f'Total: {total_tokens:6,} tokens | '
              f'TTFT: {ttft:.3f}s')

        queries.append({
            'query_id': query_count,
            'chunk_tokens': chunk_tokens_after,
            'cumulative_tokens': cumulative_tokens,
            'total_tokens': total_tokens,
            'ttft': ttft,
            'eviction_count': eviction_count,
        })

    print(f'\n✓ Baseline run {run_id} completed: {query_count} queries')
    return {
        'run_id': run_id,
        'experiment': 'baseline',
        'config': {
            'mem_thresh': mem_thresh,
            'chunk_increment': chunk_increment,
            'pre_tokens': pre_tokens,
            'dyn_tokens': dyn_tokens,
            'eviction_ratio': eviction_ratio,
            'num_evictions': num_evictions,
            'model': MODEL_NAME,
        },
        'queries': queries,
        'eviction_at_queries': eviction_at_queries,
        'eviction_count': eviction_count,
    }


# =============================================================================
# ASRManager experiment
# =============================================================================


def run_asr_experiment(
    run_id: int,
    node: ASRManager,
    tokenizer,
    pre_tokens: int,
    dyn_tokens: int,
    chunk_increment: int,
    mem_thresh: int,
    eviction_ratio: float,
    num_evictions: int,
    queries_post_final_eviction: int,
    num_runs: int,
) -> Dict[str, Any]:
    """Run the ASRManager experiment with asynchronous state reconciliation.

    Maps the notebook experiment to the production Algorithm 1 state-JSON
    protocol. Eviction is detected in the benchmark loop (same condition as
    the baseline), then the production reconciliation path is invoked
    directly via ``node._start_reconciliation()``.

    Algorithm 1 state fields computed per query:
        sequence    = pre + chunks + dyn          (full X_t)
        j_t         = len(pre + chunks)           [char length]
        k_t         = current sequence version    (incremented on eviction)
        j_epsilon_t = len(x_epsilon)              [char length]; 0 before
                      first eviction

    Swap detection mirrors the notebook: ``node.stats.swap_count`` is polled
    at the start of each iteration.

    Args:
        run_id:                       Experiment repetition index (1-based).
        node:                         Production ASRManager ROS 2 node.
        tokenizer:                    HuggingFace tokenizer.
        pre_tokens:                   Static prefix size in tokens.
        dyn_tokens:                   Dynamic suffix size in tokens.
        chunk_increment:              Tokens appended per query.
        mem_thresh:                   Eviction threshold in tokens.
        eviction_ratio:               Fraction of chars to keep after eviction.
        num_evictions:                Number of eviction/swap cycles to observe.
        queries_post_final_eviction:  Extra queries after the last swap.
        num_runs:                     Total runs (for progress display only).

    Returns:
        Result dict with keys: run_id, experiment, config, queries,
        swap_at_queries, swap_count, stats.
    """
    sep = '=' * 70
    print(f'\n{sep}')
    print(f'ASR EXPERIMENT — Run {run_id}/{num_runs}')
    print(f'{sep}\n')

    reset_asr_node_state(node)

    print('ASRManager state reset:')
    print(f'  r_primary  : {node._r_primary.name}')
    print(f'  r_secondary: {node._r_secondary.name}')
    print(f'  catchup_thresh: {node.catchup_thresh} chars\n')

    pre = generate_chunk_with_exact_tokens(tokenizer, pre_tokens)
    dyn = generate_chunk_with_exact_tokens(tokenizer, dyn_tokens)

    # Algorithm 1 state tracked by the benchmark.
    # Growing dynamic context (evicted & rebuilt per cycle)
    chunks = ''
    k_t = 0  # Sequence version counter
    # len(x_epsilon) for current k_t; 0 before first eviction
    j_epsilon_t = 0

    eviction_count = 0
    cumulative_tokens = 0
    query_count = 0
    queries: List[Dict[str, Any]] = []
    swap_at_queries: List[int] = []
    previous_swap_count = 0

    while True:
        query_count += 1

        # ------------------------------------------------------------------
        # Detect swap events (mirrors notebook swap detection logic)
        # ------------------------------------------------------------------
        current_swap_count = node.stats.swap_count
        if current_swap_count > previous_swap_count:
            swap_at_queries.append(query_count)
            excl = '!' * 66
            print(f'\n  {excl}')
            print(f'  !!! RESOURCE SWAP #{current_swap_count} '
                  f'DETECTED AT QUERY {query_count}')
            print(f'  !!! r_primary   is now: {node._r_primary.name}')
            print(f'  !!! r_secondary is now: {node._r_secondary.name}')
            print(f'  {excl}\n')
        previous_swap_count = current_swap_count

        # ------------------------------------------------------------------
        # Stop condition: enough post-eviction observation queries collected
        # ------------------------------------------------------------------
        if current_swap_count >= num_evictions:
            last_swap_q = swap_at_queries[-1]
            if query_count > last_swap_q + queries_post_final_eviction:
                break

        # ------------------------------------------------------------------
        # Eviction check — same condition & timing as baseline
        # ------------------------------------------------------------------
        static_tokens = count_tokens(tokenizer, pre + chunks)
        if (static_tokens - 4 * chunk_increment >= mem_thresh
                and eviction_count < num_evictions):
            eviction_count += 1
            k_t += 1

            # Character-based eviction (consistent with j_t / j_epsilon_t).
            keep_len = int(len(chunks) * eviction_ratio)
            evicted_chunks = chunks[-keep_len:] if keep_len > 0 else ''
            x_epsilon = pre + evicted_chunks

            old_tok = count_tokens(tokenizer, chunks)
            new_tok = count_tokens(tokenizer, evicted_chunks)
            print(f'  Query {query_count}: '
                  f'EVICTION #{eviction_count} TRIGGERED '
                  f'(static={static_tokens:,} tokens, '
                  f'k_t: {k_t - 1} → {k_t})')
            print(f'    [EVICTION] {old_tok:,} tokens '
                  f'→ {new_tok:,} tokens '
                  f'(kept {eviction_ratio:.0%})')
            print(f'    [RECON] Triggering reconciliation: '
                  f'|x_epsilon|={len(x_epsilon)} chars')

            # Update working state to the evicted suffix.
            chunks = evicted_chunks
            # j_epsilon_t marks where new chunks begin in the evicted sequence.
            j_epsilon_t = len(x_epsilon)

            # Trigger asynchronous reconciliation on r_secondary.
            node._start_reconciliation(x_epsilon, k_t)

        # ------------------------------------------------------------------
        # Append new chunk and build the Algorithm 1 state JSON
        # ------------------------------------------------------------------
        chunks += generate_chunk_with_exact_tokens(tokenizer, chunk_increment)
        cumulative_tokens += chunk_increment
        chunk_tokens = count_tokens(tokenizer, chunks)

        sequence = pre + chunks + dyn
        j_t = len(pre + chunks)  # char length of static portion
        total_tokens = count_tokens(tokenizer, sequence)

        state_json = build_state_json(
            sequence=sequence,
            j_t=j_t,
            k_t=k_t,
            j_epsilon_t=j_epsilon_t,
        )

        # ------------------------------------------------------------------
        # Measure TTFT
        # ------------------------------------------------------------------
        ttft_start = time.time()
        node.run(
            state_json,
            max_tokens=MAX_TOKENS_OUTPUT,
            temp=0.7,
            seed=42,
            stream=False,
        )
        ttft = time.time() - ttft_start

        if current_swap_count == 0:
            swap_marker = '[PRE-SWAP ] '
        else:
            swap_marker = f'[POST-SW#{current_swap_count}] '
        recon_marker = '[RECON]' if node._is_reconciling else '      '
        print(f'  Query {query_count:3d} {swap_marker} {recon_marker} | '
              f'Chunks: {chunk_tokens:6,} tokens | '
              f'Total: {total_tokens:6,} tokens | '
              f'TTFT: {ttft:.3f}s | '
              f'Server: {node._r_primary.name}')

        queries.append({
            'query_id': query_count,
            'chunk_tokens': chunk_tokens,
            'cumulative_tokens': cumulative_tokens,
            'total_tokens': total_tokens,
            'ttft': ttft,
            'is_reconciling': node._is_reconciling,
            'swap_count': current_swap_count,
            'k_t': k_t,
        })

    # Wait for any in-flight reconciliation before returning.
    if node._reconciliation_thread and node._reconciliation_thread.is_alive():
        print('\n  Waiting for reconciliation thread to finish...')
        node._reconciliation_thread.join(timeout=60.0)
        print('  ✓ Reconciliation thread finished\n')

    stats_snapshot = {
        'total_inference_requests': node.stats.total_inference_requests,
        'case1_continuations': node.stats.case1_continuations,
        'case2_bridge': node.stats.case2_bridge,
        'case2_swap': node.stats.case2_swap,
        'case3_straggler_primary': node.stats.case3_straggler_primary,
        'case3_straggler_secondary': node.stats.case3_straggler_secondary,
        'reconciliation_starts': node.stats.reconciliation_starts,
        'swap_count': node.stats.swap_count,
        'total_warmup_s': node.stats.total_warmup_s,
        'total_catchup_iterations': node.stats.total_catchup_iterations,
    }

    print(f'\n✓ ASR run {run_id} completed: {query_count} queries')
    print(f'  Stats: {stats_snapshot}')

    return {
        'run_id': run_id,
        'experiment': 'asr',
        'config': {
            'mem_thresh': mem_thresh,
            'catchup_thresh': node.catchup_thresh,
            'chunk_increment': chunk_increment,
            'pre_tokens': pre_tokens,
            'dyn_tokens': dyn_tokens,
            'eviction_ratio': eviction_ratio,
            'num_evictions': num_evictions,
            'model': MODEL_NAME,
        },
        'queries': queries,
        'swap_at_queries': swap_at_queries,
        'swap_count': current_swap_count,
        'stats': stats_snapshot,
    }


# =============================================================================
# Run all experiments
# =============================================================================


def run_all_experiments(
    node: ASRManager,
    baseline_client: InferenceClient,
    tokenizer,
    num_runs: int,
    pre_tokens: int,
    dyn_tokens: int,
    chunk_increment: int,
    mem_thresh: int,
    eviction_ratio: float,
    num_evictions: int,
    queries_post_final_eviction: int,
    results_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Execute all baseline and ASRManager repetitions and save results."""

    print('\n' + '=' * 70)
    print('STARTING TTFT EXPERIMENTS')
    print('=' * 70)
    print(f'Total runs        : {num_runs}')
    print('Experiments       : Baseline + ASRManager')
    print(f'MEM_THRESH        : {mem_thresh:,} tokens')
    print(f'CHUNK_INCREMENT   : {chunk_increment} tokens/query')
    print(f'PRE_TOKENS        : {pre_tokens}')
    print(f'NUM_EVICTIONS     : {num_evictions}')
    print(f'EVICTION_RATIO    : {eviction_ratio}')
    print('=' * 70 + '\n')

    all_results: Dict[str, List[Dict[str, Any]]] = {
        'baseline': [],
        'asr': [],
    }

    # ------------------------------------------------------------------
    # Baseline runs
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('BASELINE EXPERIMENTS')
    print('=' * 70)

    for run_id in range(1, num_runs + 1):
        result = run_baseline_experiment(
            run_id=run_id,
            inference_client=baseline_client,
            tokenizer=tokenizer,
            pre_tokens=pre_tokens,
            dyn_tokens=dyn_tokens,
            chunk_increment=chunk_increment,
            mem_thresh=mem_thresh,
            eviction_ratio=eviction_ratio,
            num_evictions=num_evictions,
            queries_post_final_eviction=queries_post_final_eviction,
            num_runs=num_runs,
        )
        all_results['baseline'].append(result)

        output_file = results_dir / f'ttft_baseline_run{run_id}.json'
        with open(output_file, 'w') as fh:
            json.dump(result, fh, indent=2)
        print(f'  → Saved to {output_file}')

    print('\n✓ All baseline experiments completed\n')

    # ------------------------------------------------------------------
    # ASRManager runs
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('ASR EXPERIMENTS')
    print('=' * 70)

    for run_id in range(1, num_runs + 1):
        result = run_asr_experiment(
            run_id=run_id,
            node=node,
            tokenizer=tokenizer,
            pre_tokens=pre_tokens,
            dyn_tokens=dyn_tokens,
            chunk_increment=chunk_increment,
            mem_thresh=mem_thresh,
            eviction_ratio=eviction_ratio,
            num_evictions=num_evictions,
            queries_post_final_eviction=queries_post_final_eviction,
            num_runs=num_runs,
        )
        all_results['asr'].append(result)

        output_file = results_dir / f'ttft_asr_run{run_id}.json'
        with open(output_file, 'w') as fh:
            json.dump(result, fh, indent=2)
        print(f'  → Saved to {output_file}')

    print('\n✓ All ASR experiments completed\n')
    return all_results


# =============================================================================
# Data analysis
# =============================================================================


def aggregate_results(
    results: Dict[str, List[Dict[str, Any]]], ) -> Dict[str, Any]:
    """Aggregate TTFT statistics across runs, grouped by cumulative token count.

    For each unique cumulative token count observed across all runs, computes
    mean, std, min, and max TTFT.  The aggregated data is used for plotting.

    Args:
        results: {'baseline': [...], 'asr': [...]} — raw per-run result dicts.

    Returns:
        dict with 'baseline' and 'asr' keys, each mapping to a sub-dict with
        'aggregated' (list of stat dicts sorted by cumulative_tokens) and
        'num_runs'.
    """

    def _process(exp_results: List[Dict]) -> Dict:
        data_by_tok: Dict[int, List[float]] = {}
        for run in exp_results:
            for q in run['queries']:
                tok = q['cumulative_tokens']
                data_by_tok.setdefault(tok, []).append(q['ttft'])

        aggregated = []
        for tok in sorted(data_by_tok):
            ttfts = data_by_tok[tok]
            aggregated.append({
                'cumulative_tokens': tok,
                'mean_ttft': float(np.mean(ttfts)),
                'std_ttft': float(np.std(ttfts)),
                'min_ttft': float(np.min(ttfts)),
                'max_ttft': float(np.max(ttfts)),
                'count': len(ttfts),
            })
        return {'aggregated': aggregated, 'num_runs': len(exp_results)}

    return {
        'baseline': _process(results['baseline']),
        'asr': _process(results['asr']),
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_ttft_comparison(
    aggregated: Dict[str, Any],
    results: Dict[str, List[Dict[str, Any]]],
    mem_thresh_tokens: int,
    save_path: Optional[Path] = None,
) -> None:
    """Create the TTFT comparison figure (2-panel layout, like the notebook).

    Panel 1: Mean TTFT ± std vs cumulative tokens — baseline vs ASRManager.
    Panel 2: Cache state size (tokens) vs cumulative tokens for one run.

    Reconciliation periods (between eviction detection and swap completion)
    are shaded in orange on Panel 1.

    Args:
        aggregated:          Output of aggregate_results().
        results:             Raw per-run result dicts.
        mem_thresh_tokens:   Eviction threshold for the vertical threshold line.
        save_path:           If provided the figure is saved to this path.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ------------------------------------------------------------------
    # Panel 1: TTFT vs cumulative tokens
    # ------------------------------------------------------------------
    baseline_agg = aggregated['baseline']['aggregated']
    asr_agg = aggregated['asr']['aggregated']

    b_cumtok = [x['cumulative_tokens'] for x in baseline_agg]
    b_mean = np.array([x['mean_ttft'] for x in baseline_agg])
    b_std = np.array([x['std_ttft'] for x in baseline_agg])

    a_cumtok = [x['cumulative_tokens'] for x in asr_agg]
    a_mean = np.array([x['mean_ttft'] for x in asr_agg])
    a_std = np.array([x['std_ttft'] for x in asr_agg])

    ax1.plot(b_cumtok,
             b_mean,
             'o-',
             color='#1f77b4',
             linewidth=2,
             markersize=4,
             label='Baseline',
             alpha=0.8)
    ax1.fill_between(b_cumtok,
                     b_mean - b_std,
                     b_mean + b_std,
                     alpha=0.2,
                     color='#1f77b4')

    ax1.plot(a_cumtok,
             a_mean,
             's-',
             color='#ff7f0e',
             linewidth=2,
             markersize=4,
             label='ASRManager',
             alpha=0.8)
    ax1.fill_between(a_cumtok,
                     a_mean - a_std,
                     a_mean + a_std,
                     alpha=0.2,
                     color='#ff7f0e')

    thresh_label = (
        f'Eviction Threshold ({mem_thresh_tokens / 1000:.0f}K tokens)')
    ax1.axvline(mem_thresh_tokens,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=thresh_label)

    # Shade reconciliation periods (chunk_tokens >= threshold → swap detected).
    if results['asr'] and results['asr'][0].get('swap_at_queries'):
        recon_periods: List[Tuple[int, int]] = []
        prev_swap = 0
        recon_start: Optional[int] = None

        for q in results['asr'][0]['queries']:
            curr_swap = q['swap_count']
            if q['chunk_tokens'] >= mem_thresh_tokens and recon_start is None:
                recon_start = q['cumulative_tokens']
            if curr_swap > prev_swap and recon_start is not None:
                recon_periods.append((recon_start, q['cumulative_tokens']))
                recon_start = None
            prev_swap = curr_swap

        for i, (start, end) in enumerate(recon_periods):
            ax1.axvspan(start,
                        end,
                        alpha=0.1,
                        color='orange',
                        label='Reconciliation Periods' if i == 0 else None)

    ax1.set_ylabel('TTFT (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time-to-First-Token During Cache Eviction',
                  fontsize=14,
                  fontweight='bold',
                  pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel 2: Cache state size vs cumulative tokens (run 1 only)
    # ------------------------------------------------------------------
    b_run = results['baseline'][0]
    a_run = results['asr'][0]

    b_ctok = [q['cumulative_tokens'] for q in b_run['queries']]
    b_csz = [q['chunk_tokens'] for q in b_run['queries']]
    a_ctok = [q['cumulative_tokens'] for q in a_run['queries']]
    a_csz = [q['chunk_tokens'] for q in a_run['queries']]

    ax2.plot(b_ctok,
             b_csz,
             'o-',
             color='#1f77b4',
             linewidth=2,
             markersize=3,
             label='Baseline (Run 1)',
             alpha=0.6)
    ax2.plot(a_ctok,
             a_csz,
             's-',
             color='#ff7f0e',
             linewidth=2,
             markersize=3,
             label='ASRManager (Run 1)',
             alpha=0.6)

    # Mark eviction points (baseline).
    query_by_id = {q['query_id']: q for q in b_run['queries']}
    for evict_qid in b_run.get('eviction_at_queries', []):
        if evict_qid in query_by_id:
            ax2.axvline(query_by_id[evict_qid]['cumulative_tokens'],
                        color='#1f77b4',
                        linestyle=':',
                        alpha=0.5,
                        linewidth=1.5)

    # Mark swap points (ASRManager).
    query_by_id_asr = {q['query_id']: q for q in a_run['queries']}
    for swap_qid in a_run.get('swap_at_queries', []):
        if swap_qid in query_by_id_asr:
            ax2.axvline(query_by_id_asr[swap_qid]['cumulative_tokens'],
                        color='#ff7f0e',
                        linestyle=':',
                        alpha=0.5,
                        linewidth=1.5)

    ax2.set_xlabel('Cumulative Tokens Appended',
                   fontsize=12,
                   fontweight='bold')
    ax2.set_ylabel('Current Chunk Size (tokens)',
                   fontsize=12,
                   fontweight='bold')
    ax2.set_title('Cache State Size vs Cumulative Tokens',
                  fontsize=12,
                  fontweight='bold',
                  pad=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\n✓ Figure saved to: {save_path}')

    plt.close(fig)


# =============================================================================
# CLI / entry point
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='ASRManager TTFT performance benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model',
                   default=MODEL_NAME,
                   help='HuggingFace model name')
    p.add_argument('--r1-url',
                   default=SERVER_1_URL,
                   help='vLLM server 1 URL (primary / baseline)')
    p.add_argument('--r2-url',
                   default=SERVER_2_URL,
                   help='vLLM server 2 URL (secondary for ASRManager)')
    p.add_argument('--mem-thresh',
                   default=MEM_THRESH,
                   type=int,
                   help='Eviction threshold (tokens)')
    p.add_argument('--catchup-thresh',
                   default=CATCHUP_THRESH,
                   type=int,
                   help='ASRManager catch-up buffer threshold (chars)')
    p.add_argument('--eviction-ratio',
                   default=EVICTION_RATIO,
                   type=float,
                   help='Fraction of chars to keep after eviction')
    p.add_argument('--chunk-increment',
                   default=CHUNK_INCREMENT,
                   type=int,
                   help='Tokens appended per query')
    p.add_argument('--pre-tokens',
                   default=PRE_TOKENS,
                   type=int,
                   help='Static prefix size (tokens)')
    p.add_argument('--dyn-tokens',
                   default=DYN_TOKENS,
                   type=int,
                   help='Dynamic suffix size (tokens)')
    p.add_argument('--num-runs',
                   default=NUM_RUNS,
                   type=int,
                   help='Number of repetitions per condition')
    p.add_argument('--num-evictions',
                   default=NUM_EVICTIONS,
                   type=int,
                   help='Eviction cycles to observe per run')
    p.add_argument('--results-dir',
                   default=str(RESULTS_DIR),
                   help='Directory to save JSON results and figures')
    p.add_argument('--asr-http-port',
                   default=ASR_HTTP_PORT,
                   type=int,
                   help='HTTP port for ASRManager internal server')
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """Benchmark entry point."""
    args = _build_arg_parser().parse_args(argv)

    queries_post = int(5_000 / args.chunk_increment)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ROS 2 initialisation
    # ------------------------------------------------------------------
    rclpy.init(args=None)

    node = ASRManager(parameter_overrides=[
        Parameter('model_name', Parameter.Type.STRING, args.model),
        Parameter('r1_url', Parameter.Type.STRING, args.r1_url),
        Parameter('r2_url', Parameter.Type.STRING, args.r2_url),
        Parameter('client_type', Parameter.Type.STRING, 'vllm'),
        Parameter('catchup_thresh', Parameter.Type.INTEGER,
                  args.catchup_thresh),
        Parameter('http_host', Parameter.Type.STRING, '127.0.0.1'),
        Parameter('http_port', Parameter.Type.INTEGER, args.asr_http_port),
    ])

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin,
                                       daemon=True,
                                       name='ros2-executor')
    executor_thread.start()

    # Separate baseline client — does not share state with ASRManager's clients.
    baseline_client = create_inference_client(
        ClientType.VLLM,
        name='baseline',
        model_name=args.model,
        url=args.r1_url,
    )

    print(f'Loading tokenizer: {args.model} …')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('✓ Tokenizer loaded\n')

    print('vLLM Server Configuration:')
    print(f'  Server 1 ({NAME_1}): {args.r1_url} '
          '(baseline + ASRManager primary)')
    print(f'  Server 2 ({NAME_2}): {args.r2_url} '
          '(ASRManager secondary)')
    print(f'  ASRManager HTTP : '
          f'http://127.0.0.1:{args.asr_http_port}\n')

    try:
        # ------------------------------------------------------------------
        # Run experiments
        # ------------------------------------------------------------------
        results = run_all_experiments(
            node=node,
            baseline_client=baseline_client,
            tokenizer=tokenizer,
            num_runs=args.num_runs,
            pre_tokens=args.pre_tokens,
            dyn_tokens=args.dyn_tokens,
            chunk_increment=args.chunk_increment,
            mem_thresh=args.mem_thresh,
            eviction_ratio=args.eviction_ratio,
            num_evictions=args.num_evictions,
            queries_post_final_eviction=queries_post,
            results_dir=results_dir,
        )

        print('\n' + '=' * 70)
        print('ALL EXPERIMENTS COMPLETED')
        print('=' * 70)
        print(f'Baseline runs : {len(results["baseline"])}')
        print(f'ASR runs      : {len(results["asr"])}')
        print(f'Results dir   : {results_dir}')
        print('=' * 70)

        # ------------------------------------------------------------------
        # Analysis & visualisation
        # ------------------------------------------------------------------
        aggregated = aggregate_results(results)

        agg_path = results_dir / 'ttft_aggregated.json'
        with open(agg_path, 'w') as fh:
            json.dump(aggregated, fh, indent=2, default=float)
        print(f'\n✓ Aggregated statistics saved to: {agg_path}')

        print('\nBaseline statistics (first 5 cumulative token counts):')
        for stat in aggregated['baseline']['aggregated'][:5]:
            print(f"  {stat['cumulative_tokens']:6,} tokens: "
                  f"mean={stat['mean_ttft']:.3f}s, "
                  f"std={stat['std_ttft']:.3f}s, "
                  f"n={stat['count']}")

        fig_path = results_dir / 'ttft_comparison_figure.png'
        plot_ttft_comparison(
            aggregated=aggregated,
            results=results,
            mem_thresh_tokens=args.mem_thresh,
            save_path=fig_path,
        )

    finally:
        executor.shutdown(timeout_sec=5.0)
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
