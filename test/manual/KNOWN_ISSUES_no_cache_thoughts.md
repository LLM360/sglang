# Known issues — `--no-cache-thoughts` split-insertion path

## Issue 1 — KV pool accounting leak under concurrent load

**Symptom**

When the synthetic bench (`bench_no_cache_thoughts_synthetic.py`) runs the
B-sweep at B=16 concurrent requests with N=10-turn TITO conversations, the
NCT sglang server crashes after the first batch of NCT requests finishes:

```
ValueError: token_to_kv_pool_allocator memory leak detected!
  self.max_total_num_tokens=686547,
  available_size=484318,
  evictable_size=202232,
  protected_size=0,
  session_held=0
```

`available + evictable + protected + session_held = 686550`, off by **+3
tokens** from the configured max. The bookkeeping thinks there are more
tokens free than physically exist — the hallmark of a double-free.

After the crash, sglang sends `SIGQUIT` ("usually means one child
failed") which takes down the entire srun, killing the baseline server
on the other GPU as well.

**Reproduction**

1. Two sglang servers on one node (slurm bench stand), one with
   `--no-cache-thoughts`.
2. Drive the NCT server with B=16 concurrent N=10-turn conversations
   via the synthetic bench's TITO path (alternating `messages=` turn 1,
   `input_ids=` turn N+1).
3. First trial of B=16 against NCT completes 16 conversations with
   visible decode batch activity, then the scheduler raises the leak
   error mid-batch. All subsequent requests fail with connection
   refused.

The leak does NOT reproduce on the baseline server under the same load,
nor on the NCT server at B=1 or B=4. The crash signature is consistent
across 3 trials at B=16.

**Suspected root cause**

The split-insertion path lives in:
  - `python/sglang/srt/mem_cache/common.py`
    - `split_kv_for_no_cache_thoughts` (line 113-192) — computes the
      virtual_kv_indices (input + answer, thoughts excluded) and
      thought_kv_indices_to_free (the thought slice).
    - `release_kv_cache` (line 639-708) — calls
      `cache_finished_req(req, split=split)` then calls
      `pop_overallocated_kv_cache()` and frees `[start_p, end_p)`
      from `req_to_token`.
  - `python/sglang/srt/mem_cache/radix_cache.py`
    - `cache_finished_req` split branch (line 511-549) — frees
      `split.thought_kv_indices_to_free` and the duplicate +
      page-aligned-tail regions of `split.virtual_kv_indices`.

Strongest candidate: the split path uses
`total_len = input_len + len(output_ids)` for slot accounting; the
baseline path uses `(input + output)[:kv_committed_len]`. If
`kv_committed_len < input_len + len(output_ids)` (e.g., under
speculative decoding, partial commits during preemption, or any
abort/retract path), the slots in `[kv_committed_len, total_len)` are
freed by BOTH `cache_finished_req(split=…)` AND the subsequent
`pop_overallocated_kv_cache` call in `release_kv_cache`. Double-free
into the allocator → `available_size` increments past `max_total_num_tokens`.

The 3-token discrepancy is consistent with a small constant-size race
(e.g., a single request leaking exactly its over-committed tail). At
concurrent B=16, the bug compounds quickly enough to trip the runtime
check on the next `check_memory` tick.

**Workarounds / next steps**

  - Until fixed, do not enable `--no-cache-thoughts` for workloads with
    sustained concurrent load (RL rollouts at B ≥ 8 are likely
    affected). Single-request and small-batch (B ≤ 4) use is stable.
  - Reproduction script: see the synthetic bench above. The crash is
    deterministic with the BBQ-8B / k2v3-7b-SFT models.
  - Suggested diagnostic: instrument `token_to_kv_pool_allocator.free`
    to log every freed range with the call stack, then diff the
    pre/post-crash logs against `req_to_token_pool` to find which
    indices were freed twice.
  - Suggested fix direction: align the split path's slot ownership
    accounting with `kv_committed_len`, mirroring what the baseline
    path does. The split should free only slots actually committed,
    leaving overallocated slots for `pop_overallocated_kv_cache` to
    handle exclusively.

This issue does NOT affect the per-step inference-speed measurements
from the N-sweep at B=1, which completed cleanly.
