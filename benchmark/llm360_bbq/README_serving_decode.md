# BBQ SGLang decode-serving benchmark

This runner measures native SGLang serving on one H200 node. It uses the
repository's `sglang.launch_server` and `sglang.bench_serving` entry points with
the following fixed release-comparison contract:

- native SGLang model implementation, BF16 weights/KV cache, and FA3 attention;
- deterministic `random-ids` inputs, sent as token IDs with seed `20260827`;
- exactly 512 input and 512 output tokens per measured request;
- greedy decoding with EOS ignored;
- a warmup phase followed by a cache flush and the measured run;
- CUDA graphs and the radix cache left enabled;
- an external read-only checkpoint bind mount plus before/after stat manifests.

`BBQ_NUM_REQUESTS`, `BBQ_MAX_CONCURRENCY`, and `BBQ_WARMUP_REQUESTS` are
configurable. Defaults are 64, 8, and 8 respectively. The workload sends all
measured requests immediately and uses the concurrency limit to saturate the
server.

Create the Slurm log directory once, then submit a model. Export variables in
the environment rather than putting JSON containing commas in `sbatch
--export`.

```bash
mkdir -p /mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm
cd /mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/sglang

export BBQ_MODEL_NAME=bbq-8b
export BBQ_MODEL_PATH=/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-pretrain-final
export BBQ_TP_SIZE=1
export BBQ_NUM_REQUESTS=64
export BBQ_MAX_CONCURRENCY=8
export BBQ_WARMUP_REQUESTS=8
sbatch benchmark/llm360_bbq/run_serving_decode.slurm
```

For a model that needs explicit config provenance, export a JSON object before
submission. It is passed only through SGLang's `--json-model-override-args`:

```bash
export BBQ_JSON_MODEL_OVERRIDE_ARGS='{"xllm_source_router_gemm_partitions": 2}'
```

Optional controls are `BBQ_SEED`, `BBQ_PORT`, `BBQ_SERVER_READY_TIMEOUT`,
`BBQ_MEM_FRACTION_STATIC`, `BBQ_MAX_TOTAL_TOKENS`, and
`BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON`. The default `{}` exercises SGLang's
production bounded-multithread/mmap loader for PyTorch `.bin` artifacts. Use
`{"enable_multithread_load": false}` only as a diagnostic sequential-load
fallback. `BBQ_MAX_TOTAL_TOKENS` is useful when an automatic KV-cache allocation
is inappropriate for a very large model.

Results are written only below
`artifacts/models/$BBQ_MODEL_NAME/decode-job-$SLURM_JOB_ID/`. The primary
artifact is `decode-serving-summary.json`; it contains aggregate output and
total token throughput plus p50/p95 TTFT and flattened inter-token latency.
`bench-serving-raw.jsonl`, command lines, server logs, environment details, GPU
inventory, source snapshots, and checkpoint stat manifests are retained beside
it. `status.txt` is `PASS` only when every request completes at exactly 512/512,
the server reports the required runtime settings (including disabled fused
QK-norm/RoPE), release-critical source hashes are captured, and the checkpoint
manifest is unchanged.

This is a performance and serving-contract benchmark, not a semantic
correctness test. Pair it with the log-probability parity and task-evaluation
artifacts for release acceptance.
