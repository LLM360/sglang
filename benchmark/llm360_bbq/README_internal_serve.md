# BBQ internal SGLang endpoints

This is a deliberately small, temporary internal deployment: one exclusive
eight-H200 Slurm node per release model, with one native-SGLang OpenAI-compatible
server on each node. It is suitable for shared evaluation and manual requests;
it is not a highly available production service.

## Exact model matrix

| Array task | Served name | Checkpoint | TP | API | Extra serving contract |
|---:|---|---|---:|---|---|
| 0 | `bbq-1b` | `/mnt/weka/shrd/k2m/junlin.chen/xllm_1b_final/model` | 1 | completion + chat | default loader |
| 1 | `bbq-4b` | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-4b-pretrain-final` | 1 | completion only | sequential loader |
| 2 | `bbq-7b` | `/mnt/weka/shrd/k2m/junlin.chen/ckpts/k2v3-7B_iso_attn_shared_small_phase1_sft_2199941/huggingface/checkpoint_0010000` | 1 | completion + chat | default loader |
| 3 | `bbq-32b` | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-32b-pretrain-final` | 2 | completion only | sequential loader |
| 4 | `k2-mova-36b` | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000` | 2 | completion + chat | router GEMM partitions 2; 32,768-token pool |
| 5 | `k2-moe-375b` | `/mnt/weka/home/mrunner/workspace/checkpoints/huggingface/k2moe375B_mid4_v2_200B_256nodes_seed42_bsz32M_seq512k_jais250k_ep8_dot_te_bestfit/checkpoints/checkpoint_0006000` | 8 | completion + chat | 32,768-token pool |

All tasks use native SGLang, BF16 weights and KV cache, FA3 attention,
`mem-fraction-static=0.85`, and the saved tokenizer/chat template. Fused
QK-norm/RoPE is omitted and must remain disabled, especially for MoVA. The
checkpoint is a nested read-only container mount and the worker verifies the
effective mount before loading.

The 4B and 32B artifacts do not contain a chat template, so their supported
OpenAI interface is `/v1/completions`; do not send them chat-completions
requests. Endpoint records publish this capability explicitly. The other four
artifacts have a saved `chat_template.jinja` and are smoked through both
interfaces before becoming `READY`.

The retained 4B and 32B checkpoints declare only 8,192 tokens of context. Do
not send 16K requests to those two endpoints. Lower-TP models intentionally
leave GPUs idle because this deployment reserves one whole node per model.

## Launch

The default allocation lifetime is 24 hours. The `main` partition currently
has no configured maximum time, so override it explicitly when a different
lifetime is justified:

```bash
cd /mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/sglang
mkdir -p ../artifacts/slurm
sbatch --time=24:00:00 benchmark/llm360_bbq/run_internal_serve.slurm
```

The file is a six-task array (`0-5`). Each task requests one exclusive node and
all eight H200s; tasks can become ready independently. To launch or retry one
model, override the array selection, for example `sbatch --array=4 ...` for
MoVA. Ports default to 31000 through 31005 and can be moved as a group with
`BBQ_SERVE_PORT_BASE`.

Startup history is roughly one minute for 1B, 1.5 minutes for 4B/7B, four
minutes for 32B, 13 minutes for MoVA, and 16 minutes for 375B. The readiness
deadline therefore defaults to 1,800 seconds.

### Optional bearer key

Without a key, any user who can route to the compute-node network can invoke
the inference API and optional management endpoints such as cache flush. This
is the least-friction mode, but a shared bearer key is safer on a multi-user
cluster. Put the key in a mode-600 file below the project root, then submit with
its path:

```bash
mkdir -p ../artifacts/internal-serving/secrets
chmod 700 ../artifacts/internal-serving/secrets
openssl rand -hex 32 > ../artifacts/internal-serving/secrets/api-key
chmod 600 ../artifacts/internal-serving/secrets/api-key
export BBQ_SERVE_API_KEY_FILE=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/internal-serving/secrets/api-key
sbatch --export=ALL,BBQ_SERVE_API_KEY_FILE="${BBQ_SERVE_API_KEY_FILE}" benchmark/llm360_bbq/run_internal_serve.slurm
```

The key is never written into endpoint metadata or command artifacts. Keyed
servers use warning-level logs because SGLang's INFO-level startup record
contains its full server arguments. SGLang still receives the key as a process
argument, so this is an internal shared-secret control, not protection against
node administrators or privileged process inspection. There is no TLS; never
expose these endpoints to the public internet. `/health` and `/metrics` remain
unauthenticated by SGLang design.

## Discovery and remote verification

After each local startup contract and completion smoke passes, the task
atomically publishes:

```text
/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/internal-serving/endpoints/<served-name>.json
```

Each record includes status, current compute hostname/IP, OpenAI base URL,
checkpoint, TP, Slurm IDs, SGLang commit, and timestamp. The canonical record
is protected from an older job overwriting a newer deployment during cleanup.
Always reread it rather than caching a node name: a Slurm requeue can move the
job to another node.

From a login node or a seventh compute node, run all six network-level smokes:

```bash
python3 benchmark/llm360_bbq/smoke_internal_serve.py \
  --output ../artifacts/internal-serving/remote-smoke.json
```

For keyed endpoints, add `--api-key-file "$BBQ_SERVE_API_KEY_FILE"`. The smoke
checks `/health`, `/v1/models`, and one greedy `/v1/completions` request in
parallel; it also checks `/v1/chat/completions` for models that publish that
capability. It exits nonzero unless all selected endpoints pass. A single
model can be checked with `--model k2-mova-36b`.

Standard OpenAI clients can use the record's `base_url` and
`served_model_name`. For example, after reading those two fields from the JSON,
configure the client with `base_url="http://<compute-node>:<port>/v1"` and the
shared key (or any placeholder string when authentication is disabled).

## Lifetime and failure behavior

These are Slurm allocations, not stable DNS services. Cancellation, time
limit, preemption, node failure, or an SGLang crash removes an endpoint.
`--requeue` lets Slurm restart eligible preempted/failed allocations, and the
registry will be updated when the replacement is ready, but an application
crash is deliberately fail-closed rather than hidden by an infinite restart
loop. Endpoint records become `STOPPED`, `FAILED`, or `UNHEALTHY` when the
worker can observe the event; a hard node loss can leave a stale `READY`
record, which is why clients and operators should use `/health` or the remote
smoke before starting a long evaluation.

Stop the full array with `scancel <array-job-id>`, or one task with
`scancel <array-job-id>_<task-id>`. Logs and per-job endpoint history live
under `artifacts/internal-serving/jobs/`.
