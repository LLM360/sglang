# BBQ SGLang release-verification snapshot

Snapshot time: **2026-08-27 16:37 UTC**. This is a review-only evidence
snapshot, not a merge or release approval. `PASS` below is the status recorded
by the named artifact; running and missing work is never promoted to `PASS`.

## Checkpoint inventory

| Release model | Checkpoint used | Physical checkpoint dtype | Serving/reference dtype | Declared context |
|---|---|---:|---:|---:|
| BBQ-1B | **Missing: no checkpoint path supplied** | — | — | — |
| BBQ-4B | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-4b-pretrain-final` | FP32 `.bin` | BF16 | 8,192 |
| BBQ-8B (also referred to as 7B) | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-pretrain-final` | FP32 `.bin` | BF16 | 8,192 |
| BBQ-32B | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-32b-pretrain-final` | FP32 `.bin` | BF16 | 8,192 |
| K2-MoVA-36B Mid4 V2 | `/mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000` | FP32 `.bin` | BF16 | 524,288 |
| K2-MoE-375B Mid4 V2 | `/mnt/weka/home/mrunner/workspace/checkpoints/huggingface/k2moe375B_mid4_v2_200B_256nodes_seed42_bsz32M_seq512k_jais250k_ep8_dot_te_bestfit/checkpoints/checkpoint_0006000` | FP32 `.bin` | BF16 | 524,288 |

No checkpoint conversion or checkpoint write is represented in this evidence.
The original Hugging Face-layout directories were consumed read-only. All 20
accepted completed model jobs listed in this report have captured
`checkpoint-stat-before.txt` and `checkpoint-stat-after.txt` manifests, and
each before/after pair is byte-for-byte identical. Manifest scope evolved:
historical probe and HF pairs enumerate the config and index, while GSM8K and
serving pairs enumerate every checkpoint entry. Pair equality proves the
metadata each job observed did not change; the read-only bind mount is the
write-protection control.

## Release evidence matrix

| Model | SGLang load/smoke/decode probe | Independent HF teacher-forced parity | Full GSM8K | Production serving decode | Evidence disposition |
|---|---|---|---|---|---|
| BBQ-1B | Missing | Missing | Missing | Missing | **Blocked on checkpoint path** |
| BBQ-4B | **PASS**, job `2243123`, TP1 | **PASS**, job `2243247` | **PASS**, job `2243513` | **PASS**, job `2243343` | Dense evidence complete |
| BBQ-8B | **PASS**, job `2243124`, TP1 | **PASS**, job `2243248` | **PASS**, job `2243514` | **PASS**, job `2243344` | Dense evidence complete |
| BBQ-32B | **PASS**, job `2243125`, TP2 | **PASS**, job `2243249` | **PASS**, job `2243512` | **PASS**, job `2243345` | Dense evidence complete |
| K2-MoVA-36B | **PASS (provisional checkpoint)**, job `2243439`, TP2 | **PASS**, job `2243740` | **PASS**, job `2243660` | **PASS**, job `2243661` | Evidence complete for the selected checkpoint; release identity still needs owner confirmation |
| K2-MoE-375B | **PASS (provisional checkpoint)**, job `2243314`, TP8 | **PASS**, corrected job `2244882` | **PASS**, job `2243663` | **PASS**, job `2243662` | Artifact-serving evidence complete; release-checkpoint confirmation and the source-MP8 caveat remain open |

## SGLang offline probe results

The decode microbenchmark uses 128 input and 128 output tokens per request. It
is useful for a deterministic engine smoke test and coarse scaling check; the
production serving benchmark later in this document is the release throughput
measurement.

| Model | Job | Init (s) | TP | Decode tok/s, bs1 | bs8 | bs16 | Long-context check | Cached-repeat determinism |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BBQ-4B | `2243123` | 52.25 | 1 | 172.36 | 1,326.03 | 2,962.73 | **PASS**, 7,936 in + 8 out | **PASS** |
| BBQ-8B | `2243124` | 61.62 | 1 | 134.28 | 1,023.61 | 2,275.31 | **PASS**, 7,936 in + 8 out | **PASS** |
| BBQ-32B | `2243125` | 100.26 | 2 | 70.29 | 532.65 | 1,093.75 | **PASS**, 7,936 in + 8 out | **PASS** |
| K2-MoVA-36B | `2243439` | 516.27 | 2 | 84.03 | 500.01 | 958.71 | **PASS**, 16,384 in + 8 out | **PASS** |
| K2-MoE-375B | `2243314` | 1,712.46 | 8 | 94.72 | 521.68 | 888.51 | **PASS**, 16,384 in + 8 out | **PASS** |

The dense checkpoints declare an 8,192-token maximum. Their 7,936-token probe
leaves room for generation and is the only long-context claim made here; the
engine's larger token-pool allocation does not increase model context support.
The two large-model initialization times include intentionally sequential shard
loading and CUDA-graph preparation, so they are diagnostic timings rather than
optimized startup targets.

## Independent Hugging Face parity

These are BF16, eager-attention, teacher-forced comparisons of SGLang candidate
log probabilities against an independent Hugging Face reference over 256
positions. Signed error is `SGLang - HF`; the importance ratio is
`exp(SGLang - HF)`. All five completed runs pass the stored gate: mean absolute
error at most 0.05 nat, p95 at most 0.15 nat, non-tie mismatch at most 1%,
tie-aware greedy agreement at least 99%, and top-5 overlap at least 95%.

| Model | Job | Status | Mean abs. error (nat) | Mean signed error (nat) | Median importance ratio | Exact greedy | Tie-aware greedy | Top-5 overlap |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| BBQ-4B | `2243247` | **PASS** | 0.018319 | +0.000377 | 1.000004 | 96.875% | 100.000% | 98.281% |
| BBQ-8B | `2243248` | **PASS** | 0.015077 | -0.000190 | 1.000009 | 99.219% | 100.000% | 98.516% |
| BBQ-32B | `2243249` | **PASS** | 0.016167 | +0.001159 | 0.999946 | 98.438% | 100.000% | 98.359% |
| K2-MoVA-36B | `2243740` | **PASS** | 0.023615 | -0.000955 | 0.999997 | 96.875% | 100.000% | 95.859% |
| K2-MoE-375B | `2244882` | **PASS** | 0.033808 | +0.004964 | 1.000025 | 98.828% | 100.000% | 96.094% |

The accepted 375B oracle loaded in 826.14 seconds and records exactly 16 prompts
and 256 compared positions. Prompt indices, input IDs, and 16-token candidate
outputs match probe job `2243314`; every prompt contributes 16 HF and 16 SGLang
positions. Its exact gate values are mean absolute error
`0.03380764701039318`, p95 absolute error `0.13743305206298828`, mean signed
error `+0.0049644680271914154`, median importance ratio
`1.0000251115668464`, exact greedy agreement `0.98828125`, tie-aware agreement
`1.0`, top-5 overlap `0.9609375`, and non-tie mismatch fraction `0.0`. All five
stored gate checks are true.

The same JSON validates the oracle-specific safety contract:

- its 58 unique router biases are exactly
  `model.layers.3.mlp.gate.bias` through
  `model.layers.60.mlp.gate.bias`; every source tensor is FP32, the generic
  loader initially materialized them as BF16, and every effective tensor was
  restored to FP32 with `preserve_fp32=true`;
- the requested 65-entry manual map exactly equals `hf_device_map`; values are
  GPU strings `0` through `7` only, with no CPU or disk placement (embeddings
  and rotary buffer on GPU 0, final norm and LM head on GPU 7);
- the partial-RoPE compatibility patch applied to exactly
  `XllmRotaryEmbedding`.

### 375B HF-oracle attempt ledger

All attempt paths below are beneath
`/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/models/k2moe-375b`.

| Job | Artifact directory | Outcome | Checkpoint-manifest state |
|---:|---|---|---|
| `2243655` | `hf-job-2243655` | **FAILED:** automatic/balanced mapping attempted disk offload for the FP32 `.bin` MoE artifact without an offload folder; no parity metrics | Before manifest only |
| `2243741` | `hf-job-2243741` | **FAILED after loading all 34,076 weights:** Transformers-5 post-load initialization rebuilt the partial-RoPE buffer with 64 rather than the artifact's 32 inverse frequencies | Before manifest only |
| `2244521` | `hf-job-2244521` | **FAILED CLOSED before weight I/O:** the first exact 61-layer/192-expert Mid4 partial-RoPE guard rejected the normalized config | Before/after manifests present and identical |
| `2244882` | `hf-job-2244882` | **PASS:** corrected exact guard and partial-RoPE compatibility path; all parity gates pass | Before/after manifests present and identical |

## Full GSM8K correctness smoke

All completed runs evaluate all 1,319 canonical rows using the exact xLLM
eight-shot raw-completion prompt contract, greedy decoding, and up to 512 new
tokens. The dense jobs predate the later gross-quality-gate field. The MoVA and
375B artifacts pass that gate (accuracy at least 50%, invalid fraction at most
5%), whose stated purpose is only to catch a gross wrong-model or corrupt load;
it is not HF-vs-SGLang baseline parity.

| Model | Job | Status | Correct / 1,319 | Accuracy | Invalid | Output tok/s |
|---|---:|---|---:|---:|---:|---:|
| BBQ-4B | `2243513` | **PASS** | 1,045 | 79.2267% | 2 | 1,651.30 |
| BBQ-8B | `2243514` | **PASS** | 1,147 | 86.9598% | 2 | 1,232.50 |
| BBQ-32B | `2243512` | **PASS** | 1,176 | 89.1585% | 0 | 930.61 |
| K2-MoVA-36B | `2243660` | **PASS** | 1,205 | 91.3571% | 0 | 856.12 |
| K2-MoE-375B | `2243663` | **PASS** | 1,210 | 91.7362% | 0 | 559.61 |

Dataset artifact:
`/mnt/weka/shrd/k2m/lingjie.chen/eval/junlin_merged_7b_linear_b1_fullsuite_20260821/gsm8k/generations.jsonl`
with SHA-256
`785c055981d4313d64120282c69a1259db8057817a1708cbb5842ab072aa2af4`.

## Production serving decode

The workload is random token IDs, greedy decoding, 512 input + 512 output
tokens, infinite offered request rate, eight warmups, radix cache enabled, CUDA
graphs enabled, FA3 attention, and BF16 weights/KV cache.

| Model | Job | TP | Requests / concurrency | Output tok/s | Total tok/s | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBQ-4B | `2243343` | 1 | 256 / 64 | 7,508.62 | 15,017.24 | 341.09 / 547.25 | 7.369 / 7.979 |
| BBQ-8B | `2243344` | 1 | 256 / 64 | 5,816.85 | 11,633.69 | 553.55 / 886.71 | 9.105 / 10.219 |
| BBQ-32B | `2243345` | 2 | 128 / 32 | 1,796.51 | 3,593.03 | 1,048.27 / 1,206.30 | 15.554 / 16.459 |
| K2-MoVA-36B | `2243661` | 2 | 128 / 32 | 1,856.25 | 3,712.51 | 461.77 / 519.52 | 16.219 / 16.982 |
| K2-MoE-375B | `2243662` | 8 | 64 / 8 | 566.79 | 1,133.57 | 213.71 / 216.39 | 13.693 / 14.413 |

## Evidence locations

All suffixes below are beneath the exact artifact root
`/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/models`.

| Model | Probe JSON | HF parity JSON | Full GSM8K JSON | Serving summary JSON |
|---|---|---|---|---|
| BBQ-4B | `bbq-4b/job-2243123/sglang-probe.json` | `bbq-4b/hf-job-2243247/hf-parity.json` | `bbq-4b/gsm8k-job-2243513/gsm8k-sglang.json` | `bbq-4b/decode-job-2243343/decode-serving-summary.json` |
| BBQ-8B | `bbq-8b/job-2243124/sglang-probe.json` | `bbq-8b/hf-job-2243248/hf-parity.json` | `bbq-8b/gsm8k-job-2243514/gsm8k-sglang.json` | `bbq-8b/decode-job-2243344/decode-serving-summary.json` |
| BBQ-32B | `bbq-32b/job-2243125/sglang-probe.json` | `bbq-32b/hf-job-2243249/hf-parity.json` | `bbq-32b/gsm8k-job-2243512/gsm8k-sglang.json` | `bbq-32b/decode-job-2243345/decode-serving-summary.json` |
| K2-MoVA-36B | `k2mova-36b/job-2243439/sglang-probe.json` | `k2mova-36b/hf-job-2243740/hf-parity.json` | `k2mova-36b/gsm8k-job-2243660/gsm8k-sglang.json` | `k2mova-36b/decode-job-2243661/decode-serving-summary.json` |
| K2-MoE-375B | `k2moe-375b/job-2243314/sglang-probe.json` | `k2moe-375b/hf-job-2244882/hf-parity.json` | `k2moe-375b/gsm8k-job-2243663/gsm8k-sglang.json` | `k2moe-375b/decode-job-2243662/decode-serving-summary.json` |

### Read-only manifest audit

Every accepted completed evidence directory has both
`checkpoint-stat-before.txt` and `checkpoint-stat-after.txt`, with an identical
pair:

- probes: jobs `2243123`, `2243124`, `2243125`, `2243439`, `2243314`;
- HF parity: jobs `2243247`, `2243248`, `2243249`, `2243740`, `2244882`;
- full GSM8K: jobs `2243513`, `2243514`, `2243512`, `2243660`, `2243663`;
- production serving: jobs `2243343`, `2243344`, `2243345`, `2243661`,
  `2243662`.

The manifest filenames live in the same exact artifact directory as each JSON
listed above. Wrapper `status.txt` for job `2244882` is also `PASS`. Failed job
`2244521` records an identical pair because its guard fired before weight I/O;
failed jobs `2243655` and `2243741` have only their before manifest and are not
counted as accepted completed evidence.

## Implementation and unit validation

| Job | Scope | Result | Exact artifact path |
|---:|---|---|---|
| `2244520` | Consolidated adapter/config/loader/benchmark suite | **PASS:** 111 passed, 1 skipped, 8 subtests; Ruff and Black checks pass | `/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/unit/job-2244520` |
| `2244576` | Focused exact 375B partial-RoPE guard tests | **PASS:** 5 passed, 4 subtests; Ruff and Black checks pass | `/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/moe375-gate-test-2244576.out` |
| `2244883` | Consolidated suite after the corrected normalized-config guard | **PASS:** 112 passed, 2 skipped, 12 subtests; Ruff and Black checks pass | `/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/unit/job-2244883` |
| `2245010` | Final suite after provenance/rate hardening and expanded formatting scope | **PASS:** 117 passed, 2 skipped, 19 subtests; Ruff and Black checks pass | `/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/unit/job-2245010` |

Provenance coverage varies by harness. Where captured, artifacts retain base
SGLang commit `198d611c005b789879ac73732e67d0d55d6362fd`, working-tree status,
checkpoint metadata, copied run scripts, and recorded source hashes. HF
directories copy their exact `reference.py`/`run.slurm` and checkpoint manifests
but do not directly record git commit/status. Historical probes hash only
`xllm.py` and `mova.py`; dense GSM8K records the evaluator plus xLLM and loader;
later large-model GSM8K records the evaluator plus all four critical runtime
files. The current harness now fails closed unless it resolves the exact local
source tree and all four critical files. Consult each artifact's own fields and
copied scripts rather than assuming uniform coverage. The working tree was not
clean across this evidence sequence, so a clean final release-branch rerun
remains desirable after the implementation is committed.

## Release caveats and required follow-up

- **1B is untested:** obtain the exact release checkpoint path before scheduling
  probe, HF parity, GSM8K, and production serving decode jobs.
- **375B checkpoint identity is provisional:** `checkpoint_0006000` was selected
  as the numerically latest visible child, but the checkpoint owner must confirm
  that it is the release artifact. A passing probe must not be transferred to a
  different checkpoint without rerunning.
- **36B checkpoint identity is also not declared final here:** evidence is tied
  specifically to `checkpoint_0010000`.
- **Artifact topology is not source-router topology:** the canonical MoVA
  artifact records 64 attention value experts/top-4 and 100 FFN experts/top-8,
  but it does not encode the router GEMM partition topology used by the source
  training run. The MoVA jobs explicitly use
  `xllm_source_router_gemm_partitions=2`; this is deliberately independent of
  runtime TP and requires training-owner confirmation.
- **375B has a separate artifact-vs-source MP8 caveat:** its source training
  topology is MP8, but the exported HF artifact does not preserve that router
  GEMM partition provenance. The accepted SGLang jobs use the artifact-compatible
  legacy no-override behavior at serving TP8; serving TP8 must not be interpreted
  as proof of source-MP8 router rounding parity. Corrected HF job `2244882`
  passes artifact parity; it does not establish source-training MP8 router
  rounding parity.
- **Keep RoPE fusion disabled for MoVA.** The SGLang launch records show
  `enable_fused_qk_norm_rope=false`; do not enable that fusion in the release
  recipe. This corresponds to retaining `--no-rope-fusion` in the originating
  MoVA workflow.
- **Do not apply the suggested tokenizer regex repair.** Transformers emits the
  Mistral-regex warning for these saved tokenizers, but it is a false positive
  for the intentional Jais250k BPE contract. Loading with
  `fix_mistral_regex=True` changes the tokenizer mapping and breaks the recorded
  prompt/token-ID contract; use the checkpoint tokenizer unchanged.
- **Large-model throughput is untuned.** H200 Triton MoE tuning files were
  missing for MoVA `E=64,N=512`, MoVA `E=100,N=384` (including the down path),
  and 375B `E=192,N=224` (including the down path). The current numbers use
  fallback kernel configurations and should be treated as conservative rather
  than peak H200 performance.
- **On-disk FP32 is intentional but expensive:** all observed checkpoint shards
  are physically FP32 while serving and HF reference execution are explicitly
  BF16. Capacity/startup planning must use the physical footprint for I/O and
  the BF16 footprint for device residency.
- **The completed 36B and 375B probes used sequential loading** to cap host-memory
  pressure. Their production decode and GSM8K jobs used the default bounded
  loader configuration (`{}`) and completed successfully, providing the relevant
  day-0 loader validation.
- **No merge or push is implied:** all results belong to the private working
  branch and this document is for review only.
