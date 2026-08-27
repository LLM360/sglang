# BBQ native-SGLang GSM8K evaluation

This evaluator runs the raw eight-shot completion prompt used by native xLLM.
It does not apply a chat template and never reads the saved model generations in
the source JSONL. The default quick run evaluates the first 128 canonical test
rows; set `BBQ_GSM8K_N=1319` for the full release result.

Example quick run for the 4B checkpoint:

```bash
sbatch --export=ALL,BBQ_MODEL_NAME=bbq-4b,BBQ_MODEL_PATH=/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-4b-pretrain-final,BBQ_TP_SIZE=1,BBQ_EXPECTED_ARCH=XllmForCausalLM \
  benchmark/llm360_bbq/run_gsm8k.slurm
```

Example full 32B run:

```bash
sbatch --export=ALL,BBQ_MODEL_NAME=bbq-32b,BBQ_MODEL_PATH=/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-32b-pretrain-final,BBQ_TP_SIZE=2,BBQ_EXPECTED_ARCH=XllmForCausalLM,BBQ_GSM8K_N=1319 \
  benchmark/llm360_bbq/run_gsm8k.slurm
```

The result is
`artifacts/models/<model>/gsm8k-job-<job-id>/gsm8k-sglang.json`. It contains
every question, completion, extracted answer, correctness decision, token
counts, accuracy, invalid count, throughput, checkpoint metadata hashes, data
hash, evaluator/source hashes, and the SGLang commit.

`PASS` includes a deliberately loose gross-correctness guard: accuracy must be
at least 50% and invalid answers at most 5%. This is intended to catch a wrong
model or corrupt load, not to claim baseline parity. Override the thresholds
with `BBQ_GSM8K_MIN_ACCURACY` and `BBQ_GSM8K_MAX_INVALID_FRACTION`; always
report the measured accuracy separately.

The wrapper mounts both the checkpoint and GSM8K source directory read-only and
compares a name/size/mtime/mode manifest of every top-level checkpoint file
before and after inference. `.bin` loading uses native SGLang in BF16 with
the production bounded-multithread/mmap loader. Set
`BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON='{"enable_multithread_load": false}'` only
for a diagnostic sequential-load fallback.

Run the CPU-only prompt/scorer/source tests with:

```bash
python3 -m unittest benchmark/llm360_bbq/test_gsm8k_sglang.py
```

Caveats:

- The quick subset is the first 128 rows in canonical order, not a random
  sample. This makes repeated and cross-model comparisons exact.
- One BOS token is added explicitly after tokenizing with
  `add_special_tokens=False`, matching native xLLM. The checkpoint tokenizer is
  used as saved; no chat template or tokenizer regex repair is applied.
- Scoring deliberately mirrors xLLM's heuristic final-number extraction and
  numeric equivalence. It is a GSM8K parity metric, not a general symbolic math
  verifier.
- Reported throughput is offline Engine generation throughput. Use the separate
  serving benchmark for production request-concurrency throughput.
