#!/usr/bin/env bash

set -euo pipefail

: "${BBQ_RUN_RESULT_ROOT:?}"
: "${BBQ_RUN_REPO_ROOT:?}"
: "${BBQ_RUN_MODEL_PATH:?}"
: "${BBQ_RUN_TP_SIZE:?}"
: "${BBQ_RUN_NUM_REQUESTS:?}"
: "${BBQ_RUN_MAX_CONCURRENCY:?}"
: "${BBQ_RUN_WARMUP_REQUESTS:?}"
: "${BBQ_RUN_SEED:?}"
: "${BBQ_RUN_PORT:?}"
: "${BBQ_RUN_READY_TIMEOUT:?}"
: "${BBQ_RUN_MEM_FRACTION_STATIC:?}"
: "${BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS:?}"
: "${BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG_JSON:?}"
: "${BBQ_RUN_SGLANG_COMMIT:?}"
: "${SLURM_JOB_ID:?}"

readonly result_root=${BBQ_RUN_RESULT_ROOT}
readonly repo_root=${BBQ_RUN_REPO_ROOT}
readonly model_path=${BBQ_RUN_MODEL_PATH}
readonly tp_size=${BBQ_RUN_TP_SIZE}
readonly num_requests=${BBQ_RUN_NUM_REQUESTS}
readonly max_concurrency=${BBQ_RUN_MAX_CONCURRENCY}
readonly warmup_requests=${BBQ_RUN_WARMUP_REQUESTS}
readonly seed=${BBQ_RUN_SEED}
readonly port=${BBQ_RUN_PORT}
readonly ready_timeout=${BBQ_RUN_READY_TIMEOUT}
readonly mem_fraction_static=${BBQ_RUN_MEM_FRACTION_STATIC}
readonly json_model_override_args=${BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS}
readonly model_loader_extra_config_json=${BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG_JSON}
readonly base_url=http://127.0.0.1:${port}
readonly raw_result=${result_root}/bench-serving-raw.jsonl
readonly summary_result=${result_root}/decode-serving-summary.json
readonly server_log=${result_root}/server.log

mkdir -p \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${SGLANG_CACHE_DIR}" \
  "${SGLANG_DG_CACHE_DIR}"

# Verify that the resolved checkpoint bind mount is actually read-only. The
# longest matching mount point is the effective mount for a nested bind.
python3 - "${model_path}" <<'PY' | tee "${result_root}/checkpoint-mount.txt"
import os
import sys


def unescape_mount_path(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


target = os.path.realpath(sys.argv[1])
matches = []
with open("/proc/self/mountinfo", encoding="utf-8") as mountinfo:
    for line in mountinfo:
        fields = line.rstrip("\n").split()
        separator = fields.index("-")
        mount_point = unescape_mount_path(fields[4])
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            options = set(fields[5].split(","))
            options.update(fields[separator + 3].split(","))
            matches.append((len(mount_point), mount_point, options))

if not matches:
    raise SystemExit(f"no effective mount found for {target}")
_, mount_point, options = max(matches)
print(f"target={target}")
print(f"effective_mount={mount_point}")
print(f"options={','.join(sorted(options))}")
if "ro" not in options:
    raise SystemExit(f"checkpoint mount is not read-only: {mount_point}")
PY

nvidia-smi -L | tee "${result_root}/gpu-inventory.txt"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} < tp_size )); then
  printf 'TP=%s requires at least that many visible GPUs; found %s\n' \
    "${tp_size}" "${#gpu_names[@]}" >&2
  exit 1
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ ${gpu_name} != *H200* ]]; then
    printf 'Expected an H200 allocation, found GPU %q\n' "${gpu_name}" >&2
    exit 1
  fi
done

python3 - <<'PY' | tee "${result_root}/environment.txt"
import sys

import sglang
import torch
import transformers

print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
print("sglang_file", sglang.__file__)
print("gpu_count", torch.cuda.device_count())
PY

server_pid=
cleanup() {
  local exit_code=${1:-$?}
  local attempt
  trap - EXIT INT TERM
  set +e
  if [[ -n ${server_pid} ]] && kill -0 -- "-${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null
    for ((attempt = 0; attempt < 30; attempt++)); do
      if ! kill -0 -- "-${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null
    fi
    wait "${server_pid}" 2>/dev/null
  fi
  exit "${exit_code}"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

server_cmd=(
  python3 -u -m sglang.launch_server
  --model-path "${model_path}"
  --model-impl sglang
  --trust-remote-code
  --dtype bfloat16
  --kv-cache-dtype bfloat16
  --tp-size "${tp_size}"
  --attention-backend fa3
  --mem-fraction-static "${mem_fraction_static}"
  --host 127.0.0.1
  --port "${port}"
  --json-model-override-args "${json_model_override_args}"
  --model-loader-extra-config "${model_loader_extra_config_json}"
)
if [[ -n ${BBQ_RUN_MAX_TOTAL_TOKENS:-} ]]; then
  server_cmd+=(--max-total-tokens "${BBQ_RUN_MAX_TOTAL_TOKENS}")
fi

{
  printf '%q ' "${server_cmd[@]}"
  printf '\n'
} > "${result_root}/server-command.txt"

setsid "${server_cmd[@]}" > "${server_log}" 2>&1 &
server_pid=$!

readonly readiness_url=${base_url}/v1/models
readonly readiness_start=${SECONDS}
while ! python3 -c \
  'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read()' \
  "${readiness_url}" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    set +e
    wait "${server_pid}"
    server_status=$?
    set -e
    printf 'SGLang server exited before readiness (status=%s)\n' "${server_status}" >&2
    tail -n 200 "${server_log}" >&2
    exit 1
  fi
  if (( SECONDS - readiness_start >= ready_timeout )); then
    printf 'SGLang server was not ready after %ss\n' "${ready_timeout}" >&2
    tail -n 200 "${server_log}" >&2
    exit 1
  fi
  sleep 5
done
printf 'Server ready after %ss\n' "$((SECONDS - readiness_start))" \
  | tee "${result_root}/server-ready.txt"

test ! -e "${raw_result}"
bench_cmd=(
  python3 -u -m sglang.bench_serving
  --backend sglang
  --base-url "${base_url}"
  --ready-check-timeout-sec 0
  --model "${model_path}"
  --tokenizer "${model_path}"
  --dataset-name random-ids
  --tokenize-prompt
  --random-input-len 512
  --random-output-len 512
  --random-range-ratio 1.0
  --num-prompts "${num_requests}"
  --max-concurrency "${max_concurrency}"
  --request-rate inf
  --seed "${seed}"
  --warmup-requests "${warmup_requests}"
  --flush-cache
  --output-file "${raw_result}"
  --output-details
  --disable-tqdm
)
{
  printf '%q ' "${bench_cmd[@]}"
  printf '\n'
} > "${result_root}/benchmark-command.txt"

set +e
"${bench_cmd[@]}" 2>&1 | tee "${result_root}/bench-serving.log"
benchmark_status=${PIPESTATUS[0]}
set -e
if (( benchmark_status != 0 )); then
  printf 'bench_serving failed with status %s\n' "${benchmark_status}" >&2
  exit "${benchmark_status}"
fi

python3 -u "${result_root}/summarize.py" \
  --input "${raw_result}" \
  --output "${summary_result}" \
  --model-name "${BBQ_MODEL_NAME}" \
  --model-path "${model_path}" \
  --tp-size "${tp_size}" \
  --num-requests "${num_requests}" \
  --max-concurrency "${max_concurrency}" \
  --warmup-requests "${warmup_requests}" \
  --seed "${seed}" \
  --input-len 512 \
  --output-len 512 \
  --json-model-override-args "${json_model_override_args}" \
  --model-loader-extra-config-json "${model_loader_extra_config_json}" \
  --sglang-commit "${BBQ_RUN_SGLANG_COMMIT}" \
  --sglang-source-root "${repo_root}" \
  --source-snapshot-root "${result_root}/source-snapshot" \
  --job-id "${SLURM_JOB_ID}" \
  | tee "${result_root}/summary.log"

test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${summary_result}")" = PASS
