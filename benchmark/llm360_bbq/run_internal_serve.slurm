#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=1400G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --job-name=bbq-internal-serve
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%A_%a.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly EXPECTED_IMAGE_SIZE=40458514432
readonly WORKER=${REPO_ROOT}/benchmark/llm360_bbq/run_internal_serve_worker.sh
readonly REGISTRY_ROOT=${PROJECT_ROOT}/artifacts/internal-serving

# Keep this matrix tied to exact HF-layout checkpoints whose native-SGLang
# serving recipes have already passed the release verification harness.
readonly -a MODEL_SLUGS=(
  bbq-1b
  bbq-4b
  bbq-7b
  bbq-32b
  k2-mova-36b
  k2-moe-375b
)
readonly -a SERVED_MODEL_NAMES=(
  bbq-1b
  bbq-4b
  bbq-7b
  bbq-32b
  k2-mova-36b
  k2-moe-375b
)
readonly -a MODEL_PATHS=(
  /mnt/weka/shrd/k2m/junlin.chen/xllm_1b_final/model
  /mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-4b-pretrain-final
  /mnt/weka/shrd/k2m/junlin.chen/ckpts/k2v3-7B_iso_attn_shared_small_phase1_sft_2199941/huggingface/checkpoint_0010000
  /mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-32b-pretrain-final
  /mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000
  /mnt/weka/home/mrunner/workspace/checkpoints/huggingface/k2moe375B_mid4_v2_200B_256nodes_seed42_bsz32M_seq512k_jais250k_ep8_dot_te_bestfit/checkpoints/checkpoint_0006000
)
readonly -a TP_SIZES=(1 1 1 2 2 8)
readonly -a JSON_MODEL_OVERRIDES=(
  '{}'
  '{}'
  '{}'
  '{}'
  '{"xllm_source_router_gemm_partitions":2}'
  '{}'
)
readonly -a LOADER_CONFIGS=(
  '{}'
  '{"enable_multithread_load":false}'
  '{}'
  '{"enable_multithread_load":false}'
  '{}'
  '{}'
)
readonly -a MAX_TOTAL_TOKENS=('' '' '' '' 32768 32768)
readonly -a CHAT_CAPABLE=(true false true false true true)

readonly task_id=${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array}
if (( task_id < 0 || task_id >= ${#MODEL_SLUGS[@]} )); then
  printf 'SLURM_ARRAY_TASK_ID must be between 0 and %s (got %s)\n' \
    "$((${#MODEL_SLUGS[@]} - 1))" "${task_id}" >&2
  exit 2
fi

readonly model_slug=${MODEL_SLUGS[task_id]}
readonly served_model_name=${SERVED_MODEL_NAMES[task_id]}
model_path=$(realpath -e -- "${MODEL_PATHS[task_id]}")
readonly model_path
readonly tp_size=${TP_SIZES[task_id]}
readonly json_model_override_args=${JSON_MODEL_OVERRIDES[task_id]}
readonly model_loader_extra_config=${LOADER_CONFIGS[task_id]}
readonly max_total_tokens=${MAX_TOTAL_TOKENS[task_id]}
readonly chat_capable=${CHAT_CAPABLE[task_id]}
readonly port_base=${BBQ_SERVE_PORT_BASE:-31000}
readonly port=$((port_base + task_id))
readonly ready_timeout=${BBQ_SERVE_READY_TIMEOUT:-1800}
readonly health_interval=${BBQ_SERVE_HEALTH_INTERVAL:-30}
readonly api_key_file=${BBQ_SERVE_API_KEY_FILE:-}
readonly array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
readonly result_root=${REGISTRY_ROOT}/jobs/${array_job_id}_${task_id}-${model_slug}
readonly endpoint_file=${REGISTRY_ROOT}/endpoints/${model_slug}.json
readonly job_endpoint_file=${result_root}/endpoint.json

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

test -f "${WORKER}"
test -f "${IMAGE}"
test "$(stat -c %s -- "${IMAGE}")" = "${EXPECTED_IMAGE_SIZE}"
test -d "${model_path}"
test -f "${model_path}/config.json"
test -f "${model_path}/tokenizer.json"
if [[ ${chat_capable} == true ]]; then
  test -f "${model_path}/chat_template.jinja"
fi
if [[ ${model_path} == *:* || ${model_path} == *,* || ${model_path} == *$'\n'* ]]; then
  printf 'Checkpoint path cannot contain colon, comma, or newline: %q\n' \
    "${model_path}" >&2
  exit 2
fi
if ! is_positive_integer "${ready_timeout}" || ! is_positive_integer "${health_interval}"; then
  printf 'BBQ_SERVE_READY_TIMEOUT and BBQ_SERVE_HEALTH_INTERVAL must be positive integers\n' >&2
  exit 2
fi
if (( port < 1024 || port > 65535 )); then
  printf 'Derived port is outside 1024..65535: %s\n' "${port}" >&2
  exit 2
fi
python3 - "${json_model_override_args}" "${model_loader_extra_config}" <<'PY'
import json
import sys

for name, raw in zip(("model override", "loader config"), sys.argv[1:]):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
PY

if [[ -n ${api_key_file} ]]; then
  resolved_api_key_file=$(realpath -e -- "${api_key_file}")
  readonly resolved_api_key_file
  if [[ ${resolved_api_key_file} != "${PROJECT_ROOT}/"* ]]; then
    printf 'BBQ_SERVE_API_KEY_FILE must be below %s so the container can read it\n' \
      "${PROJECT_ROOT}" >&2
    exit 2
  fi
  test -f "${resolved_api_key_file}"
  test -r "${resolved_api_key_file}"
  if [[ ! -s ${resolved_api_key_file} ]]; then
    printf 'API-key file is empty: %s\n' "${resolved_api_key_file}" >&2
    exit 2
  fi
else
  readonly resolved_api_key_file=
fi

if [[ ${BBQ_SERVE_VALIDATE_ONLY:-0} == 1 ]]; then
  python3 - \
    "${task_id}" \
    "${model_slug}" \
    "${served_model_name}" \
    "${model_path}" \
    "${tp_size}" \
    "${port}" \
    "${json_model_override_args}" \
    "${model_loader_extra_config}" \
    "${max_total_tokens}" \
    "${chat_capable}" <<'PY'
import json
import sys

(
    task_id,
    model_slug,
    served_model_name,
    model_path,
    tp_size,
    port,
    model_override,
    loader_config,
    max_total_tokens,
    chat_capable,
) = sys.argv[1:]
print(json.dumps({
    "task_id": int(task_id),
    "model": model_slug,
    "served_model_name": served_model_name,
    "checkpoint": model_path,
    "tp_size": int(tp_size),
    "port": int(port),
    "json_model_override_args": json.loads(model_override),
    "model_loader_extra_config": json.loads(loader_config),
    "max_total_tokens": int(max_total_tokens) if max_total_tokens else None,
    "chat_completions": chat_capable == "true",
}, sort_keys=True))
PY
  exit 0
fi

mkdir -p "${result_root}" "${REGISTRY_ROOT}/endpoints"
cp -- "$0" "${result_root}/run.slurm"
cp -- "${WORKER}" "${result_root}/run-worker.sh"
git -C "${REPO_ROOT}" rev-parse HEAD > "${result_root}/sglang-commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${result_root}/sglang-status.txt"
sglang_commit=$(<"${result_root}/sglang-commit.txt")
readonly sglang_commit

checkpoint_manifest() {
  find "${model_path}" -xdev \
    -printf '%y\t%P\t%s\t%T@\t%i\t%l\n' \
    | LC_ALL=C sort
}
checkpoint_manifest > "${result_root}/checkpoint-stat-before.txt"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export HF_HOME=${result_root}/cache/hf
export XDG_CACHE_HOME=${result_root}/cache/xdg
export TRITON_CACHE_DIR=${result_root}/cache/triton
export TORCHINDUCTOR_CACHE_DIR=${result_root}/cache/torchinductor
export TORCH_EXTENSIONS_DIR=${result_root}/cache/torch-extensions
export SGLANG_CACHE_DIR=${result_root}/cache/sglang
export SGLANG_DG_CACHE_DIR=${result_root}/cache/deep-gemm

export BBQ_RUN_MODEL_SLUG=${model_slug}
export BBQ_RUN_SERVED_MODEL_NAME=${served_model_name}
export BBQ_RUN_MODEL_PATH=${model_path}
export BBQ_RUN_TP_SIZE=${tp_size}
export BBQ_RUN_PORT=${port}
export BBQ_RUN_READY_TIMEOUT=${ready_timeout}
export BBQ_RUN_HEALTH_INTERVAL=${health_interval}
export BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS=${json_model_override_args}
export BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG=${model_loader_extra_config}
export BBQ_RUN_MAX_TOTAL_TOKENS=${max_total_tokens}
export BBQ_RUN_CHAT_CAPABLE=${chat_capable}
export BBQ_RUN_API_KEY_FILE=${resolved_api_key_file}
export BBQ_RUN_REPO_ROOT=${REPO_ROOT}
export BBQ_RUN_RESULT_ROOT=${result_root}
export BBQ_RUN_ENDPOINT_FILE=${endpoint_file}
export BBQ_RUN_JOB_ENDPOINT_FILE=${job_endpoint_file}
export BBQ_RUN_SGLANG_COMMIT=${sglang_commit}

set +e
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${model_path}:${model_path}:ro" \
  --container-workdir="${result_root}" \
  bash "${result_root}/run-worker.sh"
run_status=$?
set -e

checkpoint_manifest > "${result_root}/checkpoint-stat-after.txt"
if ! cmp --silent \
  "${result_root}/checkpoint-stat-before.txt" \
  "${result_root}/checkpoint-stat-after.txt"; then
  diff -u \
    "${result_root}/checkpoint-stat-before.txt" \
    "${result_root}/checkpoint-stat-after.txt" \
    > "${result_root}/checkpoint-stat-diff.txt" || true
  printf 'Checkpoint metadata changed during serving despite the read-only bind mount\n' >&2
  exit 97
fi

exit "${run_status}"
