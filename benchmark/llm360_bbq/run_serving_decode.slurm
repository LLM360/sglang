#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=1400G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --job-name=bbq-sglang-decode
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.err

set -euo pipefail

: "${BBQ_MODEL_NAME:?Set BBQ_MODEL_NAME to an artifact-safe label}"
: "${BBQ_MODEL_PATH:?Set BBQ_MODEL_PATH to the checkpoint directory}"
: "${BBQ_TP_SIZE:?Set BBQ_TP_SIZE}"

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly WORKER=${REPO_ROOT}/benchmark/llm360_bbq/run_serving_decode_worker.sh
readonly SUMMARIZER=${REPO_ROOT}/benchmark/llm360_bbq/summarize_serving_decode.py
readonly CHECKPOINT_LAYOUT=${REPO_ROOT}/benchmark/llm360_bbq/checkpoint_layout.py

if [[ ! ${BBQ_MODEL_NAME} =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'BBQ_MODEL_NAME may contain only letters, digits, dot, underscore, and hyphen\n' >&2
  exit 2
fi
MODEL_PATH=$(realpath -e -- "${BBQ_MODEL_PATH}")
readonly MODEL_PATH
if [[ ! -d ${MODEL_PATH} || ! -f ${MODEL_PATH}/config.json ]]; then
  printf 'Checkpoint must be a directory containing config.json: %s\n' "${MODEL_PATH}" >&2
  exit 2
fi
if [[ ${MODEL_PATH} == *:* || ${MODEL_PATH} == *,* || ${MODEL_PATH} == *$'\n'* ]]; then
  printf 'Checkpoint path cannot contain colon, comma, or newline for a container bind mount\n' >&2
  exit 2
fi

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

readonly TP_SIZE=${BBQ_TP_SIZE}
readonly NUM_REQUESTS=${BBQ_NUM_REQUESTS:-64}
readonly MAX_CONCURRENCY=${BBQ_MAX_CONCURRENCY:-8}
readonly WARMUP_REQUESTS=${BBQ_WARMUP_REQUESTS:-8}
readonly SEED=${BBQ_SEED:-20260827}
readonly READY_TIMEOUT=${BBQ_SERVER_READY_TIMEOUT:-7200}
readonly MEM_FRACTION_STATIC=${BBQ_MEM_FRACTION_STATIC:-0.85}
readonly PORT=${BBQ_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
JSON_MODEL_OVERRIDE_ARGS=${BBQ_JSON_MODEL_OVERRIDE_ARGS:-}
if [[ -z ${JSON_MODEL_OVERRIDE_ARGS} ]]; then
  JSON_MODEL_OVERRIDE_ARGS='{}'
fi
readonly JSON_MODEL_OVERRIDE_ARGS
MODEL_LOADER_EXTRA_CONFIG_JSON=${BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON:-}
if [[ -z ${MODEL_LOADER_EXTRA_CONFIG_JSON} ]]; then
  MODEL_LOADER_EXTRA_CONFIG_JSON='{}'
fi
readonly MODEL_LOADER_EXTRA_CONFIG_JSON

for pair in \
  "TP_SIZE:${TP_SIZE}" \
  "NUM_REQUESTS:${NUM_REQUESTS}" \
  "MAX_CONCURRENCY:${MAX_CONCURRENCY}" \
  "WARMUP_REQUESTS:${WARMUP_REQUESTS}" \
  "READY_TIMEOUT:${READY_TIMEOUT}"; do
  if ! is_positive_integer "${pair#*:}"; then
    printf '%s must be a positive integer (got %q)\n' "${pair%%:*}" "${pair#*:}" >&2
    exit 2
  fi
done
if [[ ! ${SEED} =~ ^[0-9]+$ ]]; then
  printf 'BBQ_SEED must be a non-negative integer\n' >&2
  exit 2
fi
if (( TP_SIZE > 8 )); then
  printf 'This single-node H200 runner supports TP <= 8\n' >&2
  exit 2
fi
if (( MAX_CONCURRENCY > NUM_REQUESTS )); then
  printf 'BBQ_MAX_CONCURRENCY cannot exceed BBQ_NUM_REQUESTS\n' >&2
  exit 2
fi
if (( PORT < 1024 || PORT > 65535 )); then
  printf 'BBQ_PORT must be between 1024 and 65535\n' >&2
  exit 2
fi
python3 - \
  "${MEM_FRACTION_STATIC}" \
  "${JSON_MODEL_OVERRIDE_ARGS}" \
  "${MODEL_LOADER_EXTRA_CONFIG_JSON}" <<'PY'
import json
import sys

fraction = float(sys.argv[1])
if not 0.0 < fraction < 1.0:
    raise SystemExit("BBQ_MEM_FRACTION_STATIC must be strictly between 0 and 1")
override = json.loads(sys.argv[2])
if not isinstance(override, dict):
    raise SystemExit("BBQ_JSON_MODEL_OVERRIDE_ARGS must encode a JSON object")
loader = json.loads(sys.argv[3])
if not isinstance(loader, dict):
    raise SystemExit("BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON must encode a JSON object")
PY
if [[ -n ${BBQ_MAX_TOTAL_TOKENS:-} ]] && ! is_positive_integer "${BBQ_MAX_TOTAL_TOKENS}"; then
  printf 'BBQ_MAX_TOTAL_TOKENS must be a positive integer when set\n' >&2
  exit 2
fi

readonly RESULT_ROOT=${PROJECT_ROOT}/artifacts/models/${BBQ_MODEL_NAME}/decode-job-${SLURM_JOB_ID}
mkdir -p "${RESULT_ROOT}"
cp -- "$0" "${RESULT_ROOT}/run.slurm"
cp -- "${WORKER}" "${RESULT_ROOT}/run-worker.sh"
cp -- "${SUMMARIZER}" "${RESULT_ROOT}/summarize.py"
cp -- "${CHECKPOINT_LAYOUT}" "${RESULT_ROOT}/checkpoint_layout.py"
sha256sum -- "${RESULT_ROOT}/checkpoint_layout.py" \
  > "${RESULT_ROOT}/checkpoint-layout-source-sha256.txt"
python3 "${RESULT_ROOT}/checkpoint_layout.py" \
  --model-path "${MODEL_PATH}" \
  > "${RESULT_ROOT}/checkpoint-layout.json"
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/sglang-commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${RESULT_ROOT}/sglang-status.txt"

readonly SOURCE_SNAPSHOT=${RESULT_ROOT}/source-snapshot
readonly -a CORE_SOURCE_RELATIVE=(
  python/sglang/srt/models/xllm.py
  python/sglang/srt/layers/mova.py
  python/sglang/srt/model_loader/weight_utils.py
  python/sglang/srt/utils/hf_transformers_utils.py
  benchmark/llm360_bbq/checkpoint_layout.py
)
mkdir -p "${SOURCE_SNAPSHOT}"
for source_relative in "${CORE_SOURCE_RELATIVE[@]}"; do
  cp -- "${REPO_ROOT}/${source_relative}" \
    "${SOURCE_SNAPSHOT}/$(basename "${source_relative}")"
done
core_source_manifest() {
  local source_set=$1
  local source_path
  local source_relative
  local checksum
  for source_relative in "${CORE_SOURCE_RELATIVE[@]}"; do
    case ${source_set} in
      live)
        source_path=${REPO_ROOT}/${source_relative}
        ;;
      snapshot)
        source_path=${SOURCE_SNAPSHOT}/$(basename "${source_relative}")
        ;;
      *)
        printf 'Unknown core source set: %s\n' "${source_set}" >&2
        return 2
        ;;
    esac
    checksum=$(sha256sum -- "${source_path}")
    printf '%s\t%s\n' "${checksum%% *}" "${source_relative}"
  done
}
core_source_manifest live > "${RESULT_ROOT}/core-source-sha256-before.txt"
core_source_manifest snapshot > "${RESULT_ROOT}/core-source-sha256-snapshot.txt"
if ! cmp --silent \
  "${RESULT_ROOT}/core-source-sha256-before.txt" \
  "${RESULT_ROOT}/core-source-sha256-snapshot.txt"; then
  diff -u \
    "${RESULT_ROOT}/core-source-sha256-before.txt" \
    "${RESULT_ROOT}/core-source-sha256-snapshot.txt" \
    > "${RESULT_ROOT}/core-source-snapshot-diff.txt" || true
  printf 'FAIL: release-critical source snapshot does not match the live tree\n' \
    | tee "${RESULT_ROOT}/status.txt" >&2
  exit 1
fi

checkpoint_manifest() {
  find "${MODEL_PATH}" -xdev \
    -printf '%y\t%P\t%s\t%T@\t%i\t%l\n' \
    | LC_ALL=C sort
}
checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-before.txt"

export BBQ_RUN_RESULT_ROOT=${RESULT_ROOT}
export BBQ_RUN_REPO_ROOT=${REPO_ROOT}
export BBQ_RUN_MODEL_PATH=${MODEL_PATH}
export BBQ_RUN_TP_SIZE=${TP_SIZE}
export BBQ_RUN_NUM_REQUESTS=${NUM_REQUESTS}
export BBQ_RUN_MAX_CONCURRENCY=${MAX_CONCURRENCY}
export BBQ_RUN_WARMUP_REQUESTS=${WARMUP_REQUESTS}
export BBQ_RUN_SEED=${SEED}
export BBQ_RUN_PORT=${PORT}
export BBQ_RUN_READY_TIMEOUT=${READY_TIMEOUT}
export BBQ_RUN_MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}
export BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS=${JSON_MODEL_OVERRIDE_ARGS}
export BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG_JSON=${MODEL_LOADER_EXTRA_CONFIG_JSON}
export BBQ_MODEL_NAME
export BBQ_RUN_SGLANG_COMMIT
BBQ_RUN_SGLANG_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
if [[ -n ${BBQ_MAX_TOTAL_TOKENS:-} ]]; then
  export BBQ_RUN_MAX_TOTAL_TOKENS=${BBQ_MAX_TOTAL_TOKENS}
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/hf-bbq-decode-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg-bbq-decode-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-bbq-decode-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-bbq-decode-${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch-extensions-bbq-decode-${SLURM_JOB_ID}
export SGLANG_CACHE_DIR=/tmp/sglang-bbq-decode-${SLURM_JOB_ID}
export SGLANG_DG_CACHE_DIR=/tmp/deep-gemm-bbq-decode-${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

set +e
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${MODEL_PATH}:${MODEL_PATH}:ro" \
  --container-workdir="${RESULT_ROOT}" \
  bash "${RESULT_ROOT}/run-worker.sh"
run_status=$?
set -e

core_source_manifest live > "${RESULT_ROOT}/core-source-sha256-after.txt"
core_source_manifest snapshot > "${RESULT_ROOT}/core-source-sha256-snapshot-after.txt"
checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-after.txt"
if ! cmp --silent \
  "${RESULT_ROOT}/checkpoint-stat-before.txt" \
  "${RESULT_ROOT}/checkpoint-stat-after.txt"; then
  diff -u \
    "${RESULT_ROOT}/checkpoint-stat-before.txt" \
    "${RESULT_ROOT}/checkpoint-stat-after.txt" \
    > "${RESULT_ROOT}/checkpoint-stat-diff.txt" || true
  printf 'FAIL: checkpoint metadata changed despite the read-only bind mount\n' \
    | tee "${RESULT_ROOT}/status.txt" >&2
  exit 1
fi
if ! cmp --silent \
  "${RESULT_ROOT}/core-source-sha256-before.txt" \
  "${RESULT_ROOT}/core-source-sha256-after.txt"; then
  diff -u \
    "${RESULT_ROOT}/core-source-sha256-before.txt" \
    "${RESULT_ROOT}/core-source-sha256-after.txt" \
    > "${RESULT_ROOT}/core-source-sha256-diff.txt" || true
  printf 'FAIL: release-critical SGLang source changed during the run\n' \
    | tee "${RESULT_ROOT}/status.txt" >&2
  exit 1
fi
if ! cmp --silent \
  "${RESULT_ROOT}/core-source-sha256-snapshot.txt" \
  "${RESULT_ROOT}/core-source-sha256-snapshot-after.txt"; then
  diff -u \
    "${RESULT_ROOT}/core-source-sha256-snapshot.txt" \
    "${RESULT_ROOT}/core-source-sha256-snapshot-after.txt" \
    > "${RESULT_ROOT}/core-source-snapshot-after-diff.txt" || true
  printf 'FAIL: release-critical source snapshot changed during the run\n' \
    | tee "${RESULT_ROOT}/status.txt" >&2
  exit 1
fi
if (( run_status != 0 )); then
  printf 'FAIL: serving benchmark exited with status %s\n' "${run_status}" \
    | tee "${RESULT_ROOT}/status.txt" >&2
  exit "${run_status}"
fi

test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "${RESULT_ROOT}/decode-serving-summary.json")" = PASS
printf 'PASS\n' > "${RESULT_ROOT}/status.txt"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
