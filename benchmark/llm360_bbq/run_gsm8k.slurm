#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=1400G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --job-name=bbq-gsm8k
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.err

set -euo pipefail

: "${BBQ_MODEL_NAME:?Set BBQ_MODEL_NAME}"
: "${BBQ_MODEL_PATH:?Set BBQ_MODEL_PATH to the direct checkpoint directory}"
: "${BBQ_TP_SIZE:?Set BBQ_TP_SIZE}"

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly EVALUATOR=${REPO_ROOT}/benchmark/llm360_bbq/gsm8k_sglang.py
readonly GSM8K_SOURCE=${BBQ_GSM8K_SOURCE:-/mnt/weka/shrd/k2m/lingjie.chen/eval/junlin_merged_7b_linear_b1_fullsuite_20260821/gsm8k/generations.jsonl}
readonly GSM8K_SOURCE_DIR=${GSM8K_SOURCE%/*}
readonly EXPECTED_GSM8K_SHA256=${BBQ_GSM8K_EXPECTED_SHA256:-785c055981d4313d64120282c69a1259db8057817a1708cbb5842ab072aa2af4}
readonly RESULT_ROOT=${PROJECT_ROOT}/artifacts/models/${BBQ_MODEL_NAME}/gsm8k-job-${SLURM_JOB_ID}
readonly LIMIT=${BBQ_GSM8K_N:-128}
readonly BATCH_SIZE=${BBQ_GSM8K_BATCH_SIZE:-32}
readonly MAX_TOTAL_TOKENS=${BBQ_MAX_TOTAL_TOKENS:-32768}
readonly MAX_RUNNING_REQUESTS=${BBQ_MAX_RUNNING_REQUESTS:-64}
readonly CHUNKED_PREFILL_SIZE=${BBQ_CHUNKED_PREFILL_SIZE:-1024}
readonly CUDA_GRAPH_MAX_BS=${BBQ_CUDA_GRAPH_MAX_BS:-32}
readonly EXPECTED_ARCH=${BBQ_EXPECTED_ARCH:-}
readonly MIN_ACCURACY=${BBQ_GSM8K_MIN_ACCURACY:-0.5}
readonly MAX_INVALID_FRACTION=${BBQ_GSM8K_MAX_INVALID_FRACTION:-0.05}
if [[ -n ${BBQ_MODEL_OVERRIDE_JSON:-} ]]; then
  readonly MODEL_OVERRIDE_JSON=${BBQ_MODEL_OVERRIDE_JSON}
else
  readonly MODEL_OVERRIDE_JSON='{}'
fi
if [[ -n ${BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON:-} ]]; then
  readonly MODEL_LOADER_EXTRA_CONFIG_JSON=${BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON}
else
  readonly MODEL_LOADER_EXTRA_CONFIG_JSON='{}'
fi

test -f "${BBQ_MODEL_PATH}/config.json"
test -f "${BBQ_MODEL_PATH}/pytorch_model.bin.index.json"
test -f "${BBQ_MODEL_PATH}/tokenizer.json"
test -f "${GSM8K_SOURCE}"
test "$(sha256sum "${GSM8K_SOURCE}" | awk '{print $1}')" = "${EXPECTED_GSM8K_SHA256}"

mkdir -p "${RESULT_ROOT}"
cp "$0" "${RESULT_ROOT}/run.slurm"
cp "${EVALUATOR}" "${RESULT_ROOT}/gsm8k_sglang.py"
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/sglang-commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${RESULT_ROOT}/sglang-status.txt"

checkpoint_manifest() {
  find "${BBQ_MODEL_PATH}" -mindepth 1 -maxdepth 1 \
    -printf '%f\t%y\t%s\t%T@\t%m\t%l\n' | LC_ALL=C sort
}

checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-before.txt"
on_exit() {
  local original_status=$?
  trap - EXIT
  checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-after.txt"
  if ! cmp "${RESULT_ROOT}/checkpoint-stat-before.txt" "${RESULT_ROOT}/checkpoint-stat-after.txt"; then
    printf 'Checkpoint metadata changed during a read-only evaluation.\n' >&2
    exit 97
  fi
  exit "${original_status}"
}
trap on_exit EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/hf-bbq-gsm8k-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-bbq-gsm8k-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-bbq-gsm8k-${SLURM_JOB_ID}
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export BBQ_RUN_MODEL_OVERRIDE_JSON=${MODEL_OVERRIDE_JSON}
export BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG_JSON=${MODEL_LOADER_EXTRA_CONFIG_JSON}

readonly SGLANG_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${BBQ_MODEL_PATH}:${BBQ_MODEL_PATH}:ro,${GSM8K_SOURCE_DIR}:${GSM8K_SOURCE_DIR}:ro" \
  --container-workdir="${RESULT_ROOT}" \
  bash -lc '
    set -euo pipefail
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    nvidia-smi -L | tee "'"${RESULT_ROOT}"'/gpu-inventory.txt"
    python3 -u "'"${RESULT_ROOT}"'/gsm8k_sglang.py" \
      --model-path "'"${BBQ_MODEL_PATH}"'" \
      --model-name "'"${BBQ_MODEL_NAME}"'" \
      --output-dir "'"${RESULT_ROOT}"'" \
      --data-path "'"${GSM8K_SOURCE}"'" \
      --expected-data-sha256 "'"${EXPECTED_GSM8K_SHA256}"'" \
      --limit "'"${LIMIT}"'" \
      --tp-size "'"${BBQ_TP_SIZE}"'" \
      --batch-size "'"${BATCH_SIZE}"'" \
      --max-total-tokens "'"${MAX_TOTAL_TOKENS}"'" \
      --max-running-requests "'"${MAX_RUNNING_REQUESTS}"'" \
      --chunked-prefill-size "'"${CHUNKED_PREFILL_SIZE}"'" \
      --cuda-graph-max-bs "'"${CUDA_GRAPH_MAX_BS}"'" \
      --expected-arch "'"${EXPECTED_ARCH}"'" \
      --model-override-json "${BBQ_RUN_MODEL_OVERRIDE_JSON}" \
      --model-loader-extra-config-json "${BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG_JSON}" \
      --min-accuracy "'"${MIN_ACCURACY}"'" \
      --max-invalid-fraction "'"${MAX_INVALID_FRACTION}"'" \
      --sglang-source-root "'"${REPO_ROOT}"'" \
      --sglang-commit "'"${SGLANG_COMMIT}"'" \
      2>&1 | tee "'"${RESULT_ROOT}"'/gsm8k.log"
  '

test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RESULT_ROOT}/gsm8k-sglang.json")" = PASS
printf 'PASS\n' > "${RESULT_ROOT}/status.txt"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
