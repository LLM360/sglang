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
#SBATCH --job-name=bbq-sglang-probe
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.err

set -euo pipefail

: "${BBQ_MODEL_NAME:?Set BBQ_MODEL_NAME}"
: "${BBQ_MODEL_PATH:?Set BBQ_MODEL_PATH to a directory containing config.json}"
: "${BBQ_TP_SIZE:?Set BBQ_TP_SIZE}"

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly PROBE=${REPO_ROOT}/benchmark/llm360_bbq/probe_sglang.py
readonly RESULT_ROOT=${PROJECT_ROOT}/artifacts/models/${BBQ_MODEL_NAME}/job-${SLURM_JOB_ID}

test -f "${BBQ_MODEL_PATH}/config.json"
test -f "${BBQ_MODEL_PATH}/pytorch_model.bin.index.json"
mkdir -p "${RESULT_ROOT}"
cp "$0" "${RESULT_ROOT}/run.slurm"
cp "${PROBE}" "${RESULT_ROOT}/probe.py"
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

export BBQ_OUTPUT_DIR=${RESULT_ROOT}
export BBQ_SGLANG_SOURCE_ROOT=${REPO_ROOT}
export BBQ_SGLANG_COMMIT
BBQ_SGLANG_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/hf-bbq-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-bbq-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-bbq-${SLURM_JOB_ID}
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${BBQ_MODEL_PATH}:${BBQ_MODEL_PATH}:ro" \
  --container-workdir="${RESULT_ROOT}" \
  bash -lc '
    set -euo pipefail
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    nvidia-smi -L | tee "'"${RESULT_ROOT}"'/gpu-inventory.txt"
    python3 - <<"PY" | tee "'"${RESULT_ROOT}"'/environment.txt"
import sys
import torch
import transformers
import sglang

print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
print("sglang_file", sglang.__file__)
print("gpu_count", torch.cuda.device_count())
PY
    python3 -u "'"${RESULT_ROOT}"'/probe.py" 2>&1 \
      | tee "'"${RESULT_ROOT}"'/probe.log"
  '

test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RESULT_ROOT}/sglang-probe.json")" = PASS
printf 'PASS\n' > "${RESULT_ROOT}/status.txt"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
