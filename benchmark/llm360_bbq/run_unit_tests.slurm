#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=bbq-sglang-unit
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly RESULT_ROOT=${PROJECT_ROOT}/artifacts/unit/job-${SLURM_JOB_ID}

mkdir -p "${RESULT_ROOT}"
cp "$0" "${RESULT_ROOT}/run.slurm"
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/sglang-commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${RESULT_ROOT}/sglang-status.txt"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}/python
export HF_HOME=/tmp/hf-bbq-unit-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton-bbq-unit-${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-bbq-unit-${SLURM_JOB_ID}

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${REPO_ROOT}" \
  bash -lc '
    set -euo pipefail
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
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
    python3 -m pytest -q -p no:cacheprovider \
      python/sglang/test/test_xllm_model.py \
      test/srt/cpu/test_topk.py::TestTopK::test_native_biased_sigmoid_routes_in_fp32 \
      test/registered/unit/model_loader/test_weight_utils.py \
      test/registered/unit/layers/test_mova.py \
      benchmark/llm360_bbq/test_checkpoint_layout.py \
      benchmark/llm360_bbq/test_gsm8k_sglang.py \
      benchmark/llm360_bbq/test_probe_sglang.py \
      benchmark/llm360_bbq/test_reference_hf.py \
      benchmark/llm360_bbq/test_summarize_serving_decode.py \
      2>&1 | tee "'"${RESULT_ROOT}"'/pytest.log"
    python3 -m ruff check --select F401,F821 \
      python/sglang/srt/model_loader/weight_utils.py \
      python/sglang/srt/models/xllm.py \
      python/sglang/srt/utils/hf_transformers_utils.py \
      python/sglang/test/test_xllm_model.py \
      test/registered/unit/layers/test_mova.py \
      test/registered/unit/model_loader/test_weight_utils.py \
      benchmark/llm360_bbq/*.py \
      2>&1 | tee "'"${RESULT_ROOT}"'/ruff.log"
    python3 -m black --check \
      python/sglang/srt/utils/hf_transformers_utils.py \
      python/sglang/srt/models/xllm.py \
      python/sglang/test/test_xllm_model.py \
      test/registered/unit/layers/test_mova.py \
      test/registered/unit/model_loader/test_weight_utils.py \
      benchmark/llm360_bbq/*.py \
      2>&1 | tee "'"${RESULT_ROOT}"'/black.log"
    bash -n benchmark/llm360_bbq/*.sh benchmark/llm360_bbq/*.slurm
    git diff --check
  '

printf 'PASS\n' > "${RESULT_ROOT}/status.txt"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
