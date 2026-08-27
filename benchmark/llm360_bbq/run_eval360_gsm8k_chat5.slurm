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
#SBATCH --job-name=mova-mid4-gsm8k-chat5
#SBATCH --output=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.out
#SBATCH --error=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/artifacts/slurm/%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT=/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang
readonly REPO_ROOT=${PROJECT_ROOT}/sglang
readonly MODEL=/mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000
readonly IMAGE=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-437516ff.sqsh
readonly EXPECTED_IMAGE_SIZE=40458514432
readonly EVAL_BUNDLE=/mnt/weka/shrd/k2m/yash.akhauri/mova_base_eval360/bundles/20260817T195133Z
readonly EVAL360_ROOT=${EVAL_BUNDLE}/eval360-src
readonly EXPECTED_EVAL360_COMMIT=78067a66ba5fd3155c70d2e1262ca6f6a99ae083
readonly EVAL360_ENV=/mnt/weka/home/yash.akhauri/projects/jacobidiff/eval360-venv
readonly EVAL360_PYTHON=${EVAL360_ENV}/bin/python
readonly MODEL_TEMPLATE=${EVAL_BUNDLE}/configs/model_gsm8k.yaml
readonly DATA_TEMPLATE=${EVAL_BUNDLE}/configs/data_gsm8k.yaml
readonly AUDITOR=${REPO_ROOT}/benchmark/llm360_bbq/audit_eval360_gsm8k.py
readonly PINNED_DATA_BLOB=/mnt/weka/shrd/k2m/yash.akhauri/mova_base_eval360/hf_cache/datasets--LLM360--eval-360-sources/blobs/719509c0369ed59a394d1da7d3acfbc54e3c14e2
readonly EXPECTED_CHAT5_SOURCE_SHA256=643b3e1717494e3d76eabd536b6d2473479c1b8ad9b10eecb4a5185c7a35c19d
readonly SERVED_MODEL=mova-mid4-v2-0010000
readonly PORT=30000
readonly BASE_URL=http://127.0.0.1:${PORT}/v1
readonly RESULT_ROOT=${PROJECT_ROOT}/artifacts/models/k2mova-36b/eval360-chat5-job-${SLURM_JOB_ID}
readonly INPUT_DATA=${RESULT_ROOT}/gsm8k-chat5-input.jsonl
readonly MODEL_CONFIG=${RESULT_ROOT}/model_gsm8k_chat5.yaml
readonly DATA_CONFIG=${RESULT_ROOT}/data_gsm8k_chat5.yaml
readonly HF_CACHE=${PROJECT_ROOT}/artifacts/cache/eval360-hf
readonly NLTK_DATA=/mnt/weka/shrd/k2m/yash.akhauri/mova_base_eval360/nltk_data
readonly SGLANG_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)

test -d "${MODEL}"
test -f "${IMAGE}"
test "$(stat -c %s "${IMAGE}")" = "${EXPECTED_IMAGE_SIZE}"
test -x "${EVAL360_PYTHON}"
test -f "${MODEL_TEMPLATE}"
test -f "${DATA_TEMPLATE}"
test -f "${AUDITOR}"
test -f "${PINNED_DATA_BLOB}"
test "$(sha256sum "${PINNED_DATA_BLOB}" | awk '{print $1}')" = "${EXPECTED_CHAT5_SOURCE_SHA256}"
test "$(git -C "${EVAL360_ROOT}" rev-parse HEAD)" = "${EXPECTED_EVAL360_COMMIT}"
test -z "$(git -C "${EVAL360_ROOT}" status --porcelain)"
printf '%s  %s\n' \
  57af5d53abf79249ac739dcc3cdbd32a9cb63e104890a137aa7ffd9b93c41a70 "${EVAL360_ROOT}/scheduler/grader/base_parsers.py" \
  f24f23cf50e87dfaa57d5d522dd90b98cadd3a5d3b9123a1524e7d9996b4c7c6 "${EVAL360_ROOT}/scheduler/grader/math.py" \
  | sha256sum --check
printf '%s  %s\n' \
  9c4c4b949a58d12624d1bed96986447d34385ffae2ab7a13024e2daaf38d7196 "${MODEL}/config.json" \
  ace597379bbed5236acd22e9e9b9adfad10673997ab6af5eff5689c8cbe3fa90 "${MODEL}/pytorch_model.bin.index.json" \
  2fa69519ff1ee3ee55ac50927b00058e88b9eb90802fabaf3c6b05804601c53b "${MODEL}/tokenizer.json" \
  581375b2ce8d56d59f609f08a7165365ac53b6543f3442ef0117b4951b0d747c "${MODEL}/tokenizer_config.json" \
  2da7d47641f4509da4ae47711e31d8b5f0f3f801ee08d87e7e9f07f814bdc4a3 "${MODEL}/generation_config.json" \
  a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590 "${MODEL}/chat_template.jinja" \
  | sha256sum --check

mkdir -p \
  "${RESULT_ROOT}/eval-output" \
  "${RESULT_ROOT}/eval360-logs" \
  "${RESULT_ROOT}/server-caches" \
  "${RESULT_ROOT}/source-snapshot" \
  "${HF_CACHE}"
cp "$0" "${RESULT_ROOT}/run.slurm"
cp "${AUDITOR}" "${RESULT_ROOT}/audit_eval360_gsm8k.py"
cp "${MODEL_TEMPLATE}" "${RESULT_ROOT}/model-template.yaml"
cp "${DATA_TEMPLATE}" "${RESULT_ROOT}/data-template.yaml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/sglang-commit.txt"
git -C "${REPO_ROOT}" status --short --branch > "${RESULT_ROOT}/sglang-status.txt"
printf '%s\n' "${EXPECTED_EVAL360_COMMIT}" > "${RESULT_ROOT}/eval360-commit.txt"

readonly -a CRITICAL_SOURCE=(
  python/sglang/srt/models/xllm.py
  python/sglang/srt/layers/mova.py
  python/sglang/srt/layers/moe/topk.py
  python/sglang/srt/utils/hf_transformers_utils.py
  python/sglang/srt/model_loader/weight_utils.py
)
source_manifest() {
  (cd "${REPO_ROOT}" && sha256sum -- "${CRITICAL_SOURCE[@]}")
}
source_manifest > "${RESULT_ROOT}/sglang-source-sha256-before.txt"
for relative_path in "${CRITICAL_SOURCE[@]}"; do
  mkdir -p "${RESULT_ROOT}/source-snapshot/$(dirname "${relative_path}")"
  cp "${REPO_ROOT}/${relative_path}" "${RESULT_ROOT}/source-snapshot/${relative_path}"
done

checkpoint_manifest() {
  find "${MODEL}" -mindepth 1 -maxdepth 1 \
    -printf '%f\t%y\t%s\t%T@\t%m\t%l\n' | LC_ALL=C sort
}
checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-before.txt"
sha256sum "${PINNED_DATA_BLOB}" > "${RESULT_ROOT}/pinned-data-sha256-before.txt"
cp "${PINNED_DATA_BLOB}" "${INPUT_DATA}"
test "$(sha256sum "${INPUT_DATA}" | awk '{print $1}')" = "${EXPECTED_CHAT5_SOURCE_SHA256}"

sed \
  -e "s|mova-base-0005500|${SERVED_MODEL}|g" \
  -e "s|__BASE_URL__|${BASE_URL}|g" \
  -e "s|parser_type: noop|parser_type: gsm8k_base|" \
  -e "s|name_modifier: gsm8k-greedy|name_modifier: gsm8k-chat5-greedy-16k|" \
  -e "s|__OUTPUT_PATH__|${RESULT_ROOT}/eval-output|g" \
  "${MODEL_TEMPLATE}" > "${MODEL_CONFIG}"
sed \
  -e "s|data_path: .*|data_path: \"${INPUT_DATA}\"|" \
  -e "s|fewshot: 0|fewshot: 5|" \
  "${DATA_TEMPLATE}" > "${DATA_CONFIG}"

export MODEL SERVED_MODEL PORT RESULT_ROOT REPO_ROOT SGLANG_COMMIT
export OPENAI_API_KEY=no-key
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export NLTK_DATA

SERVER_STEP_PID=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${SERVER_STEP_PID}" ]]; then
    kill "${SERVER_STEP_PID}" 2>/dev/null
    wait "${SERVER_STEP_PID}" 2>/dev/null
  fi
  checkpoint_manifest > "${RESULT_ROOT}/checkpoint-stat-after.txt"
  source_manifest > "${RESULT_ROOT}/sglang-source-sha256-after.txt"
  sha256sum "${PINNED_DATA_BLOB}" > "${RESULT_ROOT}/pinned-data-sha256-after.txt"
  if ! cmp "${RESULT_ROOT}/checkpoint-stat-before.txt" "${RESULT_ROOT}/checkpoint-stat-after.txt"; then
    printf 'Checkpoint metadata changed during read-only evaluation.\n' >&2
    rc=97
  fi
  if ! cmp "${RESULT_ROOT}/sglang-source-sha256-before.txt" "${RESULT_ROOT}/sglang-source-sha256-after.txt"; then
    printf 'SGLang source changed during evaluation.\n' >&2
    rc=98
  fi
  if ! cmp "${RESULT_ROOT}/pinned-data-sha256-before.txt" "${RESULT_ROOT}/pinned-data-sha256-after.txt"; then
    printf 'Pinned Eval360 source prompts changed during evaluation.\n' >&2
    rc=99
  fi
  if (( rc != 0 )); then
    printf 'FAILED rc=%s\n' "${rc}" > "${RESULT_ROOT}/status.txt"
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --unbuffered --kill-on-bad-exit=1 --wait=60 \
  --container-image="${IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT},${MODEL}:${MODEL}:ro" \
  --container-workdir="${RESULT_ROOT}" \
  bash -lc '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${REPO_ROOT}/python"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export OMP_NUM_THREADS=8
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_DEBUG=WARN
    export HF_HOME="${RESULT_ROOT}/server-caches/hf"
    export TRITON_CACHE_DIR="${RESULT_ROOT}/server-caches/triton"
    export TORCHINDUCTOR_CACHE_DIR="${RESULT_ROOT}/server-caches/torchinductor"
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    python3 - "${MODEL}" <<'"'"'PY'"'"' | tee "${RESULT_ROOT}/checkpoint-mount.txt"
import os
import sys

target = os.path.realpath(sys.argv[1])
matches = []
with open("/proc/self/mountinfo", encoding="utf-8") as mountinfo:
    for line in mountinfo:
        fields = line.rstrip("\n").split()
        separator = fields.index("-")
        mount_point = fields[4].replace("\\040", " ").replace("\\134", "\\")
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            options = set(fields[5].split(","))
            options.update(fields[separator + 3].split(","))
            matches.append((len(mount_point), mount_point, options))
if not matches:
    raise SystemExit(f"no effective mount found for {target}")
_, mount_point, options = max(matches)
print(f"target={target}")
print(f"effective_mount={mount_point}")
print("options=" + ",".join(sorted(options)))
if "ro" not in options:
    raise SystemExit(f"checkpoint mount is not read-only: {mount_point}")
PY
    nvidia-smi -L | tee "${RESULT_ROOT}/gpu-inventory.txt"
    test "$(nvidia-smi -L | grep -c H200)" = 8
    python3 - <<'"'"'PY'"'"' | tee "${RESULT_ROOT}/environment.txt"
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
PY
    exec python3 -u -m sglang.launch_server \
      --model-path "${MODEL}" \
      --model-impl sglang \
      --served-model-name "${SERVED_MODEL}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tp-size 8 \
      --dtype bfloat16 \
      --kv-cache-dtype bfloat16 \
      --attention-backend fa3 \
      --json-model-override-args "{\"xllm_source_router_gemm_partitions\":2}" \
      --trust-remote-code \
      --context-length 32768 \
      --mem-fraction-static 0.82 \
      --max-running-requests 8 \
      --chunked-prefill-size 512 \
      --cuda-graph-max-bs 8 \
      --random-seed 20260813 \
      --log-level info
  ' > "${RESULT_ROOT}/sglang-server.log" 2>&1 &
SERVER_STEP_PID=$!

server_ready=0
for _ in $(seq 1 180); do
  if curl --silent --show-error --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
    server_ready=1
    break
  fi
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    printf 'SGLang server exited during startup.\n' >&2
    tail -n 200 "${RESULT_ROOT}/sglang-server.log" >&2
    exit 1
  fi
  sleep 10
done
if [[ "${server_ready}" != 1 ]]; then
  printf 'SGLang server did not become healthy within 30 minutes.\n' >&2
  tail -n 200 "${RESULT_ROOT}/sglang-server.log" >&2
  exit 1
fi

curl --silent --show-error --fail "${BASE_URL}/models" \
  | tee "${RESULT_ROOT}/openai-models.json"
curl --silent --show-error --fail "http://127.0.0.1:${PORT}/get_server_info" \
  | tee "${RESULT_ROOT}/server-info.json"
"${EVAL360_PYTHON}" - "${RESULT_ROOT}/server-info.json" "${MODEL}" <<'PY'
import json
import os
import sys

info = json.load(open(sys.argv[1], encoding="utf-8"))
expected_path = os.path.realpath(sys.argv[2])
if os.path.realpath(info.get("model_path", "")) != expected_path:
    raise SystemExit(f"wrong served model path: {info.get('model_path')!r}")
if int(info.get("tp_size", -1)) != 8:
    raise SystemExit(f"wrong TP size: {info.get('tp_size')!r}")
if info.get("enable_fused_qk_norm_rope") is not False:
    raise SystemExit("MoVA fused QK-norm/RoPE must remain disabled")
raw_overrides = info.get("json_model_override_args", "{}")
overrides = json.loads(raw_overrides) if isinstance(raw_overrides, str) else raw_overrides
if overrides != {"xllm_source_router_gemm_partitions": 2}:
    raise SystemExit(f"wrong model overrides: {overrides!r}")
PY
curl --silent --show-error --fail \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer no-key' \
  --data-binary '{"model":"mova-mid4-v2-0010000","messages":[{"role":"user","content":"Reply with OK."}],"temperature":0,"max_tokens":8}' \
  "${BASE_URL}/chat/completions" \
  | tee "${RESULT_ROOT}/chat-smoke.json"

PYTHONPATH="${EVAL360_ROOT}" "${EVAL360_PYTHON}" -m scheduler.cli \
  --max-grading-parallelism 20 \
  --log-dir "${RESULT_ROOT}/eval360-logs" \
  --hf-cache-dir "${HF_CACHE}" \
  evaluate-now \
  --no-slurm \
  --model-paths "${MODEL_CONFIG}" \
  --data-paths "${DATA_CONFIG}"

mapfile -t generation_files < <(find "${RESULT_ROOT}/eval-output" -type f -name '*_generations.jsonl' | LC_ALL=C sort)
mapfile -t grade_files < <(find "${RESULT_ROOT}/eval-output" -type f -name '*_grades.jsonl' | LC_ALL=C sort)
if (( ${#generation_files[@]} != 1 || ${#grade_files[@]} != 1 )); then
  printf 'Expected one generations and one grades file; found %s and %s.\n' \
    "${#generation_files[@]}" "${#grade_files[@]}" >&2
  exit 1
fi

PYTHONPATH="${EVAL360_ROOT}" "${EVAL360_PYTHON}" "${RESULT_ROOT}/audit_eval360_gsm8k.py" \
  --source "${INPUT_DATA}" \
  --generations "${generation_files[0]}" \
  --grades "${grade_files[0]}" \
  --output "${RESULT_ROOT}/eval360-chat5-summary.json" \
  --expected-source-sha256 "${EXPECTED_CHAT5_SOURCE_SHA256}" \
  --expected-rows 1319 \
  | tee "${RESULT_ROOT}/audit.log"
test "$("${EVAL360_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RESULT_ROOT}/eval360-chat5-summary.json")" = PASS

find "${RESULT_ROOT}/eval-output" -type f -name '*_scores.yaml' -print -exec sed -n '1,160p' {} \; \
  | tee "${RESULT_ROOT}/eval360-scores.txt"
printf 'PASS\n' > "${RESULT_ROOT}/status.txt"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
