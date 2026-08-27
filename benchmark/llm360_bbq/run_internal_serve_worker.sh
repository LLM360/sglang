#!/usr/bin/env bash

set -euo pipefail

: "${BBQ_RUN_MODEL_SLUG:?}"
: "${BBQ_RUN_SERVED_MODEL_NAME:?}"
: "${BBQ_RUN_MODEL_PATH:?}"
: "${BBQ_RUN_TP_SIZE:?}"
: "${BBQ_RUN_PORT:?}"
: "${BBQ_RUN_READY_TIMEOUT:?}"
: "${BBQ_RUN_HEALTH_INTERVAL:?}"
: "${BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS:?}"
: "${BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG:?}"
: "${BBQ_RUN_CHAT_CAPABLE:?}"
: "${BBQ_RUN_REPO_ROOT:?}"
: "${BBQ_RUN_RESULT_ROOT:?}"
: "${BBQ_RUN_ENDPOINT_FILE:?}"
: "${BBQ_RUN_JOB_ENDPOINT_FILE:?}"
: "${BBQ_RUN_SGLANG_COMMIT:?}"
: "${SLURM_JOB_ID:?}"

readonly model_slug=${BBQ_RUN_MODEL_SLUG}
readonly served_model_name=${BBQ_RUN_SERVED_MODEL_NAME}
readonly model_path=${BBQ_RUN_MODEL_PATH}
readonly tp_size=${BBQ_RUN_TP_SIZE}
readonly port=${BBQ_RUN_PORT}
readonly ready_timeout=${BBQ_RUN_READY_TIMEOUT}
readonly health_interval=${BBQ_RUN_HEALTH_INTERVAL}
readonly json_model_override_args=${BBQ_RUN_JSON_MODEL_OVERRIDE_ARGS}
readonly model_loader_extra_config=${BBQ_RUN_MODEL_LOADER_EXTRA_CONFIG}
readonly max_total_tokens=${BBQ_RUN_MAX_TOTAL_TOKENS:-}
readonly chat_capable=${BBQ_RUN_CHAT_CAPABLE}
readonly api_key_file=${BBQ_RUN_API_KEY_FILE:-}
readonly repo_root=${BBQ_RUN_REPO_ROOT}
readonly result_root=${BBQ_RUN_RESULT_ROOT}
readonly endpoint_file=${BBQ_RUN_ENDPOINT_FILE}
readonly job_endpoint_file=${BBQ_RUN_JOB_ENDPOINT_FILE}
readonly sglang_commit=${BBQ_RUN_SGLANG_COMMIT}
readonly node_name=$(hostname -f)
readonly node_ip=$(getent ahostsv4 "${node_name}" | awk 'NR == 1 {print $1}')
readonly base_http_url=http://${node_name}:${port}
readonly base_url=${base_http_url}/v1
readonly local_http_url=http://127.0.0.1:${port}
readonly server_log=${result_root}/server.log
readonly started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
readonly array_job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
readonly array_task_id=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p \
  "${result_root}" \
  "$(dirname "${endpoint_file}")" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${SGLANG_CACHE_DIR}" \
  "${SGLANG_DG_CACHE_DIR}"

api_key=
if [[ -n ${api_key_file} ]]; then
  api_key=$(<"${api_key_file}")
  if [[ -z ${api_key} ]]; then
    printf 'API-key file became empty: %s\n' "${api_key_file}" >&2
    exit 2
  fi
fi
readonly api_key
if [[ -n ${api_key} ]]; then
  readonly api_key_required=true
else
  readonly api_key_required=false
fi

publish_state() {
  local status=$1
  local detail=${2:-}
  local state_ready_at=${3:-}
  local canonical_mode=${4:-replace}
  python3 - \
    "${endpoint_file}" \
    "${job_endpoint_file}" \
    "${canonical_mode}" \
    "${status}" \
    "${detail}" \
    "${state_ready_at}" \
    "${model_slug}" \
    "${served_model_name}" \
    "${model_path}" \
    "${tp_size}" \
    "${node_name}" \
    "${node_ip}" \
    "${port}" \
    "${base_http_url}" \
    "${base_url}" \
    "${SLURM_JOB_ID}" \
    "${array_job_id}" \
    "${array_task_id}" \
    "${sglang_commit}" \
    "${started_at}" \
    "${api_key_required}" \
    "${json_model_override_args}" \
    "${model_loader_extra_config}" \
    "${max_total_tokens}" \
    "${chat_capable}" <<'PY'
import datetime as dt
import fcntl
import json
import os
import pathlib
import sys
import tempfile

(
    endpoint_file,
    job_endpoint_file,
    canonical_mode,
    status,
    detail,
    ready_at,
    model_slug,
    served_model_name,
    model_path,
    tp_size,
    node_name,
    node_ip,
    port,
    base_http_url,
    base_url,
    job_id,
    array_job_id,
    array_task_id,
    sglang_commit,
    started_at,
    api_key_required,
    model_override,
    loader_config,
    max_total_tokens,
    chat_capable,
) = sys.argv[1:]

now = dt.datetime.now(dt.timezone.utc).isoformat()
payload = {
    "schema_version": 1,
    "status": status,
    "detail": detail or None,
    "model": model_slug,
    "served_model_name": served_model_name,
    "checkpoint": model_path,
    "tensor_parallel_size": int(tp_size),
    "node": node_name,
    "node_ipv4": node_ip or None,
    "port": int(port),
    "http_url": base_http_url,
    "base_url": base_url,
    "health_url": f"{base_http_url}/health",
    "models_url": f"{base_url}/models",
    "chat_completions_url": f"{base_url}/chat/completions" if chat_capable == "true" else None,
    "completions_url": f"{base_url}/completions",
    "interfaces": ["completions", *( ["chat_completions"] if chat_capable == "true" else [])],
    "authentication": "bearer" if api_key_required == "true" else "none",
    "slurm_job_id": job_id,
    "slurm_array_job_id": array_job_id,
    "slurm_array_task_id": int(array_task_id),
    "sglang_commit": sglang_commit,
    "started_at": started_at,
    "ready_at": ready_at or None,
    "updated_at": now,
    "json_model_override_args": json.loads(model_override),
    "model_loader_extra_config": json.loads(loader_config),
    "max_total_tokens": int(max_total_tokens) if max_total_tokens else None,
}


def atomic_write(path_string: str) -> None:
    path = pathlib.Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


atomic_write(job_endpoint_file)
lock_path = f"{endpoint_file}.lock"
with open(lock_path, "a+", encoding="utf-8") as lock:
    os.chmod(lock_path, 0o644)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if canonical_mode == "if-current":
        try:
            with open(endpoint_file, encoding="utf-8") as handle:
                current = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            current = None
        if current is not None and str(current.get("slurm_job_id")) != job_id:
            raise SystemExit(0)
    atomic_write(endpoint_file)
PY
}

# Assert that the longest matching mount really makes the checkpoint read-only.
python3 - "${model_path}" <<'PY' | tee "${result_root}/checkpoint-mount.txt"
import os
import sys


def unescape(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


target = os.path.realpath(sys.argv[1])
matches = []
with open("/proc/self/mountinfo", encoding="utf-8") as mountinfo:
    for line in mountinfo:
        fields = line.rstrip("\n").split()
        separator = fields.index("-")
        mount_point = unescape(fields[4])
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

nvidia-smi -L | tee "${result_root}/gpu-inventory.txt"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} != 8 )); then
  printf 'Expected eight GPUs on the exclusive node; found %s\n' "${#gpu_names[@]}" >&2
  exit 1
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ ${gpu_name} != *H200* ]]; then
    printf 'Expected H200 GPUs; found %q\n' "${gpu_name}" >&2
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

publish_state STARTING "loading checkpoint and initializing SGLang"

server_pid=
termination_reason=
ready_at=
server_info_tmp=
cleanup() {
  local rc=$?
  local final_status=FAILED
  local final_detail="server worker exited with status ${rc}"
  local attempt
  trap - EXIT INT TERM HUP
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
  if [[ -n ${server_info_tmp} ]]; then
    rm -f -- "${server_info_tmp}"
  fi
  if [[ -n ${termination_reason} ]]; then
    final_status=STOPPED
    final_detail="received ${termination_reason}; allocation ended"
  fi
  publish_state "${final_status}" "${final_detail}" "${ready_at}" if-current || true
  exit "${rc}"
}
on_signal() {
  termination_reason=$1
  case $1 in
    SIGINT) exit 130 ;;
    SIGTERM) exit 143 ;;
    SIGHUP) exit 129 ;;
  esac
}
trap cleanup EXIT
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGHUP' HUP

server_cmd=(
  python3 -u -m sglang.launch_server
  --model-path "${model_path}"
  --model-impl sglang
  --served-model-name "${served_model_name}"
  --trust-remote-code
  --dtype bfloat16
  --kv-cache-dtype bfloat16
  --tp-size "${tp_size}"
  --attention-backend fa3
  --mem-fraction-static 0.85
  --host 0.0.0.0
  --port "${port}"
  --json-model-override-args "${json_model_override_args}"
  --model-loader-extra-config "${model_loader_extra_config}"
)
if [[ -n ${max_total_tokens} ]]; then
  server_cmd+=(--max-total-tokens "${max_total_tokens}")
fi
if [[ -n ${api_key} ]]; then
  # SGLang logs its complete ServerArgs dataclass at INFO, including api_key.
  # Keep keyed-mode logs at WARNING so the shared secret never lands in Weka.
  server_cmd+=(
    --api-key "${api_key}"
    --log-level warning
    --log-level-http warning
  )
fi

# Never write the optional bearer token to an artifact.
if [[ -n ${api_key} ]]; then
  redacted_server_cmd=(
    "${server_cmd[@]:0:$((${#server_cmd[@]} - 6))}"
    --api-key '<redacted>'
    --log-level warning
    --log-level-http warning
  )
else
  redacted_server_cmd=("${server_cmd[@]}")
fi
{
  printf '%q ' "${redacted_server_cmd[@]}"
  printf '\n'
} > "${result_root}/server-command-redacted.txt"

setsid "${server_cmd[@]}" > "${server_log}" 2>&1 &
server_pid=$!

auth_args=()
if [[ -n ${api_key} ]]; then
  auth_args=(-H "Authorization: Bearer ${api_key}")
fi

readonly readiness_start=${SECONDS}
while ! curl --silent --show-error --fail --max-time 10 \
  "${local_http_url}/health" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    set +e
    wait "${server_pid}"
    server_status=$?
    set -e
    printf 'SGLang exited before readiness with status %s\n' "${server_status}" >&2
    tail -n 200 "${server_log}" >&2
    if (( server_status == 0 )); then
      server_status=1
    fi
    exit "${server_status}"
  fi
  if (( SECONDS - readiness_start >= ready_timeout )); then
    printf 'SGLang was not healthy after %ss\n' "${ready_timeout}" >&2
    tail -n 200 "${server_log}" >&2
    exit 1
  fi
  sleep 5
done
ready_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
readonly ready_at
printf 'Server ready after %ss at %s\n' "$((SECONDS - readiness_start))" "${ready_at}" \
  | tee "${result_root}/server-ready.txt"

curl --silent --show-error --fail --max-time 30 \
  "${auth_args[@]}" "${local_http_url}/v1/models" \
  | tee "${result_root}/openai-models.json" >/dev/null
server_info_tmp=$(mktemp "/tmp/bbq-server-info-${SLURM_JOB_ID}-${array_task_id}.XXXXXX")
chmod 600 "${server_info_tmp}"
curl --silent --show-error --fail --max-time 30 \
  "${auth_args[@]}" "${local_http_url}/server_info" \
  --output "${server_info_tmp}"
python3 - \
  "${server_info_tmp}" \
  "${result_root}/server-info-redacted.json" \
  "${model_path}" \
  "${served_model_name}" \
  "${tp_size}" \
  "${port}" \
  "${json_model_override_args}" \
  "${max_total_tokens}" <<'PY'
import json
import os
import sys

source, output, expected_path, expected_name, expected_tp, expected_port, expected_override, expected_tokens = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    info = json.load(handle)

checks = {
    "model_path": os.path.realpath(info.get("model_path", "")) == os.path.realpath(expected_path),
    "served_model_name": info.get("served_model_name") == expected_name,
    "tp_size": int(info.get("tp_size", -1)) == int(expected_tp),
    "port": int(info.get("port", -1)) == int(expected_port),
    "host": info.get("host") == "0.0.0.0",
    "model_impl": info.get("model_impl") == "sglang",
    "dtype": info.get("dtype") == "bfloat16",
    "kv_cache_dtype": info.get("kv_cache_dtype") == "bfloat16",
    "attention_backend": info.get("attention_backend") == "fa3",
    "mem_fraction_static": abs(float(info.get("mem_fraction_static", -1)) - 0.85) < 1e-12,
    "fused_qk_norm_rope_disabled": info.get("enable_fused_qk_norm_rope") is False,
}
raw_override = info.get("json_model_override_args", "{}")
actual_override = json.loads(raw_override) if isinstance(raw_override, str) else raw_override
checks["json_model_override_args"] = actual_override == json.loads(expected_override)
if expected_tokens:
    checks["max_total_tokens"] = int(info.get("max_total_tokens", -1)) == int(expected_tokens)
if not all(checks.values()):
    failed = sorted(name for name, passed in checks.items() if not passed)
    raise SystemExit(f"server-info contract mismatch: {failed}")

redacted = {
    "checks": checks,
    "model_path": info.get("model_path"),
    "served_model_name": info.get("served_model_name"),
    "tp_size": info.get("tp_size"),
    "host": info.get("host"),
    "port": info.get("port"),
    "model_impl": info.get("model_impl"),
    "dtype": info.get("dtype"),
    "kv_cache_dtype": info.get("kv_cache_dtype"),
    "attention_backend": info.get("attention_backend"),
    "mem_fraction_static": info.get("mem_fraction_static"),
    "max_total_tokens": info.get("max_total_tokens"),
    "enable_fused_qk_norm_rope": info.get("enable_fused_qk_norm_rope"),
    "json_model_override_args": actual_override,
    "version": info.get("version"),
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(redacted, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
rm -f -- "${server_info_tmp}"
server_info_tmp=

startup_completion_raw=${result_root}/startup-completion-smoke.raw.json
curl --silent --show-error --fail --max-time 120 \
  "${auth_args[@]}" \
  -H 'Content-Type: application/json' \
  --data-binary "$(python3 - "${served_model_name}" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "prompt": "The answer is",
    "temperature": 0,
    "max_tokens": 4,
}))
PY
)" \
  "${local_http_url}/v1/completions" \
  --output "${startup_completion_raw}"
python3 - \
  "${startup_completion_raw}" \
  "${result_root}/startup-completion-smoke.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
choices = payload.get("choices")
if not isinstance(choices, list) or not choices or not isinstance(choices[0].get("text"), str):
    raise SystemExit(f"invalid completion response: {payload!r}")
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
rm -f -- "${startup_completion_raw}"

if [[ ${chat_capable} == true ]]; then
  startup_chat_raw=${result_root}/startup-chat-smoke.raw.json
  curl --silent --show-error --fail --max-time 120 \
    "${auth_args[@]}" \
    -H 'Content-Type: application/json' \
    --data-binary "$(python3 - "${served_model_name}" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply briefly."}],
    "temperature": 0,
    "max_tokens": 4,
}))
PY
)" \
    "${local_http_url}/v1/chat/completions" \
    --output "${startup_chat_raw}"
  python3 - \
    "${startup_chat_raw}" \
    "${result_root}/startup-chat-smoke.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    payload = json.load(source)
choices = payload.get("choices")
if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
    raise SystemExit(f"invalid chat response: {payload!r}")
message = choices[0].get("message")
if not isinstance(message, dict):
    raise SystemExit(f"chat response has no message object: {choices[0]!r}")
if not any(isinstance(message.get(field), str) for field in ("content", "reasoning_content")):
    raise SystemExit(f"chat response has no text field: {message!r}")
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  rm -f -- "${startup_chat_raw}"
fi

publish_state READY "startup contract and supported-interface smokes passed" "${ready_at}"
printf 'READY %s %s\n' "${model_slug}" "${base_url}" | tee "${result_root}/status.txt"

consecutive_health_failures=0
while kill -0 "${server_pid}" 2>/dev/null; do
  sleep "${health_interval}"
  if curl --silent --show-error --fail --max-time 10 \
    "${local_http_url}/health" >/dev/null 2>&1; then
    consecutive_health_failures=0
    publish_state READY "health check passed" "${ready_at}" if-current
  else
    consecutive_health_failures=$((consecutive_health_failures + 1))
    publish_state UNHEALTHY \
      "health check failed ${consecutive_health_failures} consecutive time(s)" \
      "${ready_at}" if-current
  fi
done

set +e
wait "${server_pid}"
server_status=$?
set -e
printf 'SGLang server exited unexpectedly with status %s\n' "${server_status}" >&2
if (( server_status == 0 )); then
  exit 1
fi
exit "${server_status}"
