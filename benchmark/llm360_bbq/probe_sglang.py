#!/usr/bin/env python3
"""Offline SGLang correctness and decode-oriented smoke probe for BBQ models."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.utils.hf_transformers_utils import get_config

MODEL_NAME = os.environ["BBQ_MODEL_NAME"]
MODEL_PATH = Path(os.environ["BBQ_MODEL_PATH"]).resolve()
OUTPUT_DIR = Path(os.environ["BBQ_OUTPUT_DIR"]).resolve()
TP_SIZE = int(os.environ.get("BBQ_TP_SIZE", "1"))
MAX_TOTAL_TOKENS = int(os.environ.get("BBQ_MAX_TOTAL_TOKENS", "32768"))
MAX_RUNNING_REQUESTS = int(os.environ.get("BBQ_MAX_RUNNING_REQUESTS", "16"))
CHUNKED_PREFILL_SIZE = int(os.environ.get("BBQ_CHUNKED_PREFILL_SIZE", "1024"))
CUDA_GRAPH_MAX_BS = int(os.environ.get("BBQ_CUDA_GRAPH_MAX_BS", "16"))
LONG_INPUT_LEN = int(os.environ.get("BBQ_LONG_INPUT_LEN", "4096"))
BENCH_BATCH_SIZES = tuple(
    int(value)
    for value in os.environ.get("BBQ_BENCH_BATCH_SIZES", "1,8,16").split(",")
    if value
)
BENCH_INPUT_LEN = int(os.environ.get("BBQ_BENCH_INPUT_LEN", "128"))
BENCH_OUTPUT_LEN = int(os.environ.get("BBQ_BENCH_OUTPUT_LEN", "128"))
MODEL_OVERRIDE_JSON = os.environ.get("BBQ_MODEL_OVERRIDE_JSON", "{}")
MODEL_LOADER_EXTRA_CONFIG_JSON = os.environ.get(
    "BBQ_MODEL_LOADER_EXTRA_CONFIG_JSON",
    "{}",
)
LOAD_FORMAT = os.environ.get("BBQ_LOAD_FORMAT", "auto")
EXPECTED_ARCH = os.environ.get("BBQ_EXPECTED_ARCH", "")
SGLANG_COMMIT = os.environ.get("BBQ_SGLANG_COMMIT", "unknown")
SGLANG_SOURCE_ROOT = Path(
    os.environ.get("BBQ_SGLANG_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
CRITICAL_SGLANG_SOURCE_FILES = (
    Path("python/sglang/srt/models/xllm.py"),
    Path("python/sglang/srt/layers/mova.py"),
    Path("python/sglang/srt/model_loader/weight_utils.py"),
    Path("python/sglang/srt/utils/hf_transformers_utils.py"),
)


PROMPTS = (
    "The capital of France is",
    "Complete the sequence: 2, 4, 8, 16,",
    "Question: If a box has 7 rows of 9 apples, how many apples are there?\nAnswer:",
    "Write one sentence explaining why deterministic inference tests are useful:",
    "In Python, a context manager is used to",
    "Translate to French: The weather is clear today.\nTranslation:",
    "A paged key-value cache helps an inference server by",
    "The derivative of x squared with respect to x is",
    "List three primary colors:",
    "Water freezes at zero degrees on the",
    "A tensor-parallel language model splits computation across",
    "Question: A train travels 60 miles per hour for 3 hours. Distance?\nAnswer:",
    "The opposite of 'scarce' is",
    'Continue this JSON object with a numeric value: {"count":',
    "A safe production rollout should verify correctness and",
    "The first five positive prime numbers are",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_local_sglang_source(
    source_root: Path, imported_sglang_file: Path
) -> list[Path]:
    package_root = (source_root / "python" / "sglang").resolve()
    imported_file = imported_sglang_file.resolve()
    try:
        imported_file.relative_to(package_root)
    except ValueError as error:
        raise ValueError(
            f"Imported sglang is outside the required source tree: "
            f"{imported_file} not under {package_root}"
        ) from error

    source_files = [
        (source_root / relative).resolve() for relative in CRITICAL_SGLANG_SOURCE_FILES
    ]
    missing = [path for path in source_files if not path.is_file()]
    if missing:
        raise ValueError(
            "Required SGLang source provenance is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    return source_files


def _write(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "sglang-probe.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(target)


def _normalize_results(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def _output_ids(results: list[dict[str, Any]]) -> list[list[int]]:
    return [[int(token) for token in result["output_ids"]] for result in results]


def _require_lengths(
    results: list[dict[str, Any]],
    expected_results: int,
    expected_tokens: int,
    label: str,
) -> None:
    if len(results) != expected_results:
        raise AssertionError(
            f"{label}: expected {expected_results} results, got {len(results)}"
        )
    lengths = [len(result.get("output_ids", [])) for result in results]
    if lengths != [expected_tokens] * expected_results:
        raise AssertionError(
            f"{label}: expected {expected_tokens} tokens each, got {lengths}"
        )


def _sampling(max_new_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }


def _make_exact_length(seed_ids: list[int], length: int, variant: int = 0) -> list[int]:
    if length < 2:
        raise ValueError(f"Synthetic input length must be >= 2, got {length}")
    if not seed_ids:
        raise ValueError("Cannot construct a synthetic input from an empty token list")
    values = (seed_ids * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]
    values[-1] = int((values[-1] + variant) % 250000)
    return values


def main() -> None:
    if not getattr(sgl, "__file__", None):
        raise ValueError("Imported sglang package has no filesystem source path")
    source_files = _require_local_sglang_source(SGLANG_SOURCE_ROOT, Path(sgl.__file__))
    model_override_args = json.loads(MODEL_OVERRIDE_JSON)
    config = get_config(
        str(MODEL_PATH),
        trust_remote_code=True,
        model_override_args=model_override_args,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt_ids = [
        [int(token) for token in tokenizer.encode(prompt, add_special_tokens=True)]
        for prompt in PROMPTS
    ]
    if EXPECTED_ARCH and EXPECTED_ARCH not in (config.architectures or []):
        raise AssertionError(
            f"Expected architecture {EXPECTED_ARCH}, got {config.architectures}"
        )
    if LONG_INPUT_LEN + 8 > int(config.max_position_embeddings):
        raise ValueError(
            f"Requested {LONG_INPUT_LEN}+8 tokens exceeds configured context "
            f"{config.max_position_embeddings}"
        )

    record: dict[str, Any] = {
        "status": "loading",
        "model": {
            "name": MODEL_NAME,
            "path": str(MODEL_PATH),
            "architectures": list(config.architectures or []),
            "model_type": str(config.model_type),
            "checkpoint_declared_dtype": str(getattr(config, "dtype", None)),
            "serving_dtype": "bfloat16",
            "layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "attention_heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "max_position_embeddings": int(config.max_position_embeddings),
            "eos_token_id": getattr(config, "eos_token_id", None),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sglang": getattr(sgl, "__version__", "unknown"),
            "visible_gpu_count": torch.cuda.device_count(),
            "sglang_commit": SGLANG_COMMIT,
        },
        "source_sha256": {str(path): _sha256(path) for path in source_files},
        "engine": {
            "tp_size": TP_SIZE,
            "dtype": "bfloat16",
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "chunked_prefill_size": CHUNKED_PREFILL_SIZE,
            "cuda_graph_max_bs": CUDA_GRAPH_MAX_BS,
            "model_override_json": model_override_args,
            "load_format": LOAD_FORMAT,
            "model_impl": "sglang",
            "model_loader_extra_config": json.loads(MODEL_LOADER_EXTRA_CONFIG_JSON),
        },
        "prompts": list(PROMPTS),
        "prompt_ids": prompt_ids,
        "tests": {},
    }
    _write(record)

    engine = None
    try:
        started = time.perf_counter()
        engine = sgl.Engine(
            model_path=str(MODEL_PATH),
            tp_size=TP_SIZE,
            dtype="bfloat16",
            trust_remote_code=True,
            skip_tokenizer_init=True,
            load_format=LOAD_FORMAT,
            model_impl="sglang",
            model_loader_extra_config=MODEL_LOADER_EXTRA_CONFIG_JSON,
            random_seed=20260827,
            mem_fraction_static=0.82,
            max_total_tokens=MAX_TOTAL_TOKENS,
            max_running_requests=MAX_RUNNING_REQUESTS,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            cuda_graph_max_bs=CUDA_GRAPH_MAX_BS,
            json_model_override_args=MODEL_OVERRIDE_JSON,
            attention_backend="fa3",
            kv_cache_dtype="bfloat16",
            enable_fused_qk_norm_rope=False,
            log_level="info",
        )
        record["engine"]["initialization_seconds"] = time.perf_counter() - started
        record["engine"]["resolved_architecture"] = list(
            engine.server_args.get_model_config().hf_config.architectures or []
        )
        record["engine"]["resolved_attention_backend"] = str(
            engine.server_args.attention_backend
        )
        record["engine"]["resolved_disable_cuda_graph"] = bool(
            engine.server_args.disable_cuda_graph
        )
        record["engine"]["resolved_disable_radix_cache"] = bool(
            engine.server_args.disable_radix_cache
        )
        if engine.server_args.disable_cuda_graph:
            raise AssertionError("CUDA graphs were unexpectedly disabled")
        if engine.server_args.disable_radix_cache:
            raise AssertionError("Radix cache was unexpectedly disabled")
        _write(record)

        started = time.perf_counter()
        first = _normalize_results(
            engine.generate(
                input_ids=prompt_ids,
                sampling_params=_sampling(16),
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=5,
            )
        )
        first_seconds = time.perf_counter() - started
        _require_lengths(first, len(prompt_ids), 16, "broad-first")
        for index, result in enumerate(first):
            meta = result.get("meta_info", {})
            if len(meta.get("output_token_logprobs", [])) != 16:
                raise AssertionError(f"Prompt {index} lacks output-token logprobs")
            if len(meta.get("output_top_logprobs", [])) != 16:
                raise AssertionError(f"Prompt {index} lacks top-five output logprobs")

        started = time.perf_counter()
        repeated = _normalize_results(
            engine.generate(
                input_ids=prompt_ids,
                sampling_params=_sampling(16),
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=5,
            )
        )
        repeat_seconds = time.perf_counter() - started
        _require_lengths(repeated, len(prompt_ids), 16, "broad-repeat")
        if _output_ids(first) != _output_ids(repeated):
            raise AssertionError("Repeated greedy generation changed output token IDs")
        record["tests"]["broad_cached_determinism"] = {
            "status": "PASS",
            "first_seconds": first_seconds,
            "repeat_seconds": repeat_seconds,
            "first": first,
            "repeat": repeated,
        }
        _write(record)

        seed_ids = prompt_ids[0]
        long_ids = _make_exact_length(seed_ids, LONG_INPUT_LEN)
        started = time.perf_counter()
        long_result = _normalize_results(
            engine.generate(
                input_ids=long_ids,
                sampling_params=_sampling(8),
                return_logprob=True,
                logprob_start_len=max(0, len(long_ids) - 4),
                top_logprobs_num=5,
            )
        )
        long_seconds = time.perf_counter() - started
        _require_lengths(long_result, 1, 8, "long-context")
        record["tests"]["long_context"] = {
            "status": "PASS",
            "input_tokens": len(long_ids),
            "output_tokens": 8,
            "seconds": long_seconds,
            "result": long_result,
        }
        _write(record)

        benchmark_rows = []
        bench_seed = _make_exact_length(seed_ids, BENCH_INPUT_LEN)
        for batch_size in BENCH_BATCH_SIZES:
            inputs = [
                _make_exact_length(bench_seed, BENCH_INPUT_LEN, variant=index)
                for index in range(batch_size)
            ]
            started = time.perf_counter()
            results = _normalize_results(
                engine.generate(
                    input_ids=inputs,
                    sampling_params=_sampling(BENCH_OUTPUT_LEN),
                )
            )
            elapsed = time.perf_counter() - started
            _require_lengths(
                results,
                batch_size,
                BENCH_OUTPUT_LEN,
                f"decode-batch-{batch_size}",
            )
            total_output_tokens = sum(len(result["output_ids"]) for result in results)
            benchmark_rows.append(
                {
                    "batch_size": batch_size,
                    "input_tokens_per_request": BENCH_INPUT_LEN,
                    "output_tokens_per_request": BENCH_OUTPUT_LEN,
                    "elapsed_seconds": elapsed,
                    "output_tokens": total_output_tokens,
                    "output_tokens_per_second": total_output_tokens / elapsed,
                }
            )
            record["tests"]["decode_oriented_benchmark"] = {
                "status": "RUNNING",
                "rows": benchmark_rows,
            }
            _write(record)

        record["tests"]["decode_oriented_benchmark"]["status"] = "PASS"
        record["status"] = "PASS"
        _write(record)
        print(f"BBQ_SGLANG_PROBE_{MODEL_NAME}=PASS", flush=True)
    except BaseException as error:
        record["status"] = "FAIL"
        record["error"] = f"{type(error).__name__}: {error}"
        _write(record)
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
