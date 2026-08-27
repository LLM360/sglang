#!/usr/bin/env python3
"""Validate and summarize one detailed ``sglang.bench_serving`` JSONL result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument("--num-requests", required=True, type=int)
    parser.add_argument("--max-concurrency", required=True, type=int)
    parser.add_argument("--warmup-requests", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--input-len", default=512, type=int)
    parser.add_argument("--output-len", default=512, type=int)
    parser.add_argument("--json-model-override-args", required=True)
    parser.add_argument("--model-loader-extra-config-json", required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--sglang-source-root", required=True, type=Path)
    parser.add_argument("--source-snapshot-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile over an empty sample")
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("latency sample contains a non-finite value")
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")


def require_positive_finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite, got {number!r}")
    return number


def require_rate_consistent(
    name: str,
    actual: float,
    *,
    numerator: int,
    duration: float,
) -> None:
    expected = numerator / duration
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{name} is inconsistent with counts and duration: "
            f"expected {expected!r}, got {actual!r}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_single_result(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require_equal("JSONL record count", len(records), 1)
    if not isinstance(records[0], dict):
        raise ValueError("benchmark JSONL record must be an object")
    return records[0]


def validate_server_info(
    result: dict[str, Any],
    tp_size: int,
    expected_model_override: dict[str, Any],
    expected_loader_config: dict[str, Any],
) -> dict[str, Any]:
    server_info = result.get("server_info")
    if not isinstance(server_info, dict):
        raise ValueError("benchmark did not capture /server_info")

    require_equal("server model_impl", server_info.get("model_impl"), "sglang")
    require_equal("server dtype", server_info.get("dtype"), "bfloat16")
    require_equal(
        "server KV-cache dtype", server_info.get("kv_cache_dtype"), "bfloat16"
    )
    require_equal(
        "server attention backend", server_info.get("attention_backend"), "fa3"
    )
    require_equal("server tensor parallel size", server_info.get("tp_size"), tp_size)
    require_equal(
        "server disable_cuda_graph", server_info.get("disable_cuda_graph"), False
    )
    require_equal(
        "server disable_radix_cache", server_info.get("disable_radix_cache"), False
    )
    require_equal(
        "server enable_fused_qk_norm_rope",
        server_info.get("enable_fused_qk_norm_rope"),
        False,
    )
    if "json_model_override_args" not in server_info:
        raise ValueError("server_info is missing json_model_override_args")
    actual_override = json.loads(server_info["json_model_override_args"])
    require_equal(
        "server JSON model override", actual_override, expected_model_override
    )
    if "model_loader_extra_config" not in server_info:
        raise ValueError("server_info is missing model_loader_extra_config")
    actual_loader_config = json.loads(server_info["model_loader_extra_config"])
    require_equal(
        "server model loader extra config",
        actual_loader_config,
        expected_loader_config,
    )
    return server_info


def main() -> None:
    args = parse_args()
    result = load_single_result(args.input)

    require_equal("backend", result.get("backend"), "sglang")
    require_equal("dataset", result.get("dataset_name"), "random-ids")
    require_equal(
        "configured random input length", result.get("random_input_len"), args.input_len
    )
    require_equal(
        "configured random output length",
        result.get("random_output_len"),
        args.output_len,
    )
    require_equal(
        "configured random range ratio", result.get("random_range_ratio"), 1.0
    )
    require_equal("completed requests", result.get("completed"), args.num_requests)
    require_equal(
        "aggregate input token count",
        result.get("total_input_tokens"),
        args.num_requests * args.input_len,
    )
    require_equal(
        "aggregate output token count",
        result.get("total_output_tokens"),
        args.num_requests * args.output_len,
    )

    input_lens = result.get("input_lens")
    output_lens = result.get("output_lens")
    errors = result.get("errors")
    ttfts = result.get("ttfts")
    request_itls = result.get("itls")
    for name, values in (
        ("input_lens", input_lens),
        ("output_lens", output_lens),
        ("errors", errors),
        ("ttfts", ttfts),
        ("itls", request_itls),
    ):
        if not isinstance(values, list):
            raise ValueError(f"detailed benchmark field {name!r} is missing")
        require_equal(f"{name} count", len(values), args.num_requests)

    if any(length != args.input_len for length in input_lens):
        raise ValueError(
            "not every measured request had exactly the requested input length"
        )
    if any(length != args.output_len for length in output_lens):
        raise ValueError(
            "not every measured request produced exactly the requested output length; "
            "ignore-EOS or capacity may be misconfigured"
        )
    if any(error not in (None, "") for error in errors):
        raise ValueError("one or more serving requests reported an error")

    ttft_seconds = [float(value) for value in ttfts]
    if any(value <= 0 or not math.isfinite(value) for value in ttft_seconds):
        raise ValueError("TTFT samples must all be positive and finite")

    itl_seconds: list[float] = []
    for request_values in request_itls:
        if not isinstance(request_values, list):
            raise ValueError("each per-request ITL value must be a list")
        itl_seconds.extend(float(value) for value in request_values)
    if not itl_seconds:
        raise ValueError("benchmark produced no inter-token latency samples")
    if any(value < 0 or not math.isfinite(value) for value in itl_seconds):
        raise ValueError("ITL samples must all be non-negative and finite")

    duration = require_positive_finite("duration", result["duration"])
    request_throughput = require_positive_finite(
        "request throughput", result["request_throughput"]
    )
    output_throughput = require_positive_finite(
        "output throughput", result["output_throughput"]
    )
    total_throughput = require_positive_finite(
        "total throughput", result["total_throughput"]
    )
    require_rate_consistent(
        "request throughput",
        request_throughput,
        numerator=args.num_requests,
        duration=duration,
    )
    require_rate_consistent(
        "output throughput",
        output_throughput,
        numerator=args.num_requests * args.output_len,
        duration=duration,
    )
    require_rate_consistent(
        "total throughput",
        total_throughput,
        numerator=args.num_requests * (args.input_len + args.output_len),
        duration=duration,
    )

    expected_model_override = json.loads(args.json_model_override_args)
    if not isinstance(expected_model_override, dict):
        raise ValueError("JSON model override must be an object")
    expected_loader_config = json.loads(args.model_loader_extra_config_json)
    if not isinstance(expected_loader_config, dict):
        raise ValueError("model loader extra config must be an object")
    server_info = validate_server_info(
        result,
        args.tp_size,
        expected_model_override,
        expected_loader_config,
    )
    source_files = [
        args.source_snapshot_root / "xllm.py",
        args.source_snapshot_root / "mova.py",
        args.source_snapshot_root / "weight_utils.py",
        args.source_snapshot_root / "hf_transformers_utils.py",
        Path(__file__).resolve(),
        args.output.parent / "run-worker.sh",
        args.output.parent / "run.slurm",
    ]
    missing_source_files = [path for path in source_files if not path.is_file()]
    if missing_source_files:
        raise ValueError(
            "source provenance is incomplete; missing expected files: "
            + ", ".join(str(path) for path in missing_source_files)
        )
    source_sha256 = {str(path.resolve()): sha256_file(path) for path in source_files}
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "slurm_job_id": args.job_id,
            "sglang_commit": args.sglang_commit,
            "model_name": args.model_name,
            "model_path": os.path.realpath(args.model_path),
            "raw_result": str(args.input.resolve()),
            "source_sha256": source_sha256,
        },
        "workload": {
            "backend": "sglang",
            "model_implementation": "sglang",
            "dataset": "random-ids",
            "tokenized_prompt_ids": True,
            "input_tokens_per_request": args.input_len,
            "output_tokens_per_request": args.output_len,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": args.seed,
            "measured_requests": args.num_requests,
            "max_concurrency": args.max_concurrency,
            "warmup_requests": args.warmup_requests,
            "request_rate": "inf",
        },
        "runtime": {
            "tensor_parallel_size": args.tp_size,
            "dtype": server_info["dtype"],
            "kv_cache_dtype": server_info["kv_cache_dtype"],
            "attention_backend": server_info["attention_backend"],
            "json_model_override_args": expected_model_override,
            "model_loader_extra_config": expected_loader_config,
            "mem_fraction_static": server_info.get("mem_fraction_static"),
            "max_total_tokens": server_info.get("max_total_tokens"),
            "cuda_graph_enabled": not server_info["disable_cuda_graph"],
            "radix_cache_enabled": not server_info["disable_radix_cache"],
            "fused_qk_norm_rope_enabled": server_info["enable_fused_qk_norm_rope"],
            "sglang_version": server_info.get("version"),
        },
        "results": {
            "duration_seconds": duration,
            "completed_requests": int(result["completed"]),
            "total_input_tokens": int(result["total_input_tokens"]),
            "total_output_tokens": int(result["total_output_tokens"]),
            "aggregate_output_tokens_per_second": output_throughput,
            "aggregate_total_tokens_per_second": total_throughput,
            "requests_per_second": request_throughput,
            "p50_ttft_ms": percentile(ttft_seconds, 0.50) * 1000.0,
            "p95_ttft_ms": percentile(ttft_seconds, 0.95) * 1000.0,
            "p50_inter_token_latency_ms": percentile(itl_seconds, 0.50) * 1000.0,
            "p95_inter_token_latency_ms": percentile(itl_seconds, 0.95) * 1000.0,
            "ttft_sample_count": len(ttft_seconds),
            "inter_token_latency_sample_count": len(itl_seconds),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
