#!/usr/bin/env python3
"""CPU-only tests for BBQ serving-summary validation and provenance."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("summarize_serving_decode.py")
SPEC = importlib.util.spec_from_file_location("bbq_serving_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def _server_info() -> dict[str, object]:
    return {
        "model_impl": "sglang",
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "attention_backend": "fa3",
        "tp_size": 1,
        "disable_cuda_graph": False,
        "disable_radix_cache": False,
        "enable_fused_qk_norm_rope": False,
        "json_model_override_args": "{}",
        "model_loader_extra_config": "{}",
        "mem_fraction_static": 0.85,
        "max_total_tokens": 8192,
        "version": "unit",
    }


class ServingValidationTests(unittest.TestCase):
    def test_aggregate_rates_and_duration_must_be_positive_and_finite(self) -> None:
        self.assertEqual(summary.require_positive_finite("metric", 1.25), 1.25)
        for invalid in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    summary.require_positive_finite("metric", invalid)

    def test_aggregate_rates_must_match_counts_and_duration(self) -> None:
        summary.require_rate_consistent("metric", 5.0, numerator=10, duration=2.0)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            summary.require_rate_consistent("metric", 5.01, numerator=10, duration=2.0)

    def test_server_config_fields_must_be_present(self) -> None:
        for field in ("json_model_override_args", "model_loader_extra_config"):
            with self.subTest(field=field):
                server_info = _server_info()
                del server_info[field]
                with self.assertRaisesRegex(ValueError, f"missing {field}"):
                    summary.validate_server_info(
                        {"server_info": server_info}, 1, {}, {}
                    )

    def test_rope_fusion_must_be_explicitly_disabled(self) -> None:
        server_info = _server_info()
        server_info["enable_fused_qk_norm_rope"] = True
        with self.assertRaisesRegex(ValueError, "enable_fused_qk_norm_rope"):
            summary.validate_server_info({"server_info": server_info}, 1, {}, {})

    def test_summary_hashes_prelaunch_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "source-snapshot"
            snapshot.mkdir()
            snapshot_files = (
                "xllm.py",
                "mova.py",
                "weight_utils.py",
                "hf_transformers_utils.py",
            )
            for index, name in enumerate(snapshot_files):
                (snapshot / name).write_text(f"snapshot-{index}\n", encoding="utf-8")
            (root / "run-worker.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (root / "run.slurm").write_text("#!/bin/sh\n", encoding="utf-8")

            raw = root / "raw.jsonl"
            output = root / "summary.json"
            record = {
                "backend": "sglang",
                "dataset_name": "random-ids",
                "random_input_len": 512,
                "random_output_len": 512,
                "random_range_ratio": 1.0,
                "completed": 2,
                "total_input_tokens": 1024,
                "total_output_tokens": 1024,
                "input_lens": [512, 512],
                "output_lens": [512, 512],
                "errors": [None, None],
                "ttfts": [0.1, 0.2],
                "itls": [[0.01], [0.02]],
                "output_throughput": 512.0,
                "total_throughput": 1024.0,
                "request_throughput": 1.0,
                "duration": 2.0,
                "server_info": _server_info(),
            }
            raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                input=raw,
                output=output,
                model_name="unit-model",
                model_path="/tmp/unit-model",
                tp_size=1,
                num_requests=2,
                max_concurrency=2,
                warmup_requests=1,
                seed=7,
                input_len=512,
                output_len=512,
                json_model_override_args="{}",
                model_loader_extra_config_json="{}",
                sglang_commit="unit-commit",
                sglang_source_root=root / "live-source-not-used",
                source_snapshot_root=snapshot,
                job_id="unit-job",
            )
            with mock.patch.object(summary, "parse_args", return_value=args):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary.main()

            result = json.loads(output.read_text(encoding="utf-8"))
            hashes = result["provenance"]["source_sha256"]
            self.assertEqual(len(hashes), 7)
            for name in snapshot_files:
                path = str((snapshot / name).resolve())
                self.assertEqual(hashes[path], summary.sha256_file(snapshot / name))
            self.assertFalse(
                any("live-source-not-used" in path for path in hashes),
                hashes,
            )


if __name__ == "__main__":
    unittest.main()
