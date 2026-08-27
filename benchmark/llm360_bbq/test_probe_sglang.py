#!/usr/bin/env python3
"""CPU-only tests for the BBQ SGLang probe result-shape contract."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("BBQ_MODEL_NAME", "unit-model")
os.environ.setdefault("BBQ_MODEL_PATH", "/tmp/bbq-unit-model")
os.environ.setdefault("BBQ_OUTPUT_DIR", "/tmp/bbq-unit-output")

MODULE_PATH = Path(__file__).with_name("probe_sglang.py")
SPEC = importlib.util.spec_from_file_location("bbq_probe_sglang", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ResultShapeContractTests(unittest.TestCase):
    def test_exact_length_inputs_stay_inside_small_configured_vocabulary(self) -> None:
        values = probe._make_exact_length([2, 6], length=5, vocab_size=7, variant=3)
        self.assertEqual(values, [2, 6, 2, 6, 5])
        self.assertTrue(all(0 <= token < 7 for token in values))

    def test_exact_length_inputs_reject_invalid_vocabulary_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "Vocabulary size must be positive"):
            probe._make_exact_length([1], length=2, vocab_size=0)
        with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
            probe._make_exact_length([7], length=2, vocab_size=7)

    def test_local_sglang_source_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            imported_file = root / "python" / "sglang" / "__init__.py"
            imported_file.parent.mkdir(parents=True)
            imported_file.write_text("# unit\n", encoding="utf-8")
            for relative in probe.CRITICAL_SGLANG_SOURCE_FILES:
                source_file = root / relative
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_text("# unit\n", encoding="utf-8")

            source_files = probe._require_local_sglang_source(root, imported_file)
            self.assertEqual(len(source_files), 4)

            missing_root = root / "missing"
            missing_import = missing_root / "python" / "sglang" / "__init__.py"
            missing_import.parent.mkdir(parents=True)
            missing_import.write_text("# unit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
                probe._require_local_sglang_source(missing_root, missing_import)

            outside_import = root / "installed" / "sglang" / "__init__.py"
            outside_import.parent.mkdir(parents=True)
            outside_import.write_text("# unit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the required source"):
                probe._require_local_sglang_source(root, outside_import)

    def test_exact_result_and_token_counts_pass(self) -> None:
        results = [
            {"output_ids": [1, 2, 3]},
            {"output_ids": [4, 5, 6]},
        ]
        probe._require_lengths(results, 2, 3, "unit")

    def test_zero_or_partial_results_fail(self) -> None:
        for results in ([], [{"output_ids": [1, 2, 3]}]):
            with self.subTest(results=results):
                with self.assertRaisesRegex(AssertionError, "expected 2 results"):
                    probe._require_lengths(results, 2, 3, "unit")

    def test_wrong_token_count_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "expected 3 tokens each"):
            probe._require_lengths(
                [{"output_ids": [1, 2, 3]}, {"output_ids": [4, 5]}],
                2,
                3,
                "unit",
            )


if __name__ == "__main__":
    unittest.main()
