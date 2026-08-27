#!/usr/bin/env python3
"""CPU-only tests for the pinned BBQ GSM8K prompt and scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("gsm8k_sglang.py")
SPEC = importlib.util.spec_from_file_location("gsm8k_sglang", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gsm8k)


class PromptContractTests(unittest.TestCase):
    def test_pinned_eight_shot_prefix(self) -> None:
        prefix = gsm8k.fewshot_prefix()
        self.assertEqual(len(gsm8k.GSM8K_FEWSHOTS), 8)
        self.assertEqual(prefix.count("\n\nQ: "), 7)
        self.assertTrue(prefix.startswith("Q: There are 15 trees in the grove."))
        self.assertTrue(prefix.endswith("The answer is 8."))
        self.assertEqual(
            hashlib.sha256(prefix.encode()).hexdigest(),
            "5bd0f2b630cc5fe5b2d8d88c094be5dcabac634ed6e208c929087650e29bbe02",
        )

    def test_test_question_is_appended_without_chat_template(self) -> None:
        prompt = gsm8k.build_prompt("What is two plus two?")
        self.assertTrue(prompt.endswith("\n\nQ: What is two plus two?\nA:"))
        self.assertNotIn("<|ifm|", prompt)

    def test_question_extraction(self) -> None:
        raw = (
            "Q: A robe takes two bolts.  How many total?\nA: Let's think step by step."
        )
        self.assertEqual(
            gsm8k.question_from_completion_input(raw),
            "A robe takes two bolts.  How many total?",
        )


class AnswerContractTests(unittest.TestCase):
    def test_xllm_style_number_extraction(self) -> None:
        cases = (
            ("The answer is 5.50 miles.", "5.50"),
            ("The answer is -3,000.0.", "-3000"),
            ("Reasoning 12. Then the answer is 1/2.", "1/2"),
            ("Work gives 12, but finally \\boxed{16}.", "16"),
            ("First 7, then 24 dollars.", "24"),
            ("no numerical answer", ""),
        )
        for completion, expected in cases:
            with self.subTest(completion=completion):
                self.assertEqual(gsm8k.extract_answer(completion), expected)

    def test_numeric_equivalence(self) -> None:
        self.assertTrue(gsm8k.is_equiv("5.5", "5.500000"))
        self.assertTrue(gsm8k.is_equiv("1/2", "0.5"))
        self.assertFalse(gsm8k.is_equiv("-80", "80"))

    def test_native_stop_fallback(self) -> None:
        self.assertEqual(
            gsm8k.postprocess_generation("The answer is 9.\n\nQ: Next"),
            "The answer is 9.\n\n",
        )
        self.assertEqual(
            gsm8k.postprocess_generation("The answer is 9. A: extra"),
            "The answer is 9. ",
        )

    def test_score_and_invalid(self) -> None:
        score = gsm8k.score_completion("Reasoning. The answer is 18.", "18")
        self.assertTrue(score["correct"])
        self.assertFalse(score["invalid"])
        invalid = gsm8k.score_completion("I cannot solve this.", "18")
        self.assertFalse(invalid["correct"])
        self.assertTrue(invalid["invalid"])

    def test_quality_gate_is_a_gross_correctness_guard(self) -> None:
        passing = gsm8k.quality_gate(
            {"evaluated": 100, "invalid": 1, "accuracy": 0.75},
            min_accuracy=0.5,
            max_invalid_fraction=0.05,
        )
        self.assertTrue(passing["passed"])

        wrong_model = gsm8k.quality_gate(
            {"evaluated": 100, "invalid": 0, "accuracy": 0.0},
            min_accuracy=0.5,
            max_invalid_fraction=0.05,
        )
        self.assertFalse(wrong_model["passed"])
        self.assertFalse(wrong_model["checks"]["accuracy_at_least_minimum"])


class SourceLoaderTests(unittest.TestCase):
    def test_local_sglang_source_contract_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            imported_file = root / "python" / "sglang" / "__init__.py"
            imported_file.parent.mkdir(parents=True)
            imported_file.write_text("# unit\n", encoding="utf-8")
            for relative in gsm8k.CRITICAL_SGLANG_SOURCE_FILES:
                source_file = root / relative
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_text("# unit\n", encoding="utf-8")

            source_files = gsm8k.require_local_sglang_source(root, imported_file)
            self.assertEqual(len(source_files), 4)

            missing_root = root / "missing"
            missing_import = missing_root / "python" / "sglang" / "__init__.py"
            missing_import.parent.mkdir(parents=True)
            missing_import.write_text("# unit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
                gsm8k.require_local_sglang_source(missing_root, missing_import)

            outside_import = root / "installed" / "sglang" / "__init__.py"
            outside_import.parent.mkdir(parents=True)
            outside_import.write_text("# unit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the required source"):
                gsm8k.require_local_sglang_source(root, outside_import)

    def test_loader_ignores_saved_generation(self) -> None:
        payload = {
            "id": "0:sample0",
            "source_row": 0,
            "completion_input": "Q: What is 1 + 1?\nA: Let's think step by step.",
            "ground_truth": "One plus one is two.\n#### 2",
            "generation": {"must": "not be accessed"},
        }
        encoded = (json.dumps(payload) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_bytes(encoded)
            rows, digest = gsm8k.load_source_rows(
                path,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                expected_count=1,
            )
        self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())
        self.assertEqual(
            rows,
            [
                {
                    "source_row": 0,
                    "source_id": "0:sample0",
                    "question": "What is 1 + 1?",
                    "ground_truth": "2",
                }
            ],
        )

    @unittest.skipUnless(
        gsm8k.DEFAULT_DATA_PATH.is_file(), "canonical source unavailable"
    )
    def test_canonical_local_source_hash_and_rows(self) -> None:
        rows, digest = gsm8k.load_source_rows(gsm8k.DEFAULT_DATA_PATH)
        self.assertEqual(digest, gsm8k.EXPECTED_DATA_SHA256)
        self.assertEqual(len(rows), 1319)
        self.assertEqual(rows[0]["source_row"], 0)
        self.assertEqual(rows[-1]["source_row"], 1318)


if __name__ == "__main__":
    unittest.main()
