#!/usr/bin/env python3
"""CPU-only tests for fail-closed BBQ checkpoint index selection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_layout import resolve_checkpoint_layout

PINNED_1B_PATH = Path("/mnt/weka/shrd/k2m/junlin.chen/xllm_1b_final/model")
PINNED_7B_PATH = Path(
    "/mnt/weka/shrd/k2m/junlin.chen/ckpts/"
    "k2v3-7B_iso_attn_shared_small_phase1_sft_2199941/"
    "huggingface/checkpoint_0010000"
)


def write_index(root: Path, index_name: str, weight_map: dict[str, str]) -> None:
    (root / index_name).write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


class CheckpointLayoutTests(unittest.TestCase):
    def test_selects_complete_pytorch_bin_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "nested" / "pytorch_model-00001-of-00001.bin"
            shard.parent.mkdir()
            shard.write_bytes(b"bin")
            write_index(
                root,
                "pytorch_model.bin.index.json",
                {"model.weight": "nested/pytorch_model-00001-of-00001.bin"},
            )

            layout = resolve_checkpoint_layout(root)

            self.assertEqual(layout.format, "pytorch_bin")
            self.assertFalse(layout.use_safetensors)
            self.assertEqual(
                layout.as_dict()["shards"],
                ["nested/pytorch_model-00001-of-00001.bin"],
            )

    def test_selects_complete_safetensors_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model-00001-of-00001.safetensors").write_bytes(b"safe")
            write_index(
                root,
                "model.safetensors.index.json",
                {"model.weight": "model-00001-of-00001.safetensors"},
            )

            layout = resolve_checkpoint_layout(root)

            self.assertEqual(layout.format, "safetensors")
            self.assertTrue(layout.use_safetensors)
            self.assertEqual(len(layout.shard_paths), 1)

    def test_requires_exactly_one_supported_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ValueError, "exactly one supported weight index"
            ):
                resolve_checkpoint_layout(root)

            (root / "weights.bin").write_bytes(b"bin")
            (root / "weights.safetensors").write_bytes(b"safe")
            write_index(
                root,
                "pytorch_model.bin.index.json",
                {"bin": "weights.bin"},
            )
            write_index(
                root,
                "model.safetensors.index.json",
                {"safe": "weights.safetensors"},
            )
            with self.assertRaisesRegex(
                ValueError, "exactly one supported weight index"
            ):
                resolve_checkpoint_layout(root)

    def test_rejects_incomplete_or_mismatched_shards(self) -> None:
        cases = (
            ("missing.bin", None, "missing or empty shard"),
            ("empty.bin", b"", "missing or empty shard"),
            ("wrong.safetensors", b"safe", "invalid pytorch_bin shard"),
            ("../outside.bin", b"outside", "escapes the checkpoint directory"),
        )
        for shard_name, contents, error_pattern in cases:
            with self.subTest(shard_name=shard_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "model"
                    root.mkdir()
                    if contents is not None:
                        shard = root / shard_name
                        shard.parent.mkdir(parents=True, exist_ok=True)
                        shard.write_bytes(contents)
                    write_index(
                        root,
                        "pytorch_model.bin.index.json",
                        {"model.weight": shard_name},
                    )
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        resolve_checkpoint_layout(root)

    def test_rejects_index_symlink_outside_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "model"
            root.mkdir()
            outside = parent / "pytorch_model.bin.index.json"
            outside.write_text(json.dumps({"weight_map": {"x": "x.bin"}}))
            (root / "pytorch_model.bin.index.json").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "index escapes"):
                resolve_checkpoint_layout(root)

    @unittest.skipUnless(
        (PINNED_1B_PATH / "model.safetensors.index.json").is_file(),
        "pinned 1B artifact is not mounted",
    )
    def test_pinned_1b_layout_is_one_safetensors_shard(self) -> None:
        layout = resolve_checkpoint_layout(PINNED_1B_PATH)
        self.assertEqual(layout.format, "safetensors")
        self.assertEqual(len(layout.shard_paths), 1)

    @unittest.skipUnless(
        (PINNED_7B_PATH / "pytorch_model.bin.index.json").is_file(),
        "pinned 7B artifact is not mounted",
    )
    def test_pinned_7b_layout_is_36_pytorch_shards(self) -> None:
        layout = resolve_checkpoint_layout(PINNED_7B_PATH)
        self.assertEqual(layout.format, "pytorch_bin")
        self.assertEqual(len(layout.shard_paths), 36)


if __name__ == "__main__":
    unittest.main()
