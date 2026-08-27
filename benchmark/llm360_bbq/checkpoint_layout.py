#!/usr/bin/env python3
"""Fail-closed checkpoint index selection for the BBQ verification harness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointLayout:
    model_path: Path
    index_path: Path
    format: str
    shard_paths: tuple[Path, ...]

    @property
    def use_safetensors(self) -> bool:
        return self.format == "safetensors"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "index_path": str(self.index_path),
            "model_path": str(self.model_path),
            "shard_count": len(self.shard_paths),
            "shards": [
                str(path.relative_to(self.model_path)) for path in self.shard_paths
            ],
            "use_safetensors": self.use_safetensors,
        }


SUPPORTED_INDEXES = {
    "pytorch_model.bin.index.json": ("pytorch_bin", ".bin"),
    "model.safetensors.index.json": ("safetensors", ".safetensors"),
}


def _load_weight_map(index_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Checkpoint index is not valid JSON: {index_path}") from error
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Checkpoint index must contain a non-empty weight_map: {index_path}"
        )
    invalid = [
        name
        for name, shard in weight_map.items()
        if not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or not shard
    ]
    if invalid:
        raise ValueError(
            f"Checkpoint index contains invalid weight_map entries: {index_path}"
        )
    return weight_map


def resolve_checkpoint_layout(model_path: Path) -> CheckpointLayout:
    try:
        root = model_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Checkpoint path does not exist: {model_path}") from error
    if not root.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {root}")

    present = [root / name for name in SUPPORTED_INDEXES if (root / name).is_file()]
    if len(present) != 1:
        found = [path.name for path in present]
        raise ValueError(
            "Checkpoint must contain exactly one supported weight index; "
            f"supported={list(SUPPORTED_INDEXES)}, found={found}, path={root}"
        )

    selected_index = present[0]
    checkpoint_format, shard_suffix = SUPPORTED_INDEXES[selected_index.name]
    try:
        index_path = selected_index.resolve(strict=True)
        index_path.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Checkpoint index escapes the checkpoint directory: {selected_index}"
        ) from error
    weight_map = _load_weight_map(index_path)
    shard_names = sorted(set(weight_map.values()))
    shard_paths = []
    for shard_name in shard_names:
        relative = Path(shard_name)
        if relative.is_absolute() or relative.suffix != shard_suffix:
            raise ValueError(
                f"Checkpoint index references an invalid {checkpoint_format} shard: "
                f"{shard_name!r}"
            )
        shard_path = (root / relative).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Checkpoint index shard escapes the checkpoint directory: {shard_name!r}"
            ) from error
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            raise ValueError(
                f"Checkpoint index references a missing or empty shard: {shard_path}"
            )
        shard_paths.append(shard_path)

    return CheckpointLayout(
        model_path=root,
        index_path=index_path,
        format=checkpoint_format,
        shard_paths=tuple(shard_paths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    args = parser.parse_args()
    layout = resolve_checkpoint_layout(args.model_path)
    print(json.dumps(layout.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
