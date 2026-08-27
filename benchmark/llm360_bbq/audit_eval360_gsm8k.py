#!/usr/bin/env python3
"""Fail-closed audit for the pinned Eval360 GSM8K chat evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scheduler.grader.base_parsers import _gsm8k_parser
from scheduler.grader.math import check_math_verify


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate on physical newlines. ``str.splitlines()`` also treats Unicode
    # separators embedded inside valid JSON strings as record boundaries.
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} contains a non-object JSONL row")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_generation(row: dict[str, Any], *, label: str) -> str:
    generations = row.get("generations")
    if not isinstance(generations, list) or len(generations) != 1:
        raise ValueError(f"{label} must contain exactly one generation")
    if not isinstance(generations[0], str):
        raise TypeError(f"{label} generation is not text")
    return generations[0]


def index_by_row(
    rows: list[dict[str, Any]], *, label: str, expected_rows: int
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("row")
        if not isinstance(row_id, int):
            raise TypeError(f"{label} contains a non-integer row ID: {row_id!r}")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate row ID {row_id}")
        indexed[row_id] = row
    expected_ids = set(range(expected_rows))
    if set(indexed) != expected_ids:
        raise ValueError(
            f"{label} row IDs are not exactly 0..{expected_rows - 1}"
        )
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=1319)
    parser.add_argument("--allow-stored-grade-mismatch", action="store_true")
    args = parser.parse_args()

    source_sha256 = sha256_file(args.source)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError(
            "pinned chat-five-shot source hash mismatch: "
            f"expected={args.expected_source_sha256}, actual={source_sha256}"
        )

    source_rows = read_jsonl(args.source)
    generation_rows = read_jsonl(args.generations)
    grade_rows = read_jsonl(args.grades)
    lengths = {
        "source": len(source_rows),
        "generations": len(generation_rows),
        "grades": len(grade_rows),
    }
    if set(lengths.values()) != {args.expected_rows}:
        raise ValueError(f"expected {args.expected_rows} aligned rows, got {lengths}")

    source_by_row = index_by_row(
        source_rows, label="source", expected_rows=args.expected_rows
    )
    generations_by_row = index_by_row(
        generation_rows, label="generations", expected_rows=args.expected_rows
    )
    grades_by_row = index_by_row(
        grade_rows, label="grades", expected_rows=args.expected_rows
    )

    recomputed_correct = 0
    stored_correct = 0
    graded_generation_mismatches = 0
    parser_mismatches = 0
    grade_mismatches = 0
    prompt_token_total = 0
    completion_token_total = 0
    max_completion_tokens = 0
    finish_reasons: Counter[str] = Counter()

    for row_id in range(args.expected_rows):
        source = source_by_row[row_id]
        generated = generations_by_row[row_id]
        graded = grades_by_row[row_id]
        label = f"row {row_id}"
        for field in ("row", "ground_truth", "completion_input", "chat_input"):
            if generated.get(field) != source.get(field):
                raise ValueError(f"{label} generated {field} diverges from pinned source")
            if graded.get(field) != source.get(field):
                raise ValueError(f"{label} graded {field} diverges from pinned source")

        chat_input = source.get("chat_input")
        if (
            not isinstance(chat_input, list)
            or len(chat_input) != 1
            or chat_input[0].get("role") != "user"
            or not isinstance(chat_input[0].get("content"), str)
        ):
            raise ValueError(f"{label} does not contain one canonical user chat turn")
        content = chat_input[0]["content"]
        if (
            content.count("Question:") != 6
            or content.count("Answer:") != 6
            or not content.rstrip().endswith("Answer:")
        ):
            raise ValueError(
                f"{label} is not operationally five-shot plus one target question"
            )

        generation = require_single_generation(generated, label=label)
        parsed = _gsm8k_parser(generation)
        parsed_for_grading = generation if parsed is None else parsed
        # Eval360's grader intentionally replaces ``generations`` with the
        # parser output before writing the grade row.  Preserve the raw text
        # check above against the generation artifact, and validate the
        # transformed grade payload against the independently replayed parser.
        if require_single_generation(graded, label=label) != parsed_for_grading:
            graded_generation_mismatches += 1
        stored_parsed = graded.get("parsed_generations")
        if not isinstance(stored_parsed, list) or len(stored_parsed) != 1:
            raise ValueError(f"{label} lacks exactly one stored parsed generation")
        if stored_parsed[0] != parsed:
            parser_mismatches += 1

        recomputed = bool(
            check_math_verify(
                prompt=source["completion_input"],
                generations=[parsed_for_grading],
                expected=source["ground_truth"],
            )["correct"][0]
        )
        stored_values = graded.get("correct")
        if not isinstance(stored_values, list) or len(stored_values) != 1:
            raise ValueError(f"{label} lacks exactly one stored correctness value")
        stored = bool(stored_values[0])
        recomputed_correct += int(recomputed)
        stored_correct += int(stored)
        grade_mismatches += int(recomputed != stored)

        finish = generated.get("finish_reasons")
        if not isinstance(finish, list) or len(finish) != 1:
            raise ValueError(f"{label} lacks exactly one finish reason")
        finish_reasons[str(finish[0])] += 1

        usage = generated.get("response_usage")
        if not isinstance(usage, list) or len(usage) != 1 or not isinstance(usage[0], dict):
            raise ValueError(f"{label} lacks exactly one response-usage record")
        prompt_tokens = int(usage[0].get("prompt_tokens", -1))
        completion_tokens = int(usage[0].get("completion_tokens", -1))
        if prompt_tokens < 1 or not 0 <= completion_tokens <= 16384:
            raise ValueError(f"{label} has invalid token usage: {usage[0]!r}")
        prompt_token_total += prompt_tokens
        completion_token_total += completion_tokens
        max_completion_tokens = max(max_completion_tokens, completion_tokens)

    strict_match = (
        graded_generation_mismatches == 0
        and parser_mismatches == 0
        and grade_mismatches == 0
    )
    if not strict_match and not args.allow_stored_grade_mismatch:
        raise ValueError(
            "Eval360 stored parser/grades disagree with an independent replay: "
            f"graded_generation_mismatches={graded_generation_mismatches}, "
            f"parser_mismatches={parser_mismatches}, "
            f"grade_mismatches={grade_mismatches}"
        )

    summary = {
        "status": "PASS" if strict_match else "PASS_WITH_ALLOWED_STORED_MISMATCH",
        "protocol": {
            "endpoint": "openai_chat_completions",
            "operational_fewshot_examples": 5,
            "target_questions_per_prompt": 1,
            "temperature": 0.0,
            "max_tokens": 16384,
            "parser": "gsm8k_base",
            "grader": "math-verify",
        },
        "source": {
            "path": str(args.source),
            "sha256": source_sha256,
            "rows": len(source_rows),
        },
        "accuracy": recomputed_correct / args.expected_rows,
        "accuracy_percent": 100.0 * recomputed_correct / args.expected_rows,
        "recomputed_correct": recomputed_correct,
        "stored_correct": stored_correct,
        "graded_generation_mismatches": graded_generation_mismatches,
        "parser_mismatches": parser_mismatches,
        "grade_mismatches": grade_mismatches,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "token_usage": {
            "prompt_tokens": prompt_token_total,
            "completion_tokens": completion_token_total,
            "max_completion_tokens": max_completion_tokens,
        },
        "artifacts": {
            "generations": str(args.generations),
            "grades": str(args.grades),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
