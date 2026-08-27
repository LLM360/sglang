#!/usr/bin/env python3
"""Deterministic, native-SGLang GSM8K evaluation for LLM360 BBQ models.

This intentionally uses the raw completion contract from native xLLM rather
than a chat template.  The eight exemplars, prompt separators, explicit BOS,
post-processing, and numerical answer extraction are pinned to the xLLM
references listed in ``NATIVE_REFERENCE_SHA256`` below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATA_PATH = Path(
    "/mnt/weka/shrd/k2m/lingjie.chen/eval/"
    "junlin_merged_7b_linear_b1_fullsuite_20260821/"
    "gsm8k/generations.jsonl"
)
EXPECTED_DATA_SHA256 = (
    "785c055981d4313d64120282c69a1259db8057817a1708cbb5842ab072aa2af4"
)
EXPECTED_SOURCE_ROWS = 1319
MAX_NEW_TOKENS = 512
STOP_STRINGS = ("Q: ", "A:")
RANDOM_SEED = 20260827
DEFAULT_MIN_ACCURACY = 0.50
DEFAULT_MAX_INVALID_FRACTION = 0.05

# These are provenance pins, not runtime dependencies.  The prompt builder and
# answer normalizer below were transcribed from these exact local revisions.
NATIVE_REFERENCE_SHA256 = {
    "xllm/tools/build_eval360_exact_datasets.py": (
        "0aa4aa058f1d339433e760ee32a8a8aeeae64cb72c952cbc21c62017f2f03704"
    ),
    "xllm/xllm/eval/task/gsm8k.py": (
        "de1030284776f32c4635d51723abb60fa8e8587f4f872629fbdefdae8f0cb931"
    ),
    "xllm/xllm/eval/utils.py": (
        "db16aa67ebb1ac5953d860f4efbd0e103e88653bb328376a2d8708db96adbd5b"
    ),
}
CRITICAL_SGLANG_SOURCE_FILES = (
    Path("python/sglang/srt/models/xllm.py"),
    Path("python/sglang/srt/layers/mova.py"),
    Path("python/sglang/srt/model_loader/weight_utils.py"),
    Path("python/sglang/srt/utils/hf_transformers_utils.py"),
)


GSM8K_FEWSHOTS = (
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        ),
        "answer": (
            "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more "
            "were planted. So there must have been 21 - 15 = 6.\n#### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?"
        ),
        "answer": (
            "Let's think step by step. There are originally 3 cars. 2 more cars arrive. "
            "3 + 2 = 5.\n#### 5"
        ),
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?"
        ),
        "answer": (
            "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
            "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
            "How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to "
            "Denny. So he gave Denny 20 - 12 = 8.\n#### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
            "How many toys does he have now?"
        ),
        "answer": (
            "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
            "then that is 4 more toys. 5 + 4 = 9.\n#### 9"
        ),
    },
    {
        "question": (
            "There were nine computers in the server room. Five more computers were installed each day, "
            "from monday to thursday. How many computers are now in the server room?"
        ),
        "answer": (
            "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers "
            "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29"
        ),
    },
    {
        "question": (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        ),
        "answer": (
            "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had "
            "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"
        ),
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": (
            "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
            "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"
        ),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_local_sglang_source(
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gsm8k_target(answer: str) -> str:
    """Convert a canonical GSM8K solution to native xLLM exemplar text."""
    parts = answer.rsplit("\n#### ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"GSM8K answer lacks final delimiter: {answer[-200:]!r}")
    solution, result = parts
    return f"{solution} The answer is {result}."


def ground_truth_final_answer(answer: str) -> str:
    """Extract the source final answer exactly as the xLLM dataset builder does."""
    parts = answer.rsplit("\n#### ", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"GSM8K answer lacks final delimiter: {answer[-200:]!r}")
    final_answer = parts[1].strip().replace(",", "")
    if not final_answer:
        raise ValueError("GSM8K answer has an empty final answer")
    return final_answer


def fewshot_prefix() -> str:
    return "\n\n".join(
        f"Q: {example['question']}\nA: {gsm8k_target(example['answer'])}"
        for example in GSM8K_FEWSHOTS
    )


def build_prompt(question: str) -> str:
    """Build the byte-exact native xLLM eight-shot raw-completion prompt."""
    if not question or question != question.strip():
        raise ValueError(
            f"Question must be non-empty without edge whitespace: {question!r}"
        )
    return f"{fewshot_prefix()}\n\nQ: {question}\nA:"


def question_from_completion_input(completion_input: str) -> str:
    """Extract only the test question from the provided generated-eval row."""
    prefix = "Q: "
    marker = "\nA:"
    if not completion_input.startswith(prefix) or marker not in completion_input:
        raise ValueError(
            f"Unexpected completion_input format: {completion_input[:200]!r}"
        )
    question, answer_stub = completion_input[len(prefix) :].rsplit(marker, maxsplit=1)
    if answer_stub.strip() not in {"", "Let's think step by step."}:
        raise ValueError(f"Unexpected completion_input answer stub: {answer_stub!r}")
    if not question:
        raise ValueError("Empty question in completion_input")
    return question


def load_source_rows(
    path: Path,
    expected_sha256: str = EXPECTED_DATA_SHA256,
    expected_count: int = EXPECTED_SOURCE_ROWS,
) -> tuple[list[dict[str, Any]], str]:
    """Load questions and gold answers, deliberately ignoring saved generations."""
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"GSM8K source hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                source_row = int(payload["source_row"])
                source_id = str(payload["id"])
                question = question_from_completion_input(
                    str(payload["completion_input"])
                )
                ground_truth = ground_truth_final_answer(str(payload["ground_truth"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid GSM8K source row at line {line_number}"
                ) from exc
            rows.append(
                {
                    "source_row": source_row,
                    "source_id": source_id,
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )

    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} source rows, found {len(rows)}")
    source_indices = [row["source_row"] for row in rows]
    if source_indices != list(range(expected_count)):
        raise ValueError("GSM8K source rows are not the ordered standard test split")
    return rows, actual_sha256


def _fix_latex_fracs(value: str) -> str:
    value = re.sub(r"\\frac\s*\{(\d+)\}\s*\{(\d+)\}", r"\1/\2", value)
    return re.sub(r"\\frac\s*(\d)\s*(\d)", r"\1/\2", value)


def _fix_sqrt(value: str) -> str:
    return re.sub(r"\\sqrt\s*([0-9a-zA-Z]+)", r"\\sqrt{\1}", value)


def _strip_string(value: Any) -> str:
    """Core xLLM math normalizer, kept local to avoid importing training code."""
    string = str(value).strip().replace("\n", "").rstrip(".")
    string = string.replace("\\!", "").replace("\\ ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")

    without_text_unit = re.sub(r"\\text\{.*?\}$", "", string).strip()
    if without_text_unit and without_text_unit != string:
        string = without_text_unit
    units = (
        "dollars",
        "dollar",
        "miles",
        "mile",
        "hours",
        "hour",
        "gallons",
        "gallon",
        "units",
        "unit",
        "degrees",
        "degree",
        "minutes",
        "minute",
        "seconds",
        "second",
        "lbs",
        "pounds",
        "GB",
        "MB",
        "KB",
        "TB",
        "kg",
        "lb",
        "m",
        "cm",
        "mm",
        "km",
        "meters",
        "meter",
        "gumballs",
        "roti",
    )
    string = re.sub(r"\s(" + "|".join(units) + r")\b", "", string, flags=re.IGNORECASE)
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\text", "").replace("x\\in", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"\\mbox\{.*?}", "", string)
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string).replace(" ", "")
    return _fix_latex_fracs(string)


def extract_answer(prediction: Any) -> str:
    """Match native xLLM's GSM8K final-number extraction."""
    pred_str = _fix_latex_fracs(str(prediction).replace("\u00a0", " "))
    if "boxed" in pred_str:
        try:
            boxed_tail = pred_str.split("boxed")[-1]
            if boxed_tail.startswith("{"):
                depth = 1
                answer = ""
                for character in boxed_tail[1:]:
                    if character == "{":
                        depth += 1
                    elif character == "}":
                        depth -= 1
                    if depth == 0:
                        break
                    answer += character
                return _strip_string(answer)
        except Exception:
            pass

    candidate_text = pred_str
    for pattern in (
        r"Final Answer?\s*(.*)",
        r"final answer?\s*(.*)",
        r"he answer is\s*(.*)",
        r"result is\s*(.*)",
    ):
        matches = re.findall(pattern, pred_str, re.IGNORECASE)
        if matches:
            candidate_text = matches[-1]
            break

    numbers = re.findall(
        r"(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)",
        candidate_text.replace(",", ""),
    )
    return _strip_string(numbers[-1]) if numbers else ""


def parse_value(value: str) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if "/" in text:
        try:
            parts = text.split("/")
            return float(parts[0]) / float(parts[1]) if len(parts) == 2 else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def is_equiv(prediction: str, gold: str) -> bool:
    if prediction == gold:
        return True
    pred_value = parse_value(prediction)
    gold_value = parse_value(gold)
    return (
        pred_value is not None
        and gold_value is not None
        and abs(pred_value - gold_value) < 1e-6
    )


def postprocess_generation(generation: str) -> str:
    """Apply native xLLM's stop fallback even though SGLang also stops early."""
    return generation.split("Q: ")[0].split("A:")[0]


def score_completion(completion: str, ground_truth: str) -> dict[str, Any]:
    processed = postprocess_generation(completion)
    prediction = extract_answer(processed)
    normalized_gold = extract_answer(ground_truth)
    invalid = not prediction or parse_value(prediction) is None
    return {
        "processed_completion": processed,
        "predicted_answer": prediction,
        "normalized_ground_truth": normalized_gold,
        "invalid": invalid,
        "correct": bool(not invalid and is_equiv(prediction, normalized_gold)),
    }


def batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def normalize_engine_results(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_record(output_dir: Path, record: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "gsm8k-sglang.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def update_summary(record: dict[str, Any], generation_seconds: float) -> None:
    results = record["results"]
    correct = sum(int(row["correct"]) for row in results)
    invalid = sum(int(row["invalid"]) for row in results)
    input_tokens = sum(int(row["input_tokens"]) for row in results)
    output_tokens = sum(int(row["output_tokens"]) for row in results)
    count = len(results)
    record["summary"] = {
        "evaluated": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "accuracy_percent": 100.0 * correct / count if count else 0.0,
        "invalid": invalid,
        "invalid_percent": 100.0 * invalid / count if count else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_seconds": generation_seconds,
        "requests_per_second": (
            count / generation_seconds if generation_seconds else 0.0
        ),
        "output_tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds else 0.0
        ),
        "input_plus_output_tokens_per_second": (
            (input_tokens + output_tokens) / generation_seconds
            if generation_seconds
            else 0.0
        ),
    }


def quality_gate(
    summary: dict[str, Any],
    *,
    min_accuracy: float,
    max_invalid_fraction: float,
) -> dict[str, Any]:
    evaluated = int(summary["evaluated"])
    invalid_fraction = int(summary["invalid"]) / evaluated if evaluated else 1.0
    checks = {
        "accuracy_at_least_minimum": float(summary["accuracy"]) >= min_accuracy,
        "invalid_fraction_at_most_maximum": (invalid_fraction <= max_invalid_fraction),
    }
    return {
        "purpose": (
            "gross wrong-model/corrupt-load guard; report accuracy separately and "
            "do not interpret this as baseline parity"
        ),
        "min_accuracy": min_accuracy,
        "max_invalid_fraction": max_invalid_fraction,
        "observed_invalid_fraction": invalid_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic native-SGLang GSM8K with xLLM's exact 8-shot prompt."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--expected-data-sha256", default=EXPECTED_DATA_SHA256)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--expected-arch", default="")
    parser.add_argument("--model-override-json", default="{}")
    parser.add_argument("--model-loader-extra-config-json", default="{}")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY)
    parser.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=DEFAULT_MAX_INVALID_FRACTION,
    )
    parser.add_argument(
        "--sglang-source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--sglang-commit", default="unknown")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.data_path = args.data_path.resolve()
    args.sglang_source_root = args.sglang_source_root.resolve()
    if not args.model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {args.model_path}")
    for filename in ("config.json", "pytorch_model.bin.index.json", "tokenizer.json"):
        if not (args.model_path / filename).is_file():
            raise ValueError(f"Required checkpoint file is missing: {filename}")
    if not args.data_path.is_file():
        raise ValueError(f"GSM8K source does not exist: {args.data_path}")
    if not 1 <= args.limit <= EXPECTED_SOURCE_ROWS:
        raise ValueError(f"--limit must be in [1, {EXPECTED_SOURCE_ROWS}]")
    if args.tp_size < 1 or args.batch_size < 1:
        raise ValueError("--tp-size and --batch-size must be positive")
    if args.max_running_requests < args.batch_size:
        raise ValueError("--max-running-requests must be at least --batch-size")
    if args.cuda_graph_max_bs < args.batch_size:
        raise ValueError("--cuda-graph-max-bs must be at least --batch-size")
    if not 0.0 <= args.min_accuracy <= 1.0:
        raise ValueError("--min-accuracy must be in [0, 1]")
    if not 0.0 <= args.max_invalid_fraction <= 1.0:
        raise ValueError("--max-invalid-fraction must be in [0, 1]")
    model_override = json.loads(args.model_override_json)
    if not isinstance(model_override, dict):
        raise ValueError("--model-override-json must decode to an object")
    loader_config = json.loads(args.model_loader_extra_config_json)
    if not isinstance(loader_config, dict):
        raise ValueError("--model-loader-extra-config-json must decode to an object")


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Heavy imports stay out of module scope so prompt/scoring tests are CPU-only.
    import torch
    import transformers
    from transformers import AutoTokenizer

    import sglang as sgl
    from sglang.srt.utils.hf_transformers_utils import get_config

    if not getattr(sgl, "__file__", None):
        raise ValueError("Imported sglang package has no filesystem source path")
    source_files = require_local_sglang_source(
        args.sglang_source_root, Path(sgl.__file__)
    )

    selected_rows, data_sha256 = load_source_rows(
        args.data_path, expected_sha256=args.expected_data_sha256
    )
    selected_rows = selected_rows[: args.limit]
    model_override = json.loads(args.model_override_json)
    loader_config = json.loads(args.model_loader_extra_config_json)
    config = get_config(
        str(args.model_path),
        trust_remote_code=True,
        model_override_args=model_override,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    if args.expected_arch and args.expected_arch not in (config.architectures or []):
        raise ValueError(
            f"Expected architecture {args.expected_arch}, got {config.architectures}"
        )

    bos_token_id = getattr(config, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("The native xLLM prompt contract requires a BOS token")
    if tokenizer.bos_token_id is not None and int(tokenizer.bos_token_id) != int(
        bos_token_id
    ):
        raise ValueError(
            f"Config/tokenizer BOS mismatch: {bos_token_id} vs {tokenizer.bos_token_id}"
        )

    prompts = [build_prompt(row["question"]) for row in selected_rows]
    prompt_ids = [
        [int(bos_token_id)]
        + [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        for prompt in prompts
    ]
    max_context = int(config.max_position_embeddings)
    longest_prompt = max(map(len, prompt_ids))
    if longest_prompt + MAX_NEW_TOKENS > max_context:
        raise ValueError(
            f"Longest prompt ({longest_prompt}) + {MAX_NEW_TOKENS} decode tokens "
            f"exceeds model context {max_context}"
        )
    if torch.cuda.device_count() < args.tp_size:
        raise ValueError(
            f"TP={args.tp_size} requested with only {torch.cuda.device_count()} visible GPUs"
        )

    checkpoint_hash_files = (
        "config.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    checkpoint_hashes = {
        filename: sha256_file(args.model_path / filename)
        for filename in checkpoint_hash_files
        if (args.model_path / filename).is_file()
    }
    source_hashes = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())
    }
    for source_file in source_files:
        source_hashes[str(source_file)] = sha256_file(source_file)

    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "loading",
        "mode": "full-release" if args.limit == EXPECTED_SOURCE_ROWS else "quick",
        "model": {
            "name": args.model_name,
            "path": str(args.model_path),
            "architecture": list(config.architectures or []),
            "model_type": str(config.model_type),
            "checkpoint_declared_dtype": str(getattr(config, "dtype", None)),
            "serving_dtype": "bfloat16",
            "max_position_embeddings": max_context,
            "checkpoint_metadata_sha256": checkpoint_hashes,
        },
        "dataset": {
            "path": str(args.data_path),
            "sha256": data_sha256,
            "expected_rows": EXPECTED_SOURCE_ROWS,
            "selected_rows": args.limit,
            "selection": "first_n_in_canonical_source_order",
        },
        "prompt_contract": {
            "mode": "raw_completion_no_chat_template",
            "fewshot": len(GSM8K_FEWSHOTS),
            "fewshot_prefix_sha256": sha256_text(fewshot_prefix()),
            "explicit_bos_token_id": int(bos_token_id),
            "max_new_tokens": MAX_NEW_TOKENS,
            "greedy": True,
            "stop_strings": list(STOP_STRINGS),
            "postprocess": "generation.split('Q: ')[0].split('A:')[0]",
            "tokenizer_note": (
                "checkpoint tokenizer as saved; add_special_tokens=False plus one explicit BOS"
            ),
        },
        "engine": {
            "implementation": "sglang",
            "tp_size": args.tp_size,
            "dtype": "bfloat16",
            "load_format": args.load_format,
            "model_loader_extra_config": loader_config,
            "model_override": model_override,
            "batch_size": args.batch_size,
            "max_total_tokens": args.max_total_tokens,
            "max_running_requests": args.max_running_requests,
            "chunked_prefill_size": args.chunked_prefill_size,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "mem_fraction_static": args.mem_fraction_static,
            "attention_backend": args.attention_backend,
            "enable_fused_qk_norm_rope": False,
            "random_seed": RANDOM_SEED,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "sglang": getattr(sgl, "__version__", "unknown"),
            "visible_gpu_count": torch.cuda.device_count(),
            "sglang_commit": args.sglang_commit,
        },
        "source_sha256": source_hashes,
        "native_reference_sha256": NATIVE_REFERENCE_SHA256,
        "results": [],
        "summary": {},
    }
    write_record(args.output_dir, record)

    engine = None
    generation_seconds = 0.0
    try:
        load_started = time.perf_counter()
        engine = sgl.Engine(
            model_path=str(args.model_path),
            tp_size=args.tp_size,
            dtype="bfloat16",
            trust_remote_code=True,
            load_format=args.load_format,
            model_impl="sglang",
            model_loader_extra_config=json.dumps(loader_config),
            random_seed=RANDOM_SEED,
            mem_fraction_static=args.mem_fraction_static,
            max_total_tokens=args.max_total_tokens,
            max_running_requests=args.max_running_requests,
            chunked_prefill_size=args.chunked_prefill_size,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            json_model_override_args=args.model_override_json,
            attention_backend=args.attention_backend,
            kv_cache_dtype="bfloat16",
            enable_fused_qk_norm_rope=False,
            log_level="info",
        )
        record["engine"]["initialization_seconds"] = time.perf_counter() - load_started
        record["engine"]["resolved_architecture"] = list(
            engine.server_args.get_model_config().hf_config.architectures or []
        )
        record["engine"]["resolved_attention_backend"] = str(
            engine.server_args.attention_backend
        )
        record["status"] = "running"
        write_record(args.output_dir, record)

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop": list(STOP_STRINGS),
            "ignore_eos": False,
            "skip_special_tokens": True,
        }
        indexed_inputs = list(enumerate(prompt_ids))
        for batch_index, batch in enumerate(batched(indexed_inputs, args.batch_size)):
            indices = [index for index, _ in batch]
            input_batch = [tokens for _, tokens in batch]
            batch_started = time.perf_counter()
            outputs = normalize_engine_results(
                engine.generate(
                    input_ids=input_batch,
                    sampling_params=sampling_params,
                )
            )
            batch_seconds = time.perf_counter() - batch_started
            generation_seconds += batch_seconds
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Batch {batch_index} returned {len(outputs)} outputs for {len(batch)} inputs"
                )

            for index, input_tokens, output in zip(indices, input_batch, outputs):
                completion = str(output.get("text", ""))
                scored = score_completion(
                    completion, selected_rows[index]["ground_truth"]
                )
                output_ids = [int(token) for token in output.get("output_ids", [])]
                meta_info = output.get("meta_info", {}) or {}
                record["results"].append(
                    {
                        "eval_index": index,
                        "source_row": selected_rows[index]["source_row"],
                        "source_id": selected_rows[index]["source_id"],
                        "question": selected_rows[index]["question"],
                        "ground_truth": selected_rows[index]["ground_truth"],
                        "prompt_sha256": sha256_text(prompts[index]),
                        "input_tokens": len(input_tokens),
                        "output_tokens": len(output_ids),
                        "completion": completion,
                        **scored,
                        "finish_reason": json_safe(meta_info.get("finish_reason")),
                        "batch_index": batch_index,
                        "batch_seconds": batch_seconds,
                    }
                )
            update_summary(record, generation_seconds)
            record["progress"] = {
                "completed": len(record["results"]),
                "total": args.limit,
            }
            write_record(args.output_dir, record)
            print(
                f"batch={batch_index} completed={len(record['results'])}/{args.limit} "
                f"accuracy={record['summary']['accuracy_percent']:.3f}% "
                f"invalid={record['summary']['invalid']} "
                f"output_tok_s={record['summary']['output_tokens_per_second']:.2f}",
                flush=True,
            )

        record["completed_unix_seconds"] = time.time()
        update_summary(record, generation_seconds)
        record["quality_gate"] = quality_gate(
            record["summary"],
            min_accuracy=args.min_accuracy,
            max_invalid_fraction=args.max_invalid_fraction,
        )
        record["status"] = "PASS" if record["quality_gate"]["passed"] else "FAIL"
        write_record(args.output_dir, record)
        if not record["quality_gate"]["passed"]:
            raise AssertionError(
                f"GSM8K gross correctness gate failed: "
                f"{record['quality_gate']['checks']}"
            )
    except BaseException as exc:
        if record.get("status") != "FAIL":
            record["status"] = "FAILED"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        update_summary(record, generation_seconds)
        write_record(args.output_dir, record)
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
