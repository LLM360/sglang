#!/usr/bin/env python3
"""Teacher-forced Hugging Face oracle for a persisted BBQ SGLang probe."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

MODEL_NAME = os.environ["BBQ_MODEL_NAME"]
MODEL_PATH = Path(os.environ["BBQ_MODEL_PATH"]).resolve()
SGLANG_RESULT = Path(os.environ["BBQ_SGLANG_RESULT"]).resolve()
OUTPUT_DIR = Path(os.environ["BBQ_HF_OUTPUT_DIR"]).resolve()
DEVICE_MAP_MODE = os.environ.get("BBQ_HF_DEVICE_MAP", "single")
MAX_MEMORY_GIB = int(os.environ.get("BBQ_HF_MAX_MEMORY_GIB", "120"))
TIE_MARGIN_NATS = float(os.environ.get("BBQ_TIE_MARGIN_NATS", "0.25"))
PRESERVE_ROUTER_BIASES_FP32 = (
    os.environ.get("BBQ_PRESERVE_ROUTER_BIASES_FP32", "1") == "1"
)


def _write(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "hf-parity.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(target)


def _parse_sglang_rows(
    logprob_rows: list[Any], top_rows: list[Any], output_ids: list[int]
) -> list[dict[str, Any]]:
    if len(logprob_rows) != len(output_ids) or len(top_rows) != len(output_ids):
        raise AssertionError(
            "SGLang output/logprob lengths disagree: "
            f"tokens={len(output_ids)}, logprobs={len(logprob_rows)}, "
            f"top={len(top_rows)}"
        )
    parsed = []
    for entry, top_entries, output_id in zip(logprob_rows, top_rows, output_ids):
        target_logprob, target_id = entry[:2]
        if target_logprob is None or int(target_id) != output_id:
            raise AssertionError("SGLang output-token logprob alignment is incorrect")
        top_ids = [int(top_entry[1]) for top_entry in top_entries]
        top_logprobs = [float(top_entry[0]) for top_entry in top_entries]
        parsed.append(
            {
                "target_id": output_id,
                "target_logprob": float(target_logprob),
                # Temperature-zero output IDs are authoritative when BF16 ties
                # can make the diagnostic top-k list choose another equal item.
                "argmax_id": output_id,
                "top1_margin_nats": top_logprobs[0] - top_logprobs[1],
                "top5_ids": top_ids,
                "top5_logprobs": top_logprobs,
            }
        )
    return parsed


def _restore_router_biases(model: Any) -> list[dict[str, Any]]:
    """Restore selection-only router biases at checkpoint precision.

    The model is otherwise loaded in BF16, but SGLang's fused top-k keeps the
    correction biases in FP32.  Reload only those tiny tensors so the HF oracle
    exercises the same routing contract.  Dense checkpoints simply return an
    empty list.
    """

    index_path = MODEL_PATH / "pytorch_model.bin.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    bias_names = sorted(
        name
        for name in weight_map
        if name.endswith(("self_attn.v_router.bias", "mlp.gate.bias"))
    )
    names_by_shard: dict[str, list[str]] = {}
    for name in bias_names:
        names_by_shard.setdefault(weight_map[name], []).append(name)

    restored = []
    for shard_name, names in sorted(names_by_shard.items()):
        state = torch.load(
            MODEL_PATH / shard_name,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for name in names:
            parameter = model.get_parameter(name)
            source = state[name]
            rounded = parameter.detach().float().cpu()
            source_float = source.float()
            quantization_error = (source_float - rounded).abs()
            loaded_dtype = str(parameter.dtype)
            parameter.data = source_float.to(parameter.device)
            restored.append(
                {
                    "name": name,
                    "shard": shard_name,
                    "source_dtype": str(source.dtype),
                    "loaded_dtype_before": loaded_dtype,
                    "effective_dtype": str(parameter.dtype),
                    "max_bf16_load_error": float(quantization_error.max()),
                    "mean_bf16_load_error": float(quantization_error.mean()),
                }
            )
        del state
    return restored


def _summary(logits: torch.Tensor, target_id: int) -> dict[str, Any]:
    source_dtype = str(logits.dtype)
    logits = logits.float()
    raw_values, _ = torch.topk(logits, k=2)
    raw_top1 = float(raw_values[0])
    bf16_ulp = (
        2.0 ** (math.floor(math.log2(abs(raw_top1))) - 7)
        if raw_top1 != 0.0
        else 2.0**-133
    )
    logprobs = torch.log_softmax(logits, dim=-1)
    values, indices = torch.topk(logprobs, k=5)
    return {
        "target_id": int(target_id),
        "target_logprob": float(logprobs[target_id]),
        "argmax_id": int(indices[0]),
        "top1_margin_nats": float(values[0] - values[1]),
        "source_logit_dtype": source_dtype,
        "top1_raw_logit": raw_top1,
        "top2_raw_logit": float(raw_values[1]),
        "bf16_ulp_at_top1": bf16_ulp,
        "top5_ids": [int(value) for value in indices.cpu()],
        "top5_logprobs": [float(value) for value in values.cpu()],
    }


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _unscaled_rope_parameters(
    config: Any, device: Any = None, *, dim: int
) -> tuple[torch.Tensor, float]:
    inv_freq = 1.0 / (
        float(config.rope_theta)
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


def _default_rope_parameters(config: Any, device: Any = None, **_: Any):
    return _unscaled_rope_parameters(config, device, dim=int(config.head_dim))


_XLLM_375B_PARTIAL_ROPE_EXPECTED = {
    "architectures": ["XllmForCausalLM"],
    "model_type": "xllm",
    "hidden_size": 6144,
    "head_dim": 128,
    "rope_head_dim": 64,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "num_hidden_layers": 61,
    "num_experts": 192,
    "mlp_only_layers": [0, 1, 2],
    "max_position_embeddings": 524288,
    "rope_theta": 10000000.0,
    "rope_scaling": None,
    "partial_rotary_factor": None,
}


def _xllm_375b_partial_rope_config_mismatches(
    config: Any,
) -> dict[str, dict[str, Any]]:
    expected = _XLLM_375B_PARTIAL_ROPE_EXPECTED
    actual = {key: getattr(config, key, None) for key in expected}

    # Transformers 5 aliases rope_scaling to rope_parameters. The pinned build's
    # standardizer converts the artifact's raw null into precisely this unscaled
    # default dictionary. Accept only that proven equivalent representation;
    # scaling factors, extra keys, or another rope type must still fail closed.
    normalized_default_rope = {
        "rope_type": "default",
        "rope_theta": expected["rope_theta"],
    }
    if (
        type(actual["rope_scaling"]) is dict
        and actual["rope_scaling"] == normalized_default_rope
    ):
        actual["rope_scaling"] = None

    return {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }


def _is_exact_xllm_375b_partial_rope_config(config: Any) -> bool:
    return not _xllm_375b_partial_rope_config_mismatches(config)


def _xllm_375b_partial_rope_parameters(
    config: Any, device: Any = None, **_: Any
) -> tuple[torch.Tensor, float]:
    """Recreate the exact Mid4 partial-RoPE buffer during TF5 initialization.

    The artifact constructs ``XllmRotaryEmbedding`` while temporarily changing
    ``config.head_dim`` from 128 to ``rope_head_dim`` 64, then restores 128. A
    Transformers-5 post-load initializer later calls the class-level default
    RoPE function with the restored config. Use the same 64 dimensions as the
    constructor, gated to the exact known artifact topology.
    """

    mismatches = _xllm_375b_partial_rope_config_mismatches(config)
    if mismatches:
        raise ValueError(
            "The xllm-375b partial-RoPE compatibility shim only supports the "
            "exact 61-layer/192-expert Mid4 configuration; "
            f"mismatches={mismatches!r}"
        )
    return _unscaled_rope_parameters(config, device, dim=int(config.rope_head_dim))


def _install_default_rope_alias() -> bool:
    """Restore the Transformers-4 unscaled RoPE registry key.

    The checkpoints' remote-code models use ``ROPE_INIT_FUNCTIONS["default"]``.
    The validation container has a Transformers-5 development build that removed
    this spelling.  Keep the compatibility shim in the read-only oracle process
    rather than modifying checkpoint code.
    """

    if "default" in ROPE_INIT_FUNCTIONS:
        return False

    ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
    return True


def _install_k2_aurora_strict_compat() -> bool:
    """Bypass an invalid artifact-local decorator in the independent oracle.

    The canonical Mid4 module applies ``@strict`` to a normal PretrainedConfig
    subclass, while current huggingface_hub requires a dataclass.  SGLang uses
    a raw-config native fallback; HF parity still needs to import the artifact's
    model implementation, so make that one decorator a no-op in this isolated
    process.  Keep the shim exact to the affected architecture.
    """

    raw_config = json.loads((MODEL_PATH / "config.json").read_text())
    if not (
        raw_config.get("model_type") == "k2_aurora"
        and raw_config.get("architectures") == ["K2AuroraForCausalLM"]
    ):
        return False

    import huggingface_hub.dataclasses as hub_dataclasses

    def _passthrough(cls: Any = None, **_: Any):
        if cls is None:
            return lambda target: target
        return cls

    hub_dataclasses.strict = _passthrough
    return True


def _install_output_recorder_compat() -> bool:
    """Provide the newer Transformers output-recorder declaration if absent.

    Canonical K2 Aurora remote code declares ``_can_record_outputs`` with
    ``transformers.utils.output_capturing.OutputRecorder``.  The validation
    container's older Transformers build does not provide that module and does
    not consume the declaration, so mirror the declaration type in this
    read-only oracle process solely to make the canonical module importable.
    """

    raw_config = json.loads((MODEL_PATH / "config.json").read_text())
    if not (
        raw_config.get("model_type") == "k2_aurora"
        and raw_config.get("architectures") == ["K2AuroraForCausalLM"]
    ):
        return False

    module_name = "transformers.utils.output_capturing"
    try:
        importlib.import_module(module_name)
        return False
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    module = types.ModuleType(module_name)

    @dataclass
    class OutputRecorder:
        target_class: type
        index: int = 0
        layer_name: str | None = None
        class_name: str | None = None
        capture_initial_hidden_state: bool = True

    module.OutputRecorder = OutputRecorder
    sys.modules[module_name] = module
    return True


def _patch_k2_aurora_mask_api(dynamic_module: Any) -> list[str]:
    """Adapt the artifact's newer mask call to the pinned oracle runtime.

    The canonical K2Aurora remote code uses the current Transformers spelling
    ``inputs_embeds`` and derives, but does not pass, ``cache_position``.  The
    validation image exposes the immediately preceding API spelling
    ``input_embeds`` and requires ``cache_position``.  Patch only the two
    function objects imported into the exact artifact module; checkpoint files
    and the installed Transformers package remain untouched.
    """

    raw_config = json.loads((MODEL_PATH / "config.json").read_text())
    if not (
        raw_config.get("model_type") == "k2_aurora"
        and raw_config.get("architectures") == ["K2AuroraForCausalLM"]
    ):
        return []

    patched = []
    for function_name in (
        "create_causal_mask",
        "create_sliding_window_causal_mask",
    ):
        original = getattr(dynamic_module, function_name)
        parameters = inspect.signature(original).parameters
        if "inputs_embeds" in parameters:
            continue
        if not {"input_embeds", "cache_position"}.issubset(parameters):
            raise RuntimeError(
                f"Unsupported Transformers mask API for {function_name}: "
                f"{inspect.signature(original)}"
            )

        def _compat(*args: Any, _original: Any = original, **kwargs: Any):
            if "inputs_embeds" in kwargs:
                if "input_embeds" in kwargs:
                    raise TypeError("Both inputs_embeds and input_embeds were provided")
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            if "cache_position" not in kwargs:
                position_ids = kwargs.get("position_ids")
                if position_ids is None or position_ids.ndim != 2:
                    raise TypeError(
                        "K2Aurora mask compatibility requires 2D position_ids "
                        "to reconstruct cache_position"
                    )
                if not torch.equal(
                    position_ids, position_ids[:1].expand_as(position_ids)
                ):
                    raise ValueError(
                        "K2Aurora mask compatibility requires one shared cache "
                        "position sequence across the batch"
                    )
                kwargs["cache_position"] = position_ids[0]
            return _original(*args, **kwargs)

        _compat.__name__ = original.__name__
        _compat.__doc__ = original.__doc__
        setattr(dynamic_module, function_name, _compat)
        patched.append(function_name)
    return patched


def _xllm_375b_manual_device_map(config: Any) -> dict[str, int]:
    """Place the exact 375B artifact across eight GPUs without disk offload.

    Transformers 5 computes the automatic device map from the artifact's FP32
    config before applying the requested BF16 load dtype.  The resulting
    inflated 1.5-TB estimate spills standard ``.bin`` weights to disk.  This
    explicit map balances the 58 large sparse layers as 7/7/7/7/7/7/8/8 and
    keeps both embedding heads accounted for.
    """

    expected = {
        "architectures": ["XllmForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 61,
        "num_experts": 192,
        "mlp_only_layers": [0, 1, 2],
    }
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected or torch.cuda.device_count() != 8:
        raise ValueError(
            "xllm375_manual is restricted to the exact 61-layer/192-expert "
            f"artifact on eight GPUs; expected={expected}, actual={actual}, "
            f"gpu_count={torch.cuda.device_count()}"
        )

    layer_stops = (10, 17, 24, 31, 38, 45, 53, 61)
    device_map: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 7,
        "lm_head": 7,
    }
    layer_start = 0
    for device, layer_stop in enumerate(layer_stops):
        for layer_index in range(layer_start, layer_stop):
            device_map[f"model.layers.{layer_index}"] = device
        layer_start = layer_stop
    return device_map


def _aggregate(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference_rows) != len(candidate_rows):
        raise AssertionError(
            f"Cannot compare {len(reference_rows)} and {len(candidate_rows)} rows"
        )
    signed_errors = [
        candidate["target_logprob"] - reference["target_logprob"]
        for reference, candidate in zip(reference_rows, candidate_rows)
    ]
    errors = [abs(value) for value in signed_errors]
    importance_ratios = [math.exp(value) for value in signed_errors]
    exact = [
        reference["argmax_id"] == candidate["argmax_id"]
        for reference, candidate in zip(reference_rows, candidate_rows)
    ]
    tie_aware = [
        match or reference["top1_margin_nats"] <= TIE_MARGIN_NATS
        for match, reference in zip(exact, reference_rows)
    ]
    overlaps = [
        len(set(reference["top5_ids"]) & set(candidate["top5_ids"]))
        for reference, candidate in zip(reference_rows, candidate_rows)
    ]
    mismatches = [
        {
            "position": index,
            "reference_argmax_id": reference["argmax_id"],
            "candidate_argmax_id": candidate["argmax_id"],
            "reference_top1_margin_nats": reference["top1_margin_nats"],
            "tie_aware": tie_aware[index],
        }
        for index, (match, reference, candidate) in enumerate(
            zip(exact, reference_rows, candidate_rows)
        )
        if not match
    ]
    return {
        "positions": len(errors),
        "target_logprob_absolute_error": {
            "mean": sum(errors) / len(errors),
            "p50_nearest_rank": _nearest_rank(errors, 0.50),
            "p95_nearest_rank": _nearest_rank(errors, 0.95),
            "p99_nearest_rank": _nearest_rank(errors, 0.99),
            "max": max(errors),
        },
        "target_logprob_signed_error_candidate_minus_reference": {
            "mean": sum(signed_errors) / len(signed_errors),
            "p50_nearest_rank": _nearest_rank(signed_errors, 0.50),
        },
        "importance_ratio_exp_candidate_minus_reference": {
            "mean": sum(importance_ratios) / len(importance_ratios),
            "median": statistics.median(importance_ratios),
        },
        "exact_greedy_agreement": sum(exact) / len(exact),
        "tie_aware_greedy_agreement": sum(tie_aware) / len(tie_aware),
        "non_tie_mismatch_fraction": sum(not value for value in tie_aware)
        / len(tie_aware),
        "tie_margin_nats": TIE_MARGIN_NATS,
        "top5_set_overlap_fraction": sum(overlaps) / (5 * len(overlaps)),
        "top5_set_overlap_mean_count": sum(overlaps) / len(overlaps),
        "mismatches": mismatches,
    }


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_target_logprob_abs_error_le_0.05": (
            metrics["target_logprob_absolute_error"]["mean"] <= 0.05
        ),
        "p95_target_logprob_abs_error_le_0.15": (
            metrics["target_logprob_absolute_error"]["p95_nearest_rank"] <= 0.15
        ),
        "top5_overlap_ge_0.95": metrics["top5_set_overlap_fraction"] >= 0.95,
        "tie_aware_greedy_ge_0.99": metrics["tie_aware_greedy_agreement"] >= 0.99,
        "non_tie_mismatch_le_0.01": metrics["non_tie_mismatch_fraction"] <= 0.01,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    installed_k2_aurora_strict_compat = _install_k2_aurora_strict_compat()
    installed_output_recorder_compat = _install_output_recorder_compat()
    installed_default_rope_alias = _install_default_rope_alias()
    sglang_record = json.loads(SGLANG_RESULT.read_text())
    if sglang_record.get("status") != "PASS":
        raise RuntimeError(f"SGLang probe did not pass: {SGLANG_RESULT}")
    probe_model = sglang_record.get("model", {})
    if probe_model.get("name") != MODEL_NAME:
        raise ValueError(
            "SGLang probe model name does not match the requested HF oracle: "
            f"probe={probe_model.get('name')!r}, requested={MODEL_NAME!r}"
        )
    probe_path = Path(str(probe_model.get("path", ""))).resolve()
    if probe_path != MODEL_PATH:
        raise ValueError(
            "SGLang probe checkpoint path does not match the requested HF "
            f"oracle: probe={probe_path}, requested={MODEL_PATH}"
        )
    early_prompt_rows = sglang_record.get("prompt_ids", [])
    early_broad = sglang_record.get("tests", {}).get("broad_cached_determinism", {})
    if (
        len(early_prompt_rows) != 16
        or any(
            len(early_broad.get(pass_name, [])) != 16
            for pass_name in ("first", "repeat")
        )
        or any(
            len(row.get("output_ids", [])) != 16
            for pass_name in ("first", "repeat")
            for row in early_broad.get(pass_name, [])
        )
    ):
        raise ValueError(
            "SGLang probe is incomplete; HF parity requires exactly 16 prompts "
            "and 16 results in both deterministic passes"
        )
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    config._attn_implementation = "eager"
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"],
        MODEL_PATH,
        local_files_only=True,
    )
    dynamic_module = sys.modules[model_class.__module__]
    patched_mask_functions = _patch_k2_aurora_mask_api(dynamic_module)
    patched_rotary_classes = []
    patched_xllm_partial_rope_classes = []
    exact_xllm_375b = (
        MODEL_NAME == "k2moe-375b" and _is_exact_xllm_375b_partial_rope_config(config)
    )
    if MODEL_NAME == "k2moe-375b" and not exact_xllm_375b:
        mismatches = _xllm_375b_partial_rope_config_mismatches(config)
        raise ValueError(
            "k2moe-375b requested, but its config does not match the exact "
            "61-layer/192-expert Mid4 partial-RoPE contract; "
            f"mismatches={mismatches!r}"
        )
    for name, value in vars(dynamic_module).items():
        if (
            isinstance(value, type)
            and name.endswith("RotaryEmbedding")
            and not hasattr(value, "compute_default_rope_parameters")
        ):
            rope_parameters = ROPE_INIT_FUNCTIONS["default"]
            if exact_xllm_375b and name == "XllmRotaryEmbedding":
                rope_parameters = _xllm_375b_partial_rope_parameters
                patched_xllm_partial_rope_classes.append(name)
            value.compute_default_rope_parameters = staticmethod(rope_parameters)
            patched_rotary_classes.append(name)
    if exact_xllm_375b and (
        patched_rotary_classes != ["XllmRotaryEmbedding"]
        or patched_xllm_partial_rope_classes != ["XllmRotaryEmbedding"]
    ):
        raise RuntimeError(
            "The exact xllm-375b TF5 rotary compatibility patch did not apply "
            "to exactly XllmRotaryEmbedding; refusing checkpoint I/O: "
            f"rotary={patched_rotary_classes}, "
            f"partial_rope={patched_xllm_partial_rope_classes}"
        )

    record: dict[str, Any] = {
        "status": "loading",
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "sglang_result": str(SGLANG_RESULT),
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "device_map_mode": DEVICE_MAP_MODE,
        "tie_margin_nats": TIE_MARGIN_NATS,
        "installed_k2_aurora_strict_compat": installed_k2_aurora_strict_compat,
        "installed_transformers_output_recorder_compat": (
            installed_output_recorder_compat
        ),
        "installed_transformers4_default_rope_alias": installed_default_rope_alias,
        "patched_remote_mask_functions": patched_mask_functions,
        "patched_remote_rotary_classes": patched_rotary_classes,
        "patched_xllm_375b_partial_rope_classes": (patched_xllm_partial_rope_classes),
        "prompts": [],
    }
    _write(record)

    if DEVICE_MAP_MODE == "single":
        device_map: Any = {"": 0}
        max_memory = None
    elif DEVICE_MAP_MODE == "balanced":
        device_map = "balanced"
        max_memory = {
            index: f"{MAX_MEMORY_GIB}GiB" for index in range(torch.cuda.device_count())
        }
    elif DEVICE_MAP_MODE == "xllm375_manual":
        device_map = _xllm_375b_manual_device_map(config)
        max_memory = None
    else:
        raise ValueError(f"Unsupported BBQ_HF_DEVICE_MAP={DEVICE_MAP_MODE!r}")
    record["requested_device_map"] = (
        {str(key): str(value) for key, value in device_map.items()}
        if isinstance(device_map, dict)
        else device_map
    )
    record["max_memory"] = max_memory
    _write(record)

    started = time.perf_counter()
    model = model_class.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory,
    )
    model.eval()
    record["load_seconds"] = time.perf_counter() - started
    record["hf_device_map"] = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    router_biases = _restore_router_biases(model) if PRESERVE_ROUTER_BIASES_FP32 else []
    record["router_bias_contract"] = {
        "preserve_fp32": PRESERVE_ROUTER_BIASES_FP32,
        "reason": "SGLang fused top-k keeps selection-only correction biases in FP32",
        "biases": router_biases,
    }
    _write(record)

    input_device = model.get_input_embeddings().weight.device
    all_sglang_rows: list[dict[str, Any]] = []
    all_hf_rows: list[dict[str, Any]] = []
    broad = sglang_record["tests"]["broad_cached_determinism"]
    prompt_rows = sglang_record["prompt_ids"]
    expected_prompts = 16
    expected_output_tokens = 16
    if len(prompt_rows) != expected_prompts:
        raise AssertionError(
            f"SGLang probe must contain {expected_prompts} prompts, got "
            f"{len(prompt_rows)}"
        )
    for pass_name in ("first", "repeat"):
        pass_rows = broad[pass_name]
        if len(pass_rows) != expected_prompts:
            raise AssertionError(
                f"SGLang probe {pass_name} pass must contain "
                f"{expected_prompts} results, got {len(pass_rows)}"
            )
        output_lengths = [len(row.get("output_ids", [])) for row in pass_rows]
        if output_lengths != [expected_output_tokens] * expected_prompts:
            raise AssertionError(
                f"SGLang probe {pass_name} output lengths are incomplete: "
                f"{output_lengths}"
            )

    for prompt_index in range(expected_prompts):
        prompt_ids = prompt_rows[prompt_index]
        sglang_output = broad["first"][prompt_index]
        prompt_ids = [int(value) for value in prompt_ids]
        output_ids = [int(value) for value in sglang_output["output_ids"]]
        meta = sglang_output["meta_info"]
        sglang_rows = _parse_sglang_rows(
            meta["output_token_logprobs"],
            meta["output_top_logprobs"],
            output_ids,
        )
        teacher_ids = prompt_ids + output_ids[:-1]
        with torch.inference_mode():
            started = time.perf_counter()
            output = model(
                input_ids=torch.tensor(
                    [teacher_ids], dtype=torch.long, device=input_device
                ),
                use_cache=False,
            )
            logits = output.logits[0]
            forward_seconds = time.perf_counter() - started
        start = len(prompt_ids) - 1
        hf_rows = [
            _summary(logits[start + offset], target_id)
            for offset, target_id in enumerate(output_ids)
        ]
        all_sglang_rows.extend(sglang_rows)
        all_hf_rows.extend(hf_rows)
        record["prompts"].append(
            {
                "prompt_index": prompt_index,
                "input_ids": prompt_ids,
                "output_ids": output_ids,
                "forward_seconds": forward_seconds,
                "sglang": sglang_rows,
                "hf": hf_rows,
            }
        )
        _write(record)

    expected_positions = expected_prompts * expected_output_tokens
    if (
        len(all_sglang_rows) != expected_positions
        or len(all_hf_rows) != expected_positions
    ):
        raise AssertionError(
            "HF parity must compare exactly 256 token positions; got "
            f"sglang={len(all_sglang_rows)}, hf={len(all_hf_rows)}"
        )
    metrics = _aggregate(all_hf_rows, all_sglang_rows)
    gate = _gate(metrics)
    record["metrics"] = metrics
    record["gate"] = gate
    record["status"] = "PASS" if gate["passed"] else "FAIL"
    _write(record)
    if not gate["passed"]:
        raise AssertionError(f"HF parity gate failed: {gate['checks']}")
    print(f"BBQ_HF_PARITY_{MODEL_NAME}=PASS", flush=True)


if __name__ == "__main__":
    main()
