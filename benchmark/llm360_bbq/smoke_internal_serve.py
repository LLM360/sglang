#!/usr/bin/env python3
"""Remote smoke test for the atomic BBQ internal-serving endpoint registry."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Any


EXPECTED_MODELS = (
    "bbq-1b",
    "bbq-4b",
    "bbq-7b",
    "bbq-32b",
    "k2-mova-36b",
    "k2-moe-375b",
)


def request_json(
    url: str,
    *,
    timeout: float,
    api_key: str | None,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any, float]:
    headers = {"Accept": "application/json"}
    data = None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        status = response.status
    elapsed = time.perf_counter() - started
    return status, json.loads(payload) if payload else None, elapsed


def smoke_one(
    endpoint: dict[str, Any], *, timeout: float, api_key: str | None
) -> dict[str, Any]:
    model = endpoint.get("model", "<unknown>")
    result: dict[str, Any] = {
        "model": model,
        "node": endpoint.get("node"),
        "base_url": endpoint.get("base_url"),
        "registry_status": endpoint.get("status"),
        "status": "FAIL",
    }
    try:
        if endpoint.get("status") != "READY":
            raise ValueError(f"registry status is {endpoint.get('status')!r}, not READY")
        required_auth = endpoint.get("authentication") == "bearer"
        if required_auth and not api_key:
            raise ValueError("endpoint requires a bearer key; pass --api-key-file")

        health_request = urllib.request.Request(endpoint["health_url"])
        started = time.perf_counter()
        with urllib.request.urlopen(health_request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"health returned HTTP {response.status}")
            response.read()
        result["health_seconds"] = time.perf_counter() - started

        _, models, models_seconds = request_json(
            endpoint["models_url"], timeout=timeout, api_key=api_key
        )
        ids = [item.get("id") for item in models.get("data", [])]
        if endpoint["served_model_name"] not in ids:
            raise ValueError(
                f"served model {endpoint['served_model_name']!r} missing from /v1/models: {ids!r}"
            )
        result["models_seconds"] = models_seconds

        _, completion, completion_seconds = request_json(
            endpoint["completions_url"],
            timeout=timeout,
            api_key=api_key,
            body={
                "model": endpoint["served_model_name"],
                "prompt": "The answer is",
                "temperature": 0,
                "max_tokens": 4,
            },
        )
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"completion response has no choices: {completion!r}")
        if not isinstance(choices[0].get("text"), str):
            raise ValueError(f"completion text is not a string: {choices[0]!r}")
        result.update(
            {
                "status": "PASS",
                "completion_seconds": completion_seconds,
                "completion_text": choices[0]["text"],
            }
        )
        if endpoint.get("chat_completions_url"):
            _, chat, chat_seconds = request_json(
                endpoint["chat_completions_url"],
                timeout=timeout,
                api_key=api_key,
                body={
                    "model": endpoint["served_model_name"],
                    "messages": [{"role": "user", "content": "Reply briefly."}],
                    "temperature": 0,
                    "max_tokens": 4,
                },
            )
            chat_choices = chat.get("choices")
            if (
                not isinstance(chat_choices, list)
                or not chat_choices
                or not isinstance(chat_choices[0], dict)
                or not isinstance(chat_choices[0].get("message"), dict)
            ):
                raise ValueError(f"invalid chat response: {chat!r}")
            message = chat_choices[0]["message"]
            if not any(
                isinstance(message.get(field), str)
                for field in ("content", "reasoning_content")
            ):
                raise ValueError(f"chat response has no text field: {message!r}")
            result["chat_seconds"] = chat_seconds
            result["chat_content"] = message.get("content")
            result["chat_reasoning_content"] = message.get("reasoning_content")
    except (KeyError, ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/weka/home/yash.akhauri/projects/3sept_bbq_sglang/"
            "artifacts/internal-serving/endpoints"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=EXPECTED_MODELS,
        help="test only this model; repeat for multiple models",
    )
    parser.add_argument("--api-key-file", type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    models = tuple(args.model or EXPECTED_MODELS)
    api_key = None
    if args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise SystemExit(f"empty API-key file: {args.api_key_file}")

    endpoints = []
    missing = []
    for model in models:
        path = args.registry / f"{model}.json"
        try:
            endpoint = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            missing.append(str(path))
            continue
        if endpoint.get("model") != model:
            raise SystemExit(f"registry identity mismatch in {path}")
        endpoints.append(endpoint)
    if missing:
        raise SystemExit("missing endpoint records:\n" + "\n".join(missing))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = [
            pool.submit(smoke_one, endpoint, timeout=args.timeout, api_key=api_key)
            for endpoint in endpoints
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda item: models.index(item["model"]))

    summary = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "results": results,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
