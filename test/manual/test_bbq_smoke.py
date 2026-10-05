"""Smoke-load BBQ-8B-Mid1 in SGLang with --reasoning-parser k2_v3 to confirm the
model architecture and reasoning parser are wired before designing the
--no-cache-thoughts E2E test around BBQ.

Run manually on a GPU host:
    python -m sglang.launch_server ... (see below)

Or just invoke this module:
    python test/manual/test_bbq_smoke.py
"""

import json
import os
import subprocess
import sys
import time

import requests

BBQ_PATH = "/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-mid3-final"
PORT = 30000
BASE_URL = f"http://127.0.0.1:{PORT}"


def main() -> int:
    assert os.path.isdir(BBQ_PATH), f"missing model dir: {BBQ_PATH}"

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        BBQ_PATH,
        "--reasoning-parser",
        "k2_v3",
        "--enable-cache-report",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--mem-fraction-static",
        "0.85",
        "--trust-remote-code",
    ]
    print("Launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # Wait for /health (up to 5 minutes).
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                # Server died.
                out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                print("SERVER EXITED EARLY:\n" + out[-4000:], flush=True)
                return 1
            time.sleep(2)
        else:
            print("HEALTH POLL TIMED OUT", flush=True)
            return 1
        print("Server is up.", flush=True)

        # Use the OpenAI-compatible chat completions endpoint so the chat template
        # applies (which for BBQ-Mid3 primes the assistant turn with <think>\n).
        payload = {
            "model": BBQ_PATH,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 12 * 7? Reason step by step.",
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]["message"]
        content = choice.get("content", "")
        reasoning = choice.get("reasoning_content", "")
        print("=== reasoning_content ===\n", reasoning[:2000], flush=True)
        print("=== content ===\n", content[:2000], flush=True)
        print("=== usage ===\n", json.dumps(body.get("usage"), indent=2), flush=True)
        # Hard check: the model emitted </think>, so the K2V3 parser separated reasoning.
        assert reasoning, "No reasoning_content — model didn't emit <think>...</think>"
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
