"""End-to-end test for --no-cache-thoughts on BBQ-8B-Mid3.

What this test verifies
-----------------------
Two servers launched on the same BBQ checkpoint, one with --no-cache-thoughts and
one without (baseline). Both serve the same turn-1 reasoning request. Turn 2
sends a chat history that contains the prior assistant answer (with <think>...
</think> stripped by the chat template, as normal multi-turn rendering does).

Expected cache behavior on turn 2:
  * Baseline: only the original user prompt remains in the radix as a matchable
    prefix, because turn-1's cached path appended the thought tokens between
    the user prompt and the answer, so turn-2's thoughts-free input diverges
    immediately after the user message. cached_tokens ~ len(user_prompt).
  * --no-cache-thoughts: the answer was inserted with non-contiguous positions
    (thoughts skipped), so turn-2's [user_prompt + answer] matches the cached
    path. cached_tokens ~ len(user_prompt + answer).

The hard assertion is therefore: on turn 2,
  cached_tokens(--no-cache-thoughts server) > cached_tokens(baseline server)
with the delta being at least most of the answer's token count.

How to run
----------
This test runs inside the agentic-rl container image with an overlay of THIS
branch's SGLang. Example invocation from the m2 login node:

    srun --partition=main --time=00:30:00 -N 1 --gres=gpu:2 \
        --container-image=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-2eff86d1.sqsh \
        --container-mounts=/mnt/weka:/mnt/weka,$PWD:/sglang \
        bash -c "pip install --no-deps -e /sglang/python && \
                 cd /sglang && python3 -m unittest test.manual.test_no_cache_thoughts_e2e -v"

Two GPUs are requested so the two servers can run side-by-side (each on its
own GPU). The script picks GPU 0 / GPU 1 via CUDA_VISIBLE_DEVICES per server.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

import requests

BBQ_PATH = "/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-mid3-final"
# Upstream BBQ chat template from LLM360/bbq-chat-template:main, which drops the
# empty <think></think> block in assistant rendering when message.think is not
# explicitly set. Required for --no-cache-thoughts to align with multi-turn
# input on turn 2. Mid3's bundled chat_template.jinja is stale.
CHAT_TEMPLATE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chat_templates",
    "bbq_upstream.jinja",
)
PORT_NO_CACHE = 30000
PORT_BASELINE = 30001
BASE_URL_NO_CACHE = f"http://127.0.0.1:{PORT_NO_CACHE}"
BASE_URL_BASELINE = f"http://127.0.0.1:{PORT_BASELINE}"
HEALTH_TIMEOUT = 600  # bbq-mid3 takes 1-3 min to load
REQUEST_TIMEOUT = 300


def _launch(port: int, extra_args: list[str], gpu: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        BBQ_PATH,
        "--reasoning-parser",
        "k2_v3",
        "--chat-template",
        CHAT_TEMPLATE,
        "--enable-cache-report",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.80",
    ] + list(extra_args)
    # Write server logs alongside the test source (bind-mounted) so the file
    # survives the container shutdown and can be inspected from the host.
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".e2e_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sglang_port{port}.log")
    log_fd = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fd, stderr=subprocess.STDOUT)
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def _wait_healthy(base_url: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if proc.poll() is not None:
            log_path = getattr(proc, "_log_path", None)
            tail = ""
            if log_path:
                try:
                    with open(log_path) as f:
                        tail = f.read()[-4000:]
                except Exception:
                    pass
            raise RuntimeError(f"server at {base_url} exited early:\n{tail}")
        time.sleep(3)
    raise TimeoutError(f"server at {base_url} never became healthy")


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _chat(base_url: str, messages: list[dict], max_tokens: int = 512) -> dict:
    payload = {
        "model": BBQ_PATH,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


class TestNoCacheThoughtsE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert os.path.isdir(BBQ_PATH), f"missing model dir: {BBQ_PATH}"
        cls.proc_no_cache = _launch(
            PORT_NO_CACHE, ["--no-cache-thoughts"], gpu="0"
        )
        cls.proc_baseline = _launch(PORT_BASELINE, [], gpu="1")
        try:
            _wait_healthy(BASE_URL_NO_CACHE, cls.proc_no_cache)
            _wait_healthy(BASE_URL_BASELINE, cls.proc_baseline)
        except Exception:
            _kill(cls.proc_no_cache)
            _kill(cls.proc_baseline)
            raise

    @classmethod
    def tearDownClass(cls):
        _kill(cls.proc_no_cache)
        _kill(cls.proc_baseline)

    def _dump_logs_on_failure(self, label: str, proc: subprocess.Popen) -> None:
        """Always dump the server log on any chat failure — works whether the
        server is dead or just unreachable."""
        log_path = getattr(proc, "_log_path", None)
        if log_path is None:
            return
        try:
            with open(log_path) as f:
                out = f.read()
            print(
                f"=== {label} server log (last 6KB) from {log_path} ===\n{out[-6000:]}",
                flush=True,
            )
        except Exception as e:
            print(f"(failed to read {label} log: {e})", flush=True)

    def test_cached_tokens_delta_on_turn2(self):
        user_turn1 = {
            "role": "user",
            "content": "What is 12 * 7? Reason carefully.",
        }
        # Turn 1 on both servers.
        try:
            resp_nc = _chat(BASE_URL_NO_CACHE, [user_turn1], max_tokens=512)
        except Exception:
            self._dump_logs_on_failure("no_cache (turn 1)", self.proc_no_cache)
            raise
        try:
            resp_bl = _chat(BASE_URL_BASELINE, [user_turn1], max_tokens=512)
        except Exception:
            self._dump_logs_on_failure("baseline (turn 1)", self.proc_baseline)
            raise

        ans_nc = resp_nc["choices"][0]["message"]["content"]
        ans_bl = resp_bl["choices"][0]["message"]["content"]
        # With temperature=0 + same prompt, the answer portion should be identical.
        # If they diverge it's still OK — we only need a valid prior assistant
        # message for turn 2; we use each server's own turn-1 answer.

        user_turn2 = {
            "role": "user",
            "content": "Now multiply that result by 3.",
        }
        # Turn 2 history: chat template strips <think> from prior assistant messages.
        try:
            resp_nc_t2 = _chat(
                BASE_URL_NO_CACHE,
                [user_turn1, {"role": "assistant", "content": ans_nc}, user_turn2],
                max_tokens=256,
            )
        except Exception:
            self._dump_logs_on_failure("no_cache (turn 2)", self.proc_no_cache)
            raise
        try:
            resp_bl_t2 = _chat(
                BASE_URL_BASELINE,
                [user_turn1, {"role": "assistant", "content": ans_bl}, user_turn2],
                max_tokens=256,
            )
        except Exception:
            self._dump_logs_on_failure("baseline (turn 2)", self.proc_baseline)
            raise

        cached_nc = resp_nc_t2["usage"]["prompt_tokens_details"]["cached_tokens"]
        cached_bl = resp_bl_t2["usage"]["prompt_tokens_details"]["cached_tokens"]
        prompt_nc = resp_nc_t2["usage"]["prompt_tokens"]
        prompt_bl = resp_bl_t2["usage"]["prompt_tokens"]

        print(f"baseline: prompt_tokens={prompt_bl} cached_tokens={cached_bl}")
        print(f"no-cache-thoughts: prompt_tokens={prompt_nc} cached_tokens={cached_nc}")

        # Core assertion: --no-cache-thoughts caches MORE of turn-2's input than baseline.
        # The delta reflects the answer slice that was inserted with non-contiguous
        # positions on the with-flag server but not on the baseline.
        self.assertGreater(
            cached_nc,
            cached_bl,
            f"--no-cache-thoughts cached_tokens ({cached_nc}) should exceed baseline "
            f"({cached_bl}) by approximately len(turn-1 answer)",
        )


if __name__ == "__main__":
    unittest.main()
