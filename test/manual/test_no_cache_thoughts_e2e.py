"""End-to-end test for --no-cache-thoughts on BBQ-8B-Mid3, TITO-style.

What this verifies
------------------
Two SGLang servers on the same BBQ checkpoint, one with --no-cache-thoughts
and one without. Both serve the same turn-1 reasoning request. Turn 2 builds
its input via the TITO protocol — raw token IDs, never re-tokenizing prior
content through the chat template — and excludes turn 1's thought slice
(output_token_ids[:reasoning_tokens]) from the running buffer.

Expected behavior on turn 2:
  * Baseline server (no flag): cached_tokens covers the original user prompt
    only. The cached path from turn 1's finish includes the priming +
    thoughts, but turn 2's buffer has the priming followed by the answer
    (no thoughts), so the match dies at the first thought slot.
  * --no-cache-thoughts server: the cached path was inserted with non-
    contiguous positions and excludes both thoughts AND the priming tail.
    Turn 2's buffer aligns: prompt-without-priming + answer-only matches
    the cached entry up through the entire answer. cached_tokens covers
    roughly len(prompt - priming) + len(answer).

Hard assertion: cached_tokens(--no-cache-thoughts) > cached_tokens(baseline).

Why TITO-style and not OpenAI chat-completions text passthrough
---------------------------------------------------------------
Round-tripping the model's answer through text and back through the chat
template's tokenizer drifts (BPE merges differ across string-concat
boundaries), so the answer's first token ID in turn 2's input does not
equal the first token ID stored in the cached path. The radix match dies
right after the assistant header — observable as cached_tokens stuck at
the prompt-prefix length regardless of whether the flag is set. The TITO
protocol prevents this by passing input_ids directly and never letting the
chat template re-tokenize prior assistant content. See reference_tito.md.

How to run (inside the agentic-rl image; needs 2 GPUs)
------------------------------------------------------
    srun --partition=main --time=00:30:00 -N 1 --gres=gpu:2 \\
        --container-image=/mnt/weka/shrd/k2pta/agentic_rl_images/agentic-rl-2eff86d1.sqsh \\
        --container-mounts=/mnt/weka:/mnt/weka,$PWD:/sglang \\
        bash -c "pip install --no-deps --break-system-packages -e /sglang/python && \\
                 cd /sglang/test/manual && \\
                 python3 -m unittest test_no_cache_thoughts_e2e -v"
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

import requests

BBQ_PATH = "/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-mid3-final"
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


def _chat_text(base_url: str, messages: list[dict], max_tokens: int) -> dict:
    """Turn 1 — chat completions with text, asking the server to return raw token IDs."""
    payload = {
        "model": BBQ_PATH,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "return_prompt_token_ids": True,
        "return_completion_token_ids": True,
        "return_meta_info": True,
    }
    r = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def _chat_token_ids(base_url: str, input_ids: list[int], max_tokens: int) -> dict:
    """Turn 2+ — chat completions with input_ids, bypassing the chat template."""
    payload = {
        "model": BBQ_PATH,
        # messages is required by the OpenAI shape but is ignored for tokenization
        # when input_ids is set; SGLang still uses it to derive stop tokens.
        "messages": [{"role": "user", "content": "ignored when input_ids is set"}],
        "input_ids": input_ids,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "return_prompt_token_ids": True,
        "return_completion_token_ids": True,
        "return_meta_info": True,
    }
    r = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def _tokenize(text: str) -> list[int]:
    """Tokenize a string fragment via HF AutoTokenizer with add_special_tokens=False."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BBQ_PATH, trust_remote_code=True)
    return tok.encode(text, add_special_tokens=False)


def _build_turn2_input_ids(turn1_resp: dict, new_user_text: str) -> list[int]:
    """Construct turn 2's input_ids by:
       1. Taking turn 1's prompt_token_ids verbatim (no re-tokenization).
       2. Appending turn 1's answer slice — output_token_ids[reasoning_tokens:] —
          so the thought tokens are stripped before they enter turn 2's buffer.
       3. Appending the env-delta tokens for the new user turn + assistant
          generation prompt. The K2V3 boundary patch (append \\n after
          <|im_end|>) is included because the model stops on <|im_end|>
          without emitting the trailing \\n that the chat template would have.
    """
    prompt_ids = turn1_resp["choices"][0]["prompt_token_ids"]
    completion_ids = turn1_resp["choices"][0]["completion_token_ids"]
    reasoning_tokens = turn1_resp["usage"]["reasoning_tokens"]

    # Strip the thought slice — keep only what follows </think>.
    answer_ids = completion_ids[reasoning_tokens:]

    # K2V3 / Qwen3 boundary patch: model stops on <|im_end|>; template emits
    # <|im_end|>\n. The missing \n is part of the env-delta tokenization below
    # (it's the leading \n of the env-delta string).
    env_delta = (
        f"\n<|im_start|>user\n{new_user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    env_delta_ids = _tokenize(env_delta)

    return list(prompt_ids) + list(answer_ids) + list(env_delta_ids)


class TestNoCacheThoughtsE2ETito(unittest.TestCase):

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

    def _dump_logs(self, label: str, proc: subprocess.Popen) -> None:
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

    def test_tito_cached_tokens_delta_on_turn2(self):
        user_turn1 = {"role": "user", "content": "What is 12 * 7? Reason carefully."}
        new_user_text = "Now multiply that result by 3."

        # Turn 1: chat completions over text, with Tito flags so we get raw IDs back.
        try:
            resp_nc = _chat_text(BASE_URL_NO_CACHE, [user_turn1], max_tokens=512)
            resp_bl = _chat_text(BASE_URL_BASELINE, [user_turn1], max_tokens=512)
        except Exception:
            self._dump_logs("no_cache (turn 1)", self.proc_no_cache)
            self._dump_logs("baseline (turn 1)", self.proc_baseline)
            raise

        # Turn 2: build input_ids via the Tito protocol, send with input_ids in body.
        input_ids_nc = _build_turn2_input_ids(resp_nc, new_user_text)
        input_ids_bl = _build_turn2_input_ids(resp_bl, new_user_text)

        try:
            resp_nc_t2 = _chat_token_ids(
                BASE_URL_NO_CACHE, input_ids_nc, max_tokens=256
            )
            resp_bl_t2 = _chat_token_ids(
                BASE_URL_BASELINE, input_ids_bl, max_tokens=256
            )
        except Exception:
            self._dump_logs("no_cache (turn 2)", self.proc_no_cache)
            self._dump_logs("baseline (turn 2)", self.proc_baseline)
            raise

        cached_nc = resp_nc_t2["usage"]["prompt_tokens_details"]["cached_tokens"]
        cached_bl = resp_bl_t2["usage"]["prompt_tokens_details"]["cached_tokens"]
        prompt_nc = resp_nc_t2["usage"]["prompt_tokens"]
        prompt_bl = resp_bl_t2["usage"]["prompt_tokens"]

        print(f"baseline:          prompt_tokens={prompt_bl} cached_tokens={cached_bl}")
        print(f"no-cache-thoughts: prompt_tokens={prompt_nc} cached_tokens={cached_nc}")

        # Both servers should see the same prompt_tokens (we constructed both inputs
        # identically; the answer text was identical at temperature=0).
        self.assertEqual(
            prompt_nc,
            prompt_bl,
            "turn 2 input lengths diverged — answer differed across servers?",
        )

        # Core assertion: --no-cache-thoughts caches more of turn 2's input than
        # baseline. Baseline's turn 1 cache included the priming + thoughts,
        # which turn 2's Tito buffer doesn't contain at the same slot, so its
        # cache match dies after the user prompt. --no-cache-thoughts's split
        # path put the answer at non-contiguous positions and stripped the
        # priming — turn 2 aligns through the answer.
        self.assertGreater(
            cached_nc,
            cached_bl,
            f"--no-cache-thoughts cached_tokens ({cached_nc}) should exceed "
            f"baseline ({cached_bl}) by approximately len(turn-1 answer)",
        )


if __name__ == "__main__":
    unittest.main()
