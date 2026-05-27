"""Synthetic depth-controlled benchmark for the --no-cache-thoughts flag.

What this measures
------------------
Isolates the inference-speed impact of --no-cache-thoughts from the
agentic-loop variance that the search-agent bench suffers (Serper
latency, tool failures, model giving up early). Drives the two backends
through an identical synthetic N-turn conversation, then reports per-turn
latency, prefill work saved, concurrent throughput, and statistics.

Runs against the dual-sglang bench stand (one server with
--no-cache-thoughts, one baseline) and hits the RAW sglang endpoints
(/v1/chat/completions, not the server.py wrapper).

Metrics (12 of them, grouped):

Latency (per request):
  1. TTFT             — time to first content/reasoning_content delta
  2. step latency     — TTFT + decode time for the whole assistant turn
  3. decode latency   — step − TTFT  (sanity check; should match across backends)
  4. conversation     — sum of step latencies across N turns

Throughput (across requests):
  5. prefill tok/s    — (prompt_tokens − cached_tokens) / TTFT
  6. steps/sec @ B    — at concurrent batch size B
  7. rollouts/sec @ B — fixed-N rollouts completed per second at batch B

Compute work:
  8. uncached prefill — prompt_tokens − cached_tokens per step
  9. cumulative       — sum of (8) across N turns

Statistical confidence:
 11. mean ± stddev    — across K trials per condition
 12. 95% CI half-width on the per-step deltas
 13. first N where (NCT − baseline) ≥ 2σ (significance crossover)

Visualization:
 14. per-turn curves at B=1 for N ∈ {1,3,5,10,20,50}
 15. throughput curves at N=10 for B ∈ {1,4,16,64,256}

(Metric 10 — implied $/rollout — dropped per request.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from transformers import AutoTokenizer

# Vendored from search-agent-service/tito_buffer.py so this bench stays
# self-contained inside the sglang repo. Keep in sync with that file.

class TitoSession:
    def __init__(self, tokenizer, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.messages: list[dict] = []
        self.input_ids: list[int] = []
        self._seeded = False

    def seed(self, turn1_messages, prompt_token_ids, completion_token_ids,
             reasoning_tokens, assistant_message):
        self.messages = list(turn1_messages) + [assistant_message]
        self.input_ids = list(prompt_token_ids) + list(
            completion_token_ids[reasoning_tokens:]
        )
        self._seeded = True

    def append_assistant_turn(self, completion_token_ids, reasoning_tokens,
                              assistant_message):
        if not self._seeded:
            raise RuntimeError("TitoSession.seed() must be called first")
        self.input_ids.extend(completion_token_ids[reasoning_tokens:])
        self.messages.append(assistant_message)

    def append_user_turn(self, user_message):
        """Append a user turn + assistant generation prompt. The chat
        template is rendered before/after and the diff is appended to
        input_ids."""
        if not self._seeded:
            raise RuntimeError("TitoSession.seed() must be called first")
        before = self._render(add_generation_prompt=False)
        self.messages.append(user_message)
        after = self._render(add_generation_prompt=True)
        env_delta = list(after[len(before):])
        self.input_ids.extend(env_delta)
        return env_delta

    def _render(self, add_generation_prompt: bool) -> list[int]:
        # NOTE: BBQ/K2V3's tokenizer's apply_chat_template(tokenize=True)
        # returns a BatchEncoding wrapping a tokenizers.Encoding, NOT a
        # list[int] like most HF tokenizers. Render to text first, then
        # encode separately — that always returns list[int] regardless
        # of which tokenizer we're talking to.
        text = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.chat_template_kwargs,
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))


# Synthetic conversation construction
# -----------------------------------
# 20 seed topics × 6 follow-up templates rotated per turn. At temperature=0
# the same (seed, follow-ups) tuple gives identical responses, so trials
# get statistical independence from varying SEED topics, not from
# stochastic sampling.

# Realistic agentic conversation sizing
# --------------------------------------
# Real RL rollouts (search-agent, code-agent, etc.) have:
#   * Long system prompts: tool schemas + instructions, ~2K tokens
#   * Substantial "tool result" content per turn: scraped pages, search
#     results, retrieved chunks — typically 2-5K tokens
#   * Per turn the conversation grows by ~3-5K tokens
# The earlier ~500-token defaults didn't exercise the prefill regime
# where cache savings dominate. These constants target ~3-4K growth/turn
# so by N=50 the cumulative prompt is ~150-200K tokens (within the
# 524K context window of k2v3-7b but well past the small-prompt regime).

SYSTEM_PROMPT = """You are a deep research agent operating in a high-stakes investigation. Your task is to analyze evidence, reconcile sources, and produce thorough, well-cited answers.

PROTOCOL — STRICT ADHERENCE REQUIRED
You are equipped with the following capabilities, applied in an interleaved manner:
  1. web_search(query: str, num_results: int = 10) -> list of {title, link, snippet}
       Searches Google via Serper. Returns the top organic results plus an
       optional answer box and knowledge graph entry. Snippets are short.
  2. read_url(url: str) -> string
       Fetches a web page as clean markdown via Jina Reader. Long pages are
       truncated to 8000 characters. Use this to verify facts found via search.
  3. cross_reference(claim: str, sources: list[str]) -> {verdict, agreements, disagreements}
       Internal reconciliation tool — compares a claim across given sources
       and surfaces verbatim agreements and contradictions.

For EVERY substantive claim in your final answer you MUST cite:
  - The exact source URL
  - A verbatim quoted sentence from that source supporting the claim
  - The confidence interval based on how many independent sources corroborate

OUTPUT FORMAT — your final answer MUST contain three sections:
  Explanation: (your reasoning across the evidence, 3-5 paragraphs)
  Exact Answer: (your succinct final answer in 1-2 sentences)
  Confidence: (numeric score 0-100% plus a one-paragraph justification)

DO NOT emit a final answer until you have:
  - Performed at least 8 distinct tool calls
  - Read at least 3 different URLs in full
  - Verified each substantive claim against 2+ independent sources
  - Surfaced any disagreements between sources explicitly

ERROR HANDLING: if a read_url returns an error (paywall, 451, timeout),
substitute by searching for a primary-source equivalent and read THAT
instead. Never fall back to your training-data priors as a substitute
for a failed fetch — they are not citable.

You may use any combination of these capabilities, but you MUST keep each
reasoning step focused. After each tool call, summarize what you learned
and what gap remains. Do not pad your reasoning with restated context."""

# A realistic "tool result" — a chunk of public-domain text (~3K tokens
# of repeatable content). Using a stable string so token IDs are
# deterministic across runs. Each follow-up turn appends a different
# offset into this corpus so successive turns add new content (the
# cache hit on prior turns reflects real reuse).
_TOOL_RESULT_CORPUS = (
    "Federalist No. 10, written by James Madison and published on November 22, 1787, "
    "remains one of the most consequential essays in American political theory. Madison's "
    "principal concern was the threat that factions posed to a constitutional republic. "
    "He defined a faction as a number of citizens, whether amounting to a majority or "
    "minority of the whole, united by some common impulse of passion or interest adverse "
    "to the rights of other citizens or to the permanent and aggregate interests of the "
    "community. Madison conceded that the latent causes of faction are sown in the nature "
    "of man — that as long as people retain freedom of thought and unequal faculties "
    "for acquiring property, parties and factions will inevitably form. The question, "
    "therefore, was not how to suppress factions but how to control their effects. " * 6
    +
    "Madison rejected the option of removing the causes of faction. Eliminating liberty "
    "would cure the disease, but liberty was the very thing the new constitution was "
    "designed to secure. Forcing every citizen to have the same opinions, passions, and "
    "interests was impracticable. The only remaining course was to control the effects. "
    "Here Madison drew his crucial distinction between a pure democracy and a republic. "
    "In a pure democracy, where citizens administer government in person, a majority "
    "faction could readily oppress the minority — there was nothing in the structure to "
    "check passions. A republic, by contrast, delegated government to a small number of "
    "representatives chosen by the rest. This refinement of public views, Madison argued, "
    "filtered popular passions through the medium of a chosen body of citizens whose "
    "wisdom might best discern the true interest of the country. " * 6
    +
    "The second advantage Madison identified was the greater number of citizens and "
    "extent of territory which could be brought within the compass of republican as opposed "
    "to democratic government. The larger the society, the greater the variety of parties "
    "and interests, and the less probable that a majority of the whole would invade the "
    "rights of others. Even if such a common motive existed, communication and "
    "coordination across great distances would render it harder to act in concert. This "
    "argument inverted the classical assumption that small republics were inherently more "
    "stable than large ones. Montesquieu and others had argued that republican government "
    "required a small territory and a homogeneous citizenry to survive. Madison turned the "
    "argument inside out: it was precisely the LARGE size of the proposed federal republic, "
    "and the diversity of interests it would contain, that would protect liberty. " * 6
)


def _seed_topic_question(seed_idx: int) -> str:
    """A specific research question to be appended after the long system prompt
    and the seed corpus chunk. Varies by seed_idx for trial independence."""
    topics = [
        "transformer architecture's role in scaling laws for language models",
        "CRISPR-Cas9's off-target effects in therapeutic applications",
        "Raft consensus protocol's behavior under network partitions",
        "modern compiler optimization passes targeting SIMD instructions",
        "the 2008 financial crisis's effect on shadow banking regulation",
        "convolutional neural networks' equivariance properties",
        "20th century continental philosophy's response to phenomenology",
        "kidney dialysis's long-term effects on cardiovascular health",
        "modern web application request lifecycle under sustained load",
        "muscle contraction's calcium signaling at the sarcomere level",
        "Raft vs Paxos trade-offs in leader election latency",
        "quantum entanglement-based cryptographic protocols' security proofs",
        "plate tectonics' role in driving deep-ocean volcanic vents",
        "protein synthesis's regulation by ribosomal stalling",
        "climate change's effect on oceanic carbon sequestration capacity",
        "container orchestration's scheduling fairness guarantees",
        "Bayesian vs frequentist inference in high-dimensional regression",
        "public-key cryptography's post-quantum migration timeline",
        "modern CPU's branch prediction and speculative execution attacks",
        "human color vision's neural coding at the retinal level",
    ]
    topic = topics[seed_idx % len(topics)]
    return (
        f"Using the protocol above and the evidence excerpts I will provide in "
        f"subsequent turns, investigate: '{topic}'. Begin by outlining what evidence "
        f"you would need to gather and what aspects of the topic are most contested. "
        f"Wait for additional source material before forming conclusions."
    )


def _follow_up_text(turn_idx: int, seed_idx: int) -> str:
    """A realistic per-turn follow-up: includes a chunk of 'tool result'
    content (~3K tokens) plus an instruction. Different chunks per turn so
    successive turns add genuinely new prefill content."""
    chunk_size = 3000
    # Rotate the corpus offset by (turn + seed) so different conversations get
    # different sequences but the same conversation is deterministic.
    offset = ((turn_idx * 1009 + seed_idx * 311) % len(_TOOL_RESULT_CORPUS))
    chunk = (_TOOL_RESULT_CORPUS + _TOOL_RESULT_CORPUS)[offset : offset + chunk_size]
    return (
        f"<tool_result name=\"read_url\" url=\"https://example.org/source-{turn_idx}-{seed_idx}.html\">\n"
        f"{chunk}\n"
        f"</tool_result>\n\n"
        f"Incorporate this evidence into your analysis. Identify any claims it "
        f"supports or contradicts from your prior reasoning. Then decide what to "
        f"investigate next."
    )


def make_conversation_seed(seed_idx: int, n_turns: int) -> tuple[list[dict], list[dict]]:
    """Return (initial_user_messages, follow_up_messages).

    Initial = [system, user(seed question with corpus excerpt)]. System
    prompt is ~2K tokens; user message adds another ~3K. Initial prompt
    total ≈ 5K tokens.

    Follow-ups = subsequent user turns, each ~3K tokens of "tool result"
    content + a short instruction. After N turns the conversation has
    ~5K + 3K × (N-1) ≈ 3K × N tokens of prefill.
    """
    initial = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _seed_topic_question(seed_idx)},
    ]
    follow_ups = [
        {"role": "user", "content": _follow_up_text(turn_idx=t, seed_idx=seed_idx)}
        for t in range(1, n_turns)
    ]
    return initial, follow_ups


# Per-turn metrics container
# --------------------------

@dataclass
class TurnMetrics:
    turn: int
    ttft_s: float | None         # seconds; None if no token arrived
    step_s: float                # full step wall time
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    error: str | None = None

    @property
    def decode_s(self) -> float | None:
        if self.ttft_s is None:
            return None
        return self.step_s - self.ttft_s

    @property
    def uncached_prefill(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)


@dataclass
class ConversationResult:
    backend: str
    seed_idx: int
    n_turns: int
    turns: list[TurnMetrics] = field(default_factory=list)

    @property
    def total_step_s(self) -> float:
        return sum(t.step_s for t in self.turns)

    @property
    def cumulative_uncached_prefill(self) -> int:
        return sum(t.uncached_prefill for t in self.turns)


# Streaming client
# ----------------

async def _stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    body: dict,
    timeout_s: float = 300.0,
) -> tuple[TurnMetrics, str, list[int], int]:
    """POST stream=true, parse SSE, return (TurnMetrics, content, completion_token_ids, reasoning_tokens)."""
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    body["return_completion_token_ids"] = True
    # Intentionally NOT requesting return_prompt_token_ids — we derive
    # them client-side from the tokenizer. At N=40+ the prompt is 100K+
    # tokens, which serialized as a JSON int array exceeds aiohttp's
    # 131KB per-SSE-line limit and tears down the response stream.

    content_buf = ""
    reasoning_buf = ""
    completion_token_ids: list[int] = []
    prompt_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    ttft_s: float | None = None
    error: str | None = None

    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body,
                                 timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                err_body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {err_body[:200]}")
            async for raw_bytes in resp.content:
                line = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = details.get("cached_tokens", 0) or cached_tokens
                    reasoning_tokens = usage.get("reasoning_tokens", 0) or reasoning_tokens

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                if ttft_s is None and (delta.get("content")
                                       or delta.get("reasoning_content")):
                    ttft_s = time.perf_counter() - t0
                if delta.get("reasoning_content"):
                    reasoning_buf += delta["reasoning_content"]
                if delta.get("content"):
                    content_buf += delta["content"]
                if ch.get("finish_reason") is not None:
                    ctids = ch.get("completion_token_ids")
                    if ctids:
                        completion_token_ids = list(ctids)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"

    step_s = time.perf_counter() - t0
    metrics = TurnMetrics(
        turn=-1,  # set by caller
        ttft_s=ttft_s,
        step_s=step_s,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=len(completion_token_ids),
        reasoning_tokens=reasoning_tokens,
        error=error,
    )
    return metrics, content_buf, completion_token_ids, reasoning_tokens


async def run_conversation(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    tokenizer,
    seed_idx: int,
    n_turns: int,
    max_tokens: int,
    backend_label: str,
) -> ConversationResult:
    """Run one N-turn conversation, return per-turn metrics.

    Both backends always use TITO: turn 1 sends messages=, turn N+1 sends
    input_ids= from the TITO buffer. The only difference between backends
    is the server-side --no-cache-thoughts flag, so this is apples-to-
    apples — both servers see identical input_ids per turn, and at
    temperature=0 should produce byte-identical outputs.
    """
    result = ConversationResult(backend=backend_label, seed_idx=seed_idx, n_turns=n_turns)
    initial, follow_ups = make_conversation_seed(seed_idx, n_turns)
    tito: TitoSession | None = None

    for turn in range(1, n_turns + 1):
        if tito is not None:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "ignored when input_ids set"}],
                "input_ids": list(tito.input_ids),
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
        else:
            # Turn 1: still send messages= so the server tokenizes
            # naturally; we seed the TITO buffer from this turn's
            # response.
            body = {
                "model": model,
                "messages": list(initial),
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }

        metrics, content, completion_ids, reasoning_n = await _stream_chat(session, url, body)
        metrics.turn = turn
        result.turns.append(metrics)

        if metrics.error or turn == n_turns:
            break

        # Build the next user turn from the follow-ups list.
        assistant_msg = {"role": "assistant", "content": content}
        next_user_msg = follow_ups[turn - 1]  # follow_ups[0] is the turn-2 user message

        if tito is None:
            # Seed from turn 1's server response. Re-derive prompt_token_ids
            # client-side via render-then-encode (apply_chat_template with
            # tokenize=True returns BatchEncoding on this tokenizer, not
            # list[int]). At temperature=0 + same tokenizer, client and
            # server IDs match.
            seed_text = tokenizer.apply_chat_template(
                initial, tokenize=False, add_generation_prompt=True,
            )
            seed_prompt_ids = list(
                tokenizer.encode(seed_text, add_special_tokens=False)
            )
            tito = TitoSession(tokenizer)
            tito.seed(
                turn1_messages=initial,
                prompt_token_ids=seed_prompt_ids,
                completion_token_ids=completion_ids,
                reasoning_tokens=reasoning_n,
                assistant_message=assistant_msg,
            )
        else:
            tito.append_assistant_turn(
                completion_token_ids=completion_ids,
                reasoning_tokens=reasoning_n,
                assistant_message=assistant_msg,
            )
        tito.append_user_turn(next_user_msg)

    return result


# Stats helpers
# -------------

def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def ci95(xs: list[float]) -> float:
    """95% CI half-width (1.96σ/√n)."""
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5)


# Sweep runners
# -------------

def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


async def sweep_n(
    nct_url: str,
    bl_url: str,
    model: str,
    tokenizer,
    n_values: list[int],
    k_trials: int,
    max_tokens: int,
) -> dict:
    """N-sweep at batch=1: for each N value, run K trials per backend
    sequentially. Returns nested dict[n_value][backend] -> list[ConversationResult].

    Progress tracking: total work is sum(n) * K * 2 (since each turn takes
    ~constant time, total turns is a better unit than total conversations).
    After each conversation we update the running mean of seconds-per-turn
    and project the remaining turns.
    """
    out: dict[int, dict[str, list[ConversationResult]]] = {}
    total_turns_target = sum(n_values) * k_trials * 2  # × 2 backends
    completed_turns = 0
    completed_convs = 0
    total_convs = len(n_values) * k_trials * 2
    started_at = time.perf_counter()

    async with aiohttp.ClientSession() as sess:
        for n in n_values:
            n_idx = n_values.index(n) + 1
            print(
                f"\n[N-sweep] N={n}  K={k_trials}   "
                f"(condition {n_idx}/{len(n_values)})",
                flush=True,
            )
            out[n] = {"baseline": [], "no_cache_thoughts": []}
            for seed in range(k_trials):
                for label, url in [
                    ("baseline", bl_url),
                    ("no_cache_thoughts", nct_url),
                ]:
                    conv_start = time.perf_counter()
                    print(
                        f"  [{completed_convs + 1:>3}/{total_convs}] "
                        f"seed={seed} {label}...", end="", flush=True,
                    )
                    r = await run_conversation(
                        sess, url, model, tokenizer,
                        seed_idx=seed, n_turns=n,
                        max_tokens=max_tokens,
                        backend_label=label,
                    )
                    conv_wall = time.perf_counter() - conv_start
                    out[n][label].append(r)
                    completed_convs += 1
                    completed_turns += len(r.turns)
                    err_str = ""
                    if any(t.error for t in r.turns):
                        err_str = f" ERROR: {next(t.error for t in r.turns if t.error)}"
                    elapsed = time.perf_counter() - started_at
                    s_per_turn = elapsed / max(1, completed_turns)
                    remaining_turns = total_turns_target - completed_turns
                    eta = remaining_turns * s_per_turn
                    print(
                        f" {conv_wall:.1f}s ({len(r.turns)} turns){err_str}   "
                        f"[elapsed {_fmt_eta(elapsed)}  eta {_fmt_eta(eta)}]",
                        flush=True,
                    )
    return out


async def sweep_b(
    nct_url: str,
    bl_url: str,
    model: str,
    tokenizer,
    b_values: list[int],
    n_turns_fixed: int,
    k_trials: int,
    max_tokens: int,
) -> dict:
    """B-sweep at fixed N: for each B value, fire B concurrent conversations
    per backend. K trials = K repetitions of the B-batch.

    Progress tracking: total work is sum(b) * k_trials * 2 * N_fixed turns.
    After each trial we project remaining time at the running per-turn rate.
    Trials at large B finish faster per-turn (concurrency), so the ETA may
    be conservative early in the sweep.
    """
    out: dict[int, dict[str, list[list[ConversationResult]]]] = {}
    total_turns_target = sum(b_values) * k_trials * 2 * n_turns_fixed
    completed_turns = 0
    total_trials = len(b_values) * k_trials * 2
    completed_trials = 0
    started_at = time.perf_counter()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=512),
    ) as sess:
        for b in b_values:
            b_idx = b_values.index(b) + 1
            print(
                f"\n[B-sweep] B={b}  N={n_turns_fixed}  K={k_trials}   "
                f"(condition {b_idx}/{len(b_values)})  "
                f"— each trial = {b} concurrent {n_turns_fixed}-turn convs",
                flush=True,
            )
            out[b] = {"baseline": [], "no_cache_thoughts": []}
            for trial in range(k_trials):
                for label, url in [
                    ("baseline", bl_url),
                    ("no_cache_thoughts", nct_url),
                ]:
                    t0 = time.perf_counter()
                    tasks = [
                        run_conversation(
                            sess, url, model, tokenizer,
                            seed_idx=trial * b + i, n_turns=n_turns_fixed,
                            max_tokens=max_tokens,
                            backend_label=label,
                        )
                        for i in range(b)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=False)
                    wall = time.perf_counter() - t0
                    out[b][label].append(results)
                    n_errors = sum(1 for r in results if any(t.error for t in r.turns))
                    trial_turns = sum(len(r.turns) for r in results)
                    completed_turns += trial_turns
                    completed_trials += 1
                    elapsed = time.perf_counter() - started_at
                    s_per_turn = elapsed / max(1, completed_turns)
                    remaining_turns = total_turns_target - completed_turns
                    eta = remaining_turns * s_per_turn
                    print(
                        f"  [{completed_trials:>3}/{total_trials}] "
                        f"trial={trial} {label}: B={b} in {wall:.1f}s "
                        f"= {b / wall:.2f} rollouts/s  ({n_errors} errors)   "
                        f"[elapsed {_fmt_eta(elapsed)}  eta {_fmt_eta(eta)}]",
                        flush=True,
                    )
    return out


# Output formatters
# -----------------

def fmt_ms(s: float | None) -> str:
    return "  —" if s is None else f"{s * 1000:>6.1f}"


def print_n_sweep_tables(n_sweep: dict, n_values: list[int]) -> None:
    print("\n" + "=" * 100)
    print("N-sweep — per-turn metrics, B=1, mean ± stddev across K seeds")
    print("=" * 100)
    for n in n_values:
        bl_runs = n_sweep[n]["baseline"]
        nct_runs = n_sweep[n]["no_cache_thoughts"]
        if not bl_runs or not nct_runs:
            continue
        print(f"\n  N={n}")
        print(f"  {'turn':>4}  {'backend':<22} {'TTFT ms':>10} {'step ms':>10} "
              f"{'decode ms':>10} {'uncached pfx':>13}  {'CI95(TTFT)':>11}")
        for turn in range(1, n + 1):
            for label, runs in [("baseline", bl_runs), ("no_cache_thoughts", nct_runs)]:
                ttfts_ms = [r.turns[turn - 1].ttft_s * 1000
                            for r in runs if len(r.turns) >= turn
                            and r.turns[turn - 1].ttft_s is not None]
                steps_ms = [r.turns[turn - 1].step_s * 1000
                            for r in runs if len(r.turns) >= turn]
                decodes_ms = [(r.turns[turn - 1].decode_s) * 1000
                              for r in runs if len(r.turns) >= turn
                              and r.turns[turn - 1].decode_s is not None]
                uncached = [r.turns[turn - 1].uncached_prefill
                            for r in runs if len(r.turns) >= turn]
                ttft_m, ttft_s = mean_std(ttfts_ms)
                step_m, step_s_dev = mean_std(steps_ms)
                dec_m, _ = mean_std(decodes_ms)
                unc_m = statistics.mean(uncached) if uncached else 0
                print(
                    f"  {turn:>4}  {label:<22} {ttft_m:>7.1f}±{ttft_s:<2.0f} "
                    f"{step_m:>7.1f}±{step_s_dev:<2.0f} {dec_m:>9.1f}  "
                    f"{unc_m:>13.0f}  {ci95(ttfts_ms):>10.1f}"
                )

    print("\n" + "-" * 100)
    print("N-sweep — conversation totals (sum across N turns), mean across K seeds")
    print("-" * 100)
    print(f"  {'N':>3}  {'metric':<30}  {'baseline':>13}  {'NCT':>13}  "
          f"{'delta':>10}  {'speedup':>9}")
    for n in n_values:
        bl_runs = n_sweep[n]["baseline"]
        nct_runs = n_sweep[n]["no_cache_thoughts"]
        if not bl_runs or not nct_runs:
            continue
        bl_totals = [r.total_step_s for r in bl_runs]
        nct_totals = [r.total_step_s for r in nct_runs]
        bl_pf = [r.cumulative_uncached_prefill for r in bl_runs]
        nct_pf = [r.cumulative_uncached_prefill for r in nct_runs]
        bl_t, _ = mean_std(bl_totals)
        nct_t, _ = mean_std(nct_totals)
        bl_p, _ = mean_std([float(x) for x in bl_pf])
        nct_p, _ = mean_std([float(x) for x in nct_pf])
        speedup = bl_t / nct_t if nct_t else float("inf")
        print(f"  {n:>3}  {'conversation latency (s)':<30}  {bl_t:>13.2f}  "
              f"{nct_t:>13.2f}  {(nct_t - bl_t):>+9.2f}s  {speedup:>8.2f}x")
        print(f"  {' ':>3}  {'cumulative uncached prefill':<30}  {bl_p:>13.0f}  "
              f"{nct_p:>13.0f}  {(nct_p - bl_p):>+10.0f}  {bl_p / nct_p if nct_p else 0:>8.2f}x")

    # Significance crossover
    print("\n" + "-" * 100)
    print("Per-turn TTFT significance: smallest turn where (baseline − NCT) ≥ 2σ")
    print("-" * 100)
    for n in n_values:
        bl_runs = n_sweep[n]["baseline"]
        nct_runs = n_sweep[n]["no_cache_thoughts"]
        crossover_turn = None
        for turn in range(1, n + 1):
            bl_ttfts = [r.turns[turn - 1].ttft_s for r in bl_runs
                        if len(r.turns) >= turn and r.turns[turn - 1].ttft_s is not None]
            nct_ttfts = [r.turns[turn - 1].ttft_s for r in nct_runs
                         if len(r.turns) >= turn and r.turns[turn - 1].ttft_s is not None]
            if len(bl_ttfts) < 2 or len(nct_ttfts) < 2:
                continue
            delta = statistics.mean(bl_ttfts) - statistics.mean(nct_ttfts)
            pooled_sd = ((statistics.variance(bl_ttfts) + statistics.variance(nct_ttfts)) / 2) ** 0.5
            if delta >= 2 * pooled_sd:
                crossover_turn = turn
                break
        print(f"  N={n:>3}: significance crossover at turn = "
              f"{crossover_turn if crossover_turn else 'never within range'}")


def print_b_sweep_table(b_sweep: dict, b_values: list[int], n_fixed: int) -> None:
    print("\n" + "=" * 100)
    print(f"B-sweep — throughput at N={n_fixed} turns, mean across K trials")
    print("=" * 100)
    print(f"  {'B':>4}  {'backend':<22} {'wall s':>9} {'rollouts/s':>11} "
          f"{'steps/s':>9} {'speedup':>9}")
    bl_throughput: dict[int, float] = {}
    for b in b_values:
        bl_trials = b_sweep[b]["baseline"]
        nct_trials = b_sweep[b]["no_cache_thoughts"]
        if not bl_trials or not nct_trials:
            continue
        # Aggregate across trials
        bl_walls = [sum(r.total_step_s for r in trial) / max(1, len(trial))
                    for trial in bl_trials]
        nct_walls = [sum(r.total_step_s for r in trial) / max(1, len(trial))
                     for trial in nct_trials]
        # rollouts/s = B / max(individual conversation latencies, since they run in parallel)
        bl_roll_per_s = [b / max(r.total_step_s for r in trial)
                         for trial in bl_trials if trial]
        nct_roll_per_s = [b / max(r.total_step_s for r in trial)
                          for trial in nct_trials if trial]
        bl_t, _ = mean_std(bl_walls)
        nct_t, _ = mean_std(nct_walls)
        bl_rps, _ = mean_std(bl_roll_per_s)
        nct_rps, _ = mean_std(nct_roll_per_s)
        bl_throughput[b] = bl_rps
        speedup = nct_rps / bl_rps if bl_rps else float("inf")
        print(f"  {b:>4}  {'baseline':<22} {bl_t:>8.2f}s {bl_rps:>10.2f} "
              f"{bl_rps * n_fixed:>8.1f}  {'1.00x':>9}")
        print(f"  {b:>4}  {'no_cache_thoughts':<22} {nct_t:>8.2f}s {nct_rps:>10.2f} "
              f"{nct_rps * n_fixed:>8.1f}  {speedup:>8.2f}x")


# Main
# ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nct-url", required=True,
                    help="Raw sglang endpoint with --no-cache-thoughts (e.g. http://node:30000/v1)")
    ap.add_argument("--baseline-url", required=True,
                    help="Raw sglang endpoint without the flag (e.g. http://node:30001/v1)")
    ap.add_argument("--model", required=True,
                    help="Model name as known to sglang (the path it was launched with, or the "
                         "served-model-name).")
    ap.add_argument("--tokenizer-path", default=None,
                    help="HF tokenizer path. Defaults to --model.")
    ap.add_argument("--n-values", type=int, nargs="+",
                    default=[1, 3, 5, 10, 20, 50])
    ap.add_argument("--b-values", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256])
    ap.add_argument("--k-trials-n", type=int, default=5,
                    help="Trials per N condition (B=1 sweep)")
    ap.add_argument("--k-trials-b", type=int, default=3,
                    help="Trials per B condition (N-fixed sweep)")
    ap.add_argument("--n-fixed-for-b-sweep", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Per-turn max_tokens cap; keeps step latency predictable")
    ap.add_argument("--skip-n-sweep", action="store_true")
    ap.add_argument("--skip-b-sweep", action="store_true")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    # Strip trailing /v1 if user passed it for /chat/completions URL construction.
    def _chat_url(base: str) -> str:
        return base.rstrip("/") + "/chat/completions" if not base.endswith("/chat/completions") else base

    nct_url = _chat_url(args.nct_url)
    bl_url = _chat_url(args.baseline_url)

    print(f"NCT endpoint:      {nct_url}", flush=True)
    print(f"baseline endpoint: {bl_url}", flush=True)
    print(f"model:             {args.model}", flush=True)
    print(f"N values:          {args.n_values}", flush=True)
    print(f"B values:          {args.b_values}", flush=True)
    print(f"K (N-sweep):       {args.k_trials_n}", flush=True)
    print(f"K (B-sweep):       {args.k_trials_b}", flush=True)
    print(f"N (fixed for B):   {args.n_fixed_for_b_sweep}", flush=True)
    print(f"max_tokens/turn:   {args.max_tokens}", flush=True)

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model, trust_remote_code=True,
    )

    n_sweep = None
    b_sweep = None
    if not args.skip_n_sweep:
        n_sweep = asyncio.run(sweep_n(
            nct_url, bl_url, args.model, tokenizer,
            args.n_values, args.k_trials_n, args.max_tokens,
        ))
        print_n_sweep_tables(n_sweep, args.n_values)
    if not args.skip_b_sweep:
        b_sweep = asyncio.run(sweep_b(
            nct_url, bl_url, args.model, tokenizer,
            args.b_values, args.n_fixed_for_b_sweep,
            args.k_trials_b, args.max_tokens,
        ))
        print_b_sweep_table(b_sweep, args.b_values, args.n_fixed_for_b_sweep)

    if args.output_json:
        # Flatten to a JSON-serializable shape.
        def _ser_run(r: ConversationResult) -> dict:
            return {
                "backend": r.backend, "seed_idx": r.seed_idx, "n_turns": r.n_turns,
                "turns": [
                    {"turn": t.turn, "ttft_s": t.ttft_s, "step_s": t.step_s,
                     "prompt_tokens": t.prompt_tokens, "cached_tokens": t.cached_tokens,
                     "completion_tokens": t.completion_tokens,
                     "reasoning_tokens": t.reasoning_tokens, "error": t.error}
                    for t in r.turns
                ],
            }
        out_obj = {}
        if n_sweep:
            out_obj["n_sweep"] = {
                str(n): {label: [_ser_run(r) for r in runs]
                         for label, runs in by_label.items()}
                for n, by_label in n_sweep.items()
            }
        if b_sweep:
            out_obj["b_sweep"] = {
                str(b): {label: [[_ser_run(r) for r in trial] for trial in trials]
                         for label, trials in by_label.items()}
                for b, by_label in b_sweep.items()
            }
        with open(args.output_json, "w") as f:
            json.dump(out_obj, f, indent=2, default=str)
        print(f"\nFull results written to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
