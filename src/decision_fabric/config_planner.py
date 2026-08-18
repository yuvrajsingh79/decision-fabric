"""Stage 5 — Configure.

Picking the model is half the saving. The other half is the config: effort,
thinking mode, max_tokens, caching, batch, fast mode. This module turns the
graph's derived requirements into a request the *specific* model will accept —
the config surface differs per model, and an invalid knob is a 400, not a
degraded answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ontology import ModelSpec

EFFORT_LADDER = ["low", "medium", "high", "xhigh", "max"]

# Required deep_reasoning level -> effort rung. Thinking tokens are billed as
# output, so this is the single biggest cost knob after model choice.
EFFORT_THRESHOLDS = [(0.60, "low"), (0.75, "medium"), (0.86, "high"), (0.93, "xhigh")]

# For pre-4.6 models that still take a fixed thinking budget.
LEGACY_BUDGETS = {"low": 0, "medium": 2048, "high": 6000, "xhigh": 12000, "max": 16000}

# Anthropic's minimum cacheable prefix. Below this, cache_control is a no-op
# that silently costs you the 1.25x write multiplier for nothing.
MIN_CACHEABLE_TOKENS = 1024

STREAM_THRESHOLD_TOKENS = 16_000

# Mirror of config/models.yaml `effort_output_multiplier`, used when a caller
# does not pass the live table. Thinking tokens count against max_tokens, so
# the ceiling has to leave room for the effort level this planner just chose —
# otherwise the plan truncates its own answer.
DEFAULT_EFFORT_MULTIPLIERS = {
    "none": 1.0, "low": 1.3, "medium": 1.9, "high": 3.0, "xhigh": 4.5, "max": 7.0,
}
MAX_TOKENS_HEADROOM = 1.4


@dataclass
class RequestPlan:
    model_id: str
    effort: str | None
    thinking: dict[str, Any] | None
    max_tokens: int
    stream: bool
    use_cache: bool
    cache_ttl: str | None
    batch: bool
    fast_mode: bool
    betas: list[str] = field(default_factory=list)
    use_beta_endpoint: bool = False
    fallbacks: Any = None
    # The effort rung this plan *represents*, even on models without an
    # `effort` knob. Cost projection needs it; the request must not carry it.
    effort_rung: str = "none"
    expected_output_tokens: int = 0
    rationale: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [self.model_id]
        if self.effort:
            bits.append(f"effort={self.effort}")
        if self.thinking:
            t = self.thinking.get("type")
            bits.append(f"thinking={t}" + (
                f"({self.thinking['budget_tokens']})" if "budget_tokens" in self.thinking else ""))
        else:
            bits.append("thinking=default")
        bits.append(f"max_tokens={self.max_tokens}")
        if self.use_cache:
            bits.append(f"cache={self.cache_ttl or '5m'}")
        if self.batch:
            bits.append("batch")
        if self.fast_mode:
            bits.append("fast")
        if self.stream:
            bits.append("stream")
        return " ".join(bits)

    def cost_effort_key(self) -> str:
        """Key into modifiers.effort_output_multiplier for this plan."""
        return self.effort_rung

    def to_request_kwargs(self, messages: list[dict], system: Any = None) -> dict[str, Any]:
        """Exactly the kwargs this model accepts — no knob it would reject."""
        kw: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kw["system"] = system
        if self.thinking is not None:
            kw["thinking"] = self.thinking
        if self.effort is not None:
            kw["output_config"] = {"effort": self.effort}
        if self.use_cache:
            kw["cache_control"] = ({"type": "ephemeral", "ttl": self.cache_ttl}
                                   if self.cache_ttl else {"type": "ephemeral"})
        if self.fast_mode:
            kw["speed"] = "fast"
        if self.betas:
            kw["betas"] = list(self.betas)
        if self.fallbacks is not None:
            kw["fallbacks"] = self.fallbacks
        return kw


def _effort_for(required_reasoning: float, bias: int) -> str:
    rung = len(EFFORT_THRESHOLDS)
    for thresh, name in EFFORT_THRESHOLDS:
        if required_reasoning < thresh:
            rung = EFFORT_LADDER.index(name)
            break
    rung = max(0, min(len(EFFORT_LADDER) - 1, rung + bias))
    return EFFORT_LADDER[rung]


def plan_request(
    spec: ModelSpec,
    *,
    required: dict[str, float],
    output_class_meta: dict[str, Any],
    policy: dict[str, Any],
    slo: dict[str, Any],
    stable_context_tokens: int = 0,
    context_reused: bool = False,
    effort_multipliers: dict[str, float] | None = None,
) -> RequestPlan:
    cs = spec.config_surface
    rationale: list[str] = []

    # --- effort ---
    req_reasoning = max(required.get("deep_reasoning", 0.0), required.get("math_symbolic", 0.0))
    effort_name = _effort_for(req_reasoning, int(policy.get("effort_bias", 0)))
    rationale.append(
        f"required deep_reasoning={req_reasoning:.2f} -> effort '{effort_name}' "
        f"(policy bias {policy.get('effort_bias', 0):+d})"
    )

    effort: str | None = None
    thinking: dict[str, Any] | None = None
    style = cs.get("thinking_style")

    if style == "always_on":
        # Fable 5: thinking is always on; any explicit `thinking` config is a 400.
        thinking = None
        effort = effort_name if cs.get("supports_effort") else None
        rationale.append("thinking always on for this model; depth controlled by effort only")
    elif style == "adaptive":
        # Never emit thinking:disabled — on Opus 5 it can leak tool calls into
        # visible text. Lowering effort is the cheaper, safer knob.
        thinking = {"type": "adaptive"}
        effort = effort_name if cs.get("supports_effort") else None
        rationale.append("adaptive thinking + effort (disabled-thinking avoided by policy)")
    elif style == "budget_tokens":
        # Pre-4.6 model: fixed budget, and `output_config.effort` would error.
        budget = LEGACY_BUDGETS[effort_name]
        if budget:
            budget = max(budget, int(cs.get("thinking_min_budget", 1024)))
            thinking = {"type": "enabled", "budget_tokens": budget}
            rationale.append(f"legacy thinking budget_tokens={budget} (model has no effort knob)")
        else:
            thinking = None
            rationale.append("thinking off — required reasoning is below this model's floor")
        effort = None

    # --- max_tokens ---
    mults = effort_multipliers or DEFAULT_EFFORT_MULTIPLIERS
    rung_key = effort_name if (effort or (thinking and "budget_tokens" in thinking)) else "none"
    expected_out = int(output_class_meta["expected_output_tokens"])
    needed = int(expected_out * float(mults.get(rung_key, 1.0)) * MAX_TOKENS_HEADROOM)
    max_tokens = max(int(output_class_meta["max_tokens"]), needed)
    if needed > int(output_class_meta["max_tokens"]):
        rationale.append(
            f"max_tokens raised to {min(max_tokens, spec.max_output_tokens)} — effort "
            f"'{rung_key}' bills ~{mults.get(rung_key, 1.0):.1f}x output and thinking "
            f"counts against the ceiling"
        )
    if thinking and "budget_tokens" in thinking:
        # budget_tokens must be strictly less than max_tokens.
        max_tokens = max(max_tokens, thinking["budget_tokens"] + 2048)
    max_tokens = min(max_tokens, spec.max_output_tokens)

    # --- caching ---
    use_cache = stable_context_tokens >= MIN_CACHEABLE_TOKENS and bool(cs.get("supports_caching"))
    cache_ttl = None
    if use_cache:
        if context_reused and cs.get("supports_1h_cache"):
            cache_ttl = "1h"
        rationale.append(
            f"{stable_context_tokens} stable prefix tokens >= {MIN_CACHEABLE_TOKENS} -> cache"
            + (" (1h ttl, prefix is reused)" if cache_ttl else "")
        )
    elif 0 < stable_context_tokens < MIN_CACHEABLE_TOKENS:
        rationale.append(
            f"stable prefix {stable_context_tokens} tok < {MIN_CACHEABLE_TOKENS} — caching skipped "
            "(would pay the 1.25x write and never hit)"
        )

    # --- batch / fast mode (mutually exclusive) ---
    batch = bool(policy.get("allow_batch") and slo.get("allow_batch") and cs.get("supports_batch"))
    if batch:
        rationale.append("SLO tolerates async -> Batch API at 50%")

    fast_mode = bool(
        not batch
        and policy.get("allow_fast_mode")
        and slo.get("allow_fast_mode")
        and cs.get("supports_fast_mode")
        and slo.get("max_relative_latency", 1.0) <= 0.25
    )
    betas: list[str] = []
    use_beta = False
    if fast_mode:
        betas.append("fast-mode-2026-02-01")
        use_beta = True
        rationale.append("realtime SLO -> fast mode (2.5x output speed at premium rates)")

    # --- refusal fallbacks on the top rung ---
    fallbacks = None
    if cs.get("supports_refusal_fallbacks"):
        betas.append("server-side-fallback-2026-07-01")
        fallbacks = "default"
        use_beta = True
        rationale.append("top-rung model -> server-side refusal fallbacks enabled")

    stream = max_tokens > STREAM_THRESHOLD_TOKENS
    if stream:
        rationale.append(f"max_tokens {max_tokens} > {STREAM_THRESHOLD_TOKENS} -> stream to avoid HTTP timeout")

    return RequestPlan(
        model_id=spec.id,
        effort=effort,
        thinking=thinking,
        max_tokens=max_tokens,
        stream=stream,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        batch=batch,
        fast_mode=fast_mode,
        betas=betas,
        use_beta_endpoint=use_beta,
        effort_rung=(effort_name if (effort or (thinking and "budget_tokens" in thinking)) else "none"),
        expected_output_tokens=int(output_class_meta["expected_output_tokens"]),
        fallbacks=fallbacks,
        rationale=rationale,
    )
