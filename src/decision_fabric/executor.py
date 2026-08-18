"""Stage 6 — Execute.

Live mode calls Claude through the official Anthropic SDK with exactly the
config the planner produced. Dry-run mode produces a deterministic simulated
response so the whole fabric — cascade, verification, escalation, learning —
can be demonstrated and tested without spending a cent.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from .config_planner import RequestPlan
from .features import estimate_tokens
from .ontology import ModelSpec

# Spread of the simulated quality draw. Wide enough that a marginal model
# fails sometimes, narrow enough that a strong one is not randomly punished.
SIM_QUALITY_SIGMA = 0.10

try:  # the SDK is optional in dry-run
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


@dataclass
class ExecutionResult:
    model_id: str
    text: str
    usage: dict[str, int]
    latency_s: float
    stop_reason: str | None
    simulated: bool
    sim_quality: float | None = None
    error: str | None = None
    request_id: str | None = None
    notes: list[str] = field(default_factory=list)


class Executor:
    """Set `dry_run=False` and export ANTHROPIC_API_KEY (or `ant auth login`)
    to hit the real API. Nothing else changes — the plan is identical."""

    def __init__(self, dry_run: bool | None = None) -> None:
        self._client = None
        if dry_run is None:
            dry_run = not self._credentials_available()
        self.dry_run = dry_run
        if not self.dry_run:
            if anthropic is None:
                raise RuntimeError("live mode requires `pip install anthropic`")
            # Zero-arg constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile — do not hardcode a key.
            self._client = anthropic.Anthropic()

    @staticmethod
    def _credentials_available() -> bool:
        """True only if something can actually authenticate.

        Checking for the ~/.config/anthropic *directory* is not enough: the
        `ant` CLI creates it on any invocation, including `ant auth status`,
        before a login has ever completed. That would flip the router into live
        mode with no usable credential, turning a clean dry-run into a run of
        auth failures. Require a non-empty credentials file instead.
        """
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        creds = os.path.expanduser("~/.config/anthropic/credentials")
        if not os.path.isdir(creds):
            return False
        return any(
            f.endswith(".json") and os.path.getsize(os.path.join(creds, f)) > 2
            for f in os.listdir(creds)
        )

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("no live client; Executor is in dry-run mode")
        return self._client

    # ------------------------------------------------------------ token counts

    def count_tokens(self, model_id: str, messages: list[dict], system: Any = None) -> int:
        """Real token count when live; a chars/token estimate otherwise."""
        if self.dry_run:
            blob = "".join(str(m.get("content", "")) for m in messages) + str(system or "")
            return estimate_tokens(blob)
        try:
            kw: dict[str, Any] = {"model": model_id, "messages": messages}
            if system is not None:
                kw["system"] = system
            return int(self.client.messages.count_tokens(**kw).input_tokens)
        except Exception:
            blob = "".join(str(m.get("content", "")) for m in messages) + str(system or "")
            return estimate_tokens(blob)

    # ---------------------------------------------------------------- execute

    def execute(
        self,
        plan: RequestPlan,
        spec: ModelSpec,
        messages: list[dict],
        system: Any = None,
        *,
        model_quality: float = 0.8,
    ) -> ExecutionResult:
        if self.dry_run:
            return self._simulate(plan, spec, messages, model_quality)
        return self._live(plan, messages, system)

    # ---------------------------------------------------------------- live

    def _live(self, plan: RequestPlan, messages: list[dict], system: Any) -> ExecutionResult:
        kwargs = plan.to_request_kwargs(messages, system)
        endpoint = self.client.beta.messages if plan.use_beta_endpoint else self.client.messages
        t0 = time.time()
        try:
            if plan.stream:
                with endpoint.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                msg = endpoint.create(**kwargs)
        except Exception as e:  # narrow below; broad here so one bad route
            return self._live_error(e, plan, time.time() - t0)  # never kills a batch

        latency = time.time() - t0
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        return ExecutionResult(
            model_id=getattr(msg, "model", plan.model_id),
            text=text,
            usage={
                "input_tokens": getattr(u, "input_tokens", 0) or 0,
                "output_tokens": getattr(u, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            },
            latency_s=round(latency, 3),
            stop_reason=getattr(msg, "stop_reason", None),
            simulated=False,
            request_id=getattr(msg, "_request_id", None),
            notes=(["refused: " + str(getattr(msg.stop_details, "category", "?"))]
                   if getattr(msg, "stop_reason", None) == "refusal" else []),
        )

    def _live_error(self, e: Exception, plan: RequestPlan, latency: float) -> ExecutionResult:
        if anthropic is not None:
            if isinstance(e, anthropic.BadRequestError):
                # Usually means the plan emitted a knob this model rejects —
                # a graph bug worth surfacing loudly, not retrying.
                kind = f"bad_request (config invalid for {plan.model_id}): {e}"
            elif isinstance(e, anthropic.RateLimitError):
                kind = "rate_limited"
            elif isinstance(e, anthropic.AuthenticationError):
                kind = "auth_failed"
            elif isinstance(e, anthropic.APIStatusError):
                kind = f"api_status_{e.status_code}"
            elif isinstance(e, anthropic.APIConnectionError):
                kind = "connection_error"
            else:
                kind = f"{type(e).__name__}: {e}"
        else:  # pragma: no cover
            kind = f"{type(e).__name__}: {e}"
        return ExecutionResult(
            model_id=plan.model_id, text="", usage={}, latency_s=round(latency, 3),
            stop_reason=None, simulated=False, error=kind,
        )

    # ---------------------------------------------------------------- simulate

    def _simulate(
        self, plan: RequestPlan, spec: ModelSpec, messages: list[dict], model_quality: float
    ) -> ExecutionResult:
        """Deterministic per (query, model, plan) so demos and tests are stable."""
        blob = "".join(str(m.get("content", "")) for m in messages)
        seed = int(hashlib.sha256((blob + plan.model_id + plan.summary()).encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)

        in_tok = estimate_tokens(blob)
        # Output length tracks max_tokens loosely; thinking-heavy configs emit more.
        thinking_factor = {"none": 1.0, "low": 1.3, "medium": 1.9,
                           "high": 3.0, "xhigh": 4.5, "max": 7.0}[plan.cost_effort_key()]
        # Centre the simulation on the task's expected answer length (the same
        # number the cost projection used) rather than on max_tokens, so a
        # generous ceiling does not fabricate spend that would never happen.
        base_out = plan.expected_output_tokens or max(200, plan.max_tokens // 8)
        out_tok = int(base_out * (0.75 + 0.5 * rng.random()) * thinking_factor)
        out_tok = min(out_tok, plan.max_tokens)

        cache_read = int(in_tok * 0.6) if plan.use_cache else 0
        fresh_in = in_tok - cache_read

        # Simulated answer quality: draw around the model's expected quality
        # for *this* requirement set. A model scored at 0.93 mostly clears a
        # 0.72 floor; one scored at 0.55 mostly does not. That is what makes the
        # cascade demo meaningful — cheap probes really do fail some of the
        # time, and the escalation path really does get exercised.
        sim_quality = round(max(0.0, min(1.0, rng.gauss(model_quality, SIM_QUALITY_SIGMA))), 3)

        latency = round(0.4 + float(spec.nfr.get("relative_latency", 0.5)) * 6 * thinking_factor / 3, 2)

        return ExecutionResult(
            model_id=plan.model_id,
            text=(f"[simulated {plan.model_id} response — {out_tok} output tokens; "
                  f"config: {plan.summary()}]"),
            usage={
                "input_tokens": fresh_in,
                "output_tokens": out_tok,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": in_tok if (plan.use_cache and not cache_read) else 0,
            },
            latency_s=latency,
            stop_reason="end_turn",
            simulated=True,
            sim_quality=sim_quality,
        )
