"""Cost projection. Every number the router shows a CFO comes from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .ontology import ModelSpec
from .seed import effective_pricing


@dataclass
class CostBreakdown:
    model_id: str
    input_tokens: int
    cached_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    input_usd: float
    output_usd: float
    total_usd: float
    multipliers: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "in_tok": self.input_tokens,
            "cache_read_tok": self.cached_read_tokens,
            "cache_write_tok": self.cache_write_tokens,
            "out_tok": self.output_tokens,
            "usd": round(self.total_usd, 6),
        }


def project_cost(
    spec: ModelSpec,
    modifiers: dict[str, Any],
    *,
    fresh_input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    effort: str | None = None,
    batch: bool = False,
    fast_mode: bool = False,
    on: date | None = None,
) -> CostBreakdown:
    """Projected spend for one request under a specific config.

    Thinking tokens are billed as output, so `effort` scales the output side —
    this is why "same model, lower effort" is a real lever and not a rounding
    error.
    """
    in_rate, out_rate = effective_pricing(spec, on)
    if fast_mode:
        fm = spec.config_surface.get("fast_mode_pricing")
        if not fm:
            raise ValueError(f"{spec.id} has no fast-mode pricing")
        in_rate, out_rate = float(fm["input_per_mtok"]), float(fm["output_per_mtok"])

    eff_mult = float(modifiers["effort_output_multiplier"].get(effort or "none", 1.0))
    billed_output = output_tokens * eff_mult

    cache_read_mult = float(modifiers["cache_read_multiplier"])
    cache_write_mult = float(modifiers["cache_write_multiplier"])
    batch_mult = float(modifiers["batch_multiplier"]) if batch else 1.0

    input_usd = (
        fresh_input_tokens
        + cache_read_tokens * cache_read_mult
        + cache_write_tokens * cache_write_mult
    ) / 1_000_000 * in_rate * batch_mult
    output_usd = billed_output / 1_000_000 * out_rate * batch_mult

    return CostBreakdown(
        model_id=spec.id,
        input_tokens=fresh_input_tokens,
        cached_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=int(billed_output),
        input_usd=input_usd,
        output_usd=output_usd,
        total_usd=input_usd + output_usd,
        multipliers={
            "effort_output": eff_mult,
            "batch": batch_mult,
            "cache_read": cache_read_mult,
            "cache_write": cache_write_mult,
        },
    )


def actual_cost(
    spec: ModelSpec,
    modifiers: dict[str, Any],
    usage: Any,
    *,
    batch: bool = False,
    fast_mode: bool = False,
    on: date | None = None,
) -> CostBreakdown:
    """Cost from a real `response.usage` object (or a dict shaped like one)."""
    get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
    in_rate, out_rate = effective_pricing(spec, on)
    if fast_mode:
        fm = spec.config_surface["fast_mode_pricing"]
        in_rate, out_rate = float(fm["input_per_mtok"]), float(fm["output_per_mtok"])

    fresh = int(get("input_tokens"))
    creads = int(get("cache_read_input_tokens"))
    cwrites = int(get("cache_creation_input_tokens"))
    out = int(get("output_tokens"))

    batch_mult = float(modifiers["batch_multiplier"]) if batch else 1.0
    input_usd = (
        fresh
        + creads * float(modifiers["cache_read_multiplier"])
        + cwrites * float(modifiers["cache_write_multiplier"])
    ) / 1_000_000 * in_rate * batch_mult
    output_usd = out / 1_000_000 * out_rate * batch_mult

    return CostBreakdown(
        model_id=spec.id,
        input_tokens=fresh,
        cached_read_tokens=creads,
        cache_write_tokens=cwrites,
        output_tokens=out,
        input_usd=input_usd,
        output_usd=output_usd,
        total_usd=input_usd + output_usd,
        multipliers={"batch": batch_mult},
    )
