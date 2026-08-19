#!/usr/bin/env python3
"""Config-surface conformance check — the cheapest high-value live test.

The fabric's central claim is that encoding each model's config surface in the
graph means the planner *cannot* emit a knob that model rejects. Dry-run cannot
verify that: only the real API knows whether `thinking` on Fable 5, or
`output_config.effort` on Haiku 4.5, is a 400.

This sends one minimal request per (model, effort rung) the planner can actually
produce, with `max_tokens` clamped low, and reports which configs the API
accepts. Total cost is a few cents.

    python demo/live_check.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric.config_planner import plan_request  # noqa: E402
from decision_fabric.env import load_dotenv  # noqa: E402
from decision_fabric.executor import Executor  # noqa: E402
from decision_fabric.pricing import actual_cost  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402

# One requirement level per effort rung the planner maps onto.
PROBES = [
    ("trivial",  {"instruction_following": 0.40}),
    ("shallow",  {"deep_reasoning": 0.65}),
    ("moderate", {"deep_reasoning": 0.80}),
    ("deep",     {"deep_reasoning": 0.90}),
    ("research", {"deep_reasoning": 0.97, "math_symbolic": 0.97}),
]


def main() -> int:
    loaded = load_dotenv()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}\n")

    ex = Executor(dry_run=False)  # raises if no credentials resolve
    fab = Fabric()
    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]

    print(f"{'model':<20}{'requirement':<11}{'emitted config':<52}{'result':<10}{'$':>9}")
    print("-" * 104)

    total = 0.0
    failures = []
    for spec in fab.kg.models():
        for label, required in PROBES:
            plan = plan_request(
                spec,
                required=required,
                output_class_meta={"expected_output_tokens": 8, "max_tokens": 64},
                policy=fab.policy("balanced"),
                slo={"name": "background", **fab.latency_slos["background"]},
                effort_multipliers=fab.modifiers["effort_output_multiplier"],
            )
            # Keep the probe tiny: we are testing config ACCEPTANCE, not output.
            # High effort bills thinking as output, so an unclamped ceiling makes
            # a "cheap" conformance check cost real money.
            plan.max_tokens = 256
            if plan.thinking and "budget_tokens" in plan.thinking:
                # This model needs budget_tokens < max_tokens, and its documented
                # minimum budget is 1024 — so the ceiling must clear that.
                floor = int(spec.config_surface.get("thinking_min_budget", 1024))
                plan.thinking = {"type": "enabled", "budget_tokens": floor}
                plan.max_tokens = floor + 256
            plan.stream = False

            kw = plan.to_request_kwargs(msgs)
            shown = " ".join(
                f"{k}={v}" for k, v in kw.items()
                if k in ("thinking", "output_config", "speed", "betas", "fallbacks")
            ) or "(no optional knobs)"

            res = ex.execute(plan, spec, msgs)
            if res.error:
                status = "REJECTED"
                failures.append((spec.id, label, shown, res.error))
                cost = 0.0
            else:
                status = "accepted"
                cost = actual_cost(spec, fab.modifiers, res.usage).total_usd
                ex.record_spend("conformance", spec.id, res.usage, cost, label)
                total += cost
            print(f"{spec.id:<20}{label:<11}{shown[:50]:<52}{status:<10}{cost:>9.5f}")

    print("-" * 104)
    print(f"{'total':<83}{total:>9.5f}")

    if failures:
        print(f"\n{len(failures)} config(s) the API rejected — the graph's config surface is wrong:")
        for model, label, shown, err in failures:
            print(f"  {model} / {label}: {shown}\n    {err}")
        return 1

    print("\nAll emitted configs accepted. The per-model config surface in "
          "config/models.yaml matches the live API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
