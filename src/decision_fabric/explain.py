"""Human-readable decision reports.

An enterprise cost-control layer that can't answer "why did you pick that
model" is a black box people turn off. Everything printed here is read back
out of the graph and the decision record — nothing is re-derived.
"""
from __future__ import annotations

from .router import RoutingDecision

BAR = "─" * 78


def _fmt_usd(x: float) -> str:
    return f"${x:,.6f}" if x < 1 else f"${x:,.4f}"


def render(d: RoutingDecision, *, show_trace: bool = True, show_answer: bool = False) -> str:
    L: list[str] = []
    L.append(BAR)
    q = d.query.replace("\n", " ")
    L.append(f"QUERY   {q[:70]}{'…' if len(q) > 70 else ''}")
    L.append(
        f"BOUND   task={d.features.task_type}  domain={d.features.domain}  "
        f"slo={d.slo_name}  policy={d.policy['name']}(floor {d.policy['quality_floor']:.2f})"
    )
    if d.features.signals:
        L.append(f"SIGNALS {', '.join(d.features.signals)}")
    if not d.features.verified:
        L.append(
            f"WARNING task type '{d.features.task_type}' is a heuristic GUESS "
            f"(confidence {d.features.task_confidence:.2f}); the LLM classifier "
            f"did not run. Routing was made conservative to compensate."
        )

    L.append("")
    L.append("REQUIREMENTS DERIVED FROM THE GRAPH")
    for cap, r in sorted(d.requirements.items(), key=lambda kv: -kv[1].level):
        seen, parts = set(), []
        for src in r.sources:
            node = src.replace("introduced by ", "").split(" ")[0]
            if node in seen:
                continue
            seen.add(node)
            parts.append(
                node.replace("task:", "").replace("signal:", "+").replace("domain:", "@")
            )
        srcs = ", ".join(parts)
        L.append(f"  {cap:<22} >= {r.level:.2f}   <- {srcs[:60]}")

    L.append("")
    L.append("CANDIDATES")
    L.append(f"  {'model':<20}{'quality':>9}{'est cost':>12}{'value':>9}  status")
    for c in sorted(d.selection.all_candidates, key=lambda c: c.model.rung):
        if c is d.selection.primary:
            status = "SELECTED (primary)"
        elif d.selection.probe is not None and c.model.id == d.selection.probe.model.id:
            status = "PROBE (cascade)"
        elif c.eligible and c.expected_quality < float(d.policy["quality_floor"]):
            # Clears every capability bar but not the policy's quality floor.
            # Calling this "not cheapest" implies a cost decision when it was a
            # quality decision — and hides the fact that a cheaper policy would
            # have selected it.
            status = (f"below policy floor {d.policy['quality_floor']:.2f} "
                      f"(q={c.expected_quality:.2f})")
        elif c.eligible:
            status = "eligible, not cheapest"
        else:
            status = c.reasons[0] if c.reasons else "rejected"
        learned = ""
        if c.learned_quality is not None:
            learned = f" [seed {c.seed_quality:.2f} + learned {c.learned_quality:.2f} @w{c.learned_weight:.2f}]"
        L.append(
            f"  {c.model.id:<20}{c.expected_quality:>9.3f}{_fmt_usd(c.est_cost_usd):>12}"
            f"{c.value_score:>9.0f}  {status}{learned}"
        )

    L.append("")
    L.append(f"PLAN    {d.selection.first_plan.summary()}")
    for r in d.selection.first_plan.rationale:
        L.append(f"        - {r}")

    if d.selection.cascade_ev:
        ev = d.selection.cascade_ev
        verdict = "ON" if d.selection.cascade else "OFF"
        L.append("")
        L.append(
            f"CASCADE {verdict}: probe {_fmt_usd(ev['probe_cost'])} + verify "
            f"{_fmt_usd(ev['verify_cost'])} + {ev['p_fail']:.0%} x "
            f"{_fmt_usd(ev['direct_cost'])} = {_fmt_usd(ev['expected_cascade_cost'])} "
            f"vs {_fmt_usd(ev['direct_cost'])} direct"
        )

    if d.attempts:
        L.append("")
        L.append("EXECUTION")
        for a in d.attempts:
            v = a.verdict
            L.append(
                f"  [{a.role:<10}] {a.model_id:<20} {_fmt_usd(a.cost.total_usd):>12}  "
                f"{a.result.usage.get('output_tokens', 0):>6} out tok  "
                f"{a.result.latency_s:>6.2f}s  "
                f"{v.verifier} {v.score:.2f} {'ACCEPT' if v.accepted else 'REJECT'}"
            )
            for reason in v.reasons[:2]:
                L.append(f"               · {reason}")

    L.append("")
    L.append(
        f"COST    routed {_fmt_usd(d.total_usd)} "
        f"(router overhead {_fmt_usd(d.overhead_usd)})  "
        f"| baseline {_fmt_usd(d.baseline_usd)}  | saved {_fmt_usd(d.saved_usd)} "
        f"({d.saved_pct:+.1f}%)"
    )
    if d.drift:
        L.append("")
        L.append("GRAPH UPDATED")
        for c in d.drift[:6]:
            L.append(f"  {c}")

    if show_answer and d.answer:
        L.append("")
        L.append("ANSWER")
        for line in d.answer.splitlines()[:20]:
            L.append(f"  {line}")

    if show_trace:
        L.append("")
        L.append("TRACE")
        for t in d.trace:
            L.append(f"  {t}")
    L.append(BAR)
    return "\n".join(L)


def render_savings(report: dict) -> str:
    L = [BAR, "SAVINGS REPORT", BAR]
    L.append(
        f"routes={report['routes']}  cascades={report['cascades']}  "
        f"escalations={report['escalations']} ({report['escalation_rate']:.1%})"
    )
    L.append(
        f"routed {_fmt_usd(report['actual_usd'])} vs baseline "
        f"{_fmt_usd(report['baseline_usd'])}  ->  saved "
        f"{_fmt_usd(report['saved_usd'])} ({report['saved_pct']:.1f}%)"
    )
    L.append("")
    L.append("BY MODEL")
    for r in report["by_model"]:
        L.append(f"  {r['model']:<22}{r['routes']:>5} routes  {_fmt_usd(r['usd']):>12}")
    L.append("")
    L.append("BY TASK TYPE (largest savings first)")
    L.append(f"  {'task':<24}{'n':>4}{'routed':>13}{'baseline':>13}{'saved':>13}")
    for r in report["by_task"]:
        L.append(
            f"  {r['task']:<24}{r['routes']:>4}{_fmt_usd(r['actual_usd']):>13}"
            f"{_fmt_usd(r['baseline_usd']):>13}{_fmt_usd(r['saved_usd']):>13}"
        )
    L.append(BAR)
    return "\n".join(L)
