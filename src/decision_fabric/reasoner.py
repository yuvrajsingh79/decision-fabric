"""Stages 2-4 — Ground, Constrain, Score.

This is the adaptive reasoning core. Nothing here is an if/else over model
names: requirements are *derived* by propagating REQUIRES and ELEVATES edges,
candidates are whatever PROVIDES enough, and the numbers on those edges move as
evidence accumulates. Swap the graph, get a different router.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .config_planner import RequestPlan, plan_request
from .graph import KnowledgeGraph
from .ontology import Candidate, EdgeType, ModelSpec, NodeType, Requirement
from .pricing import project_cost

# How far below the policy floor a probe may sit and still be worth trying
# under verification.
CASCADE_TOLERANCE = 0.12


@dataclass
class Selection:
    primary: Candidate
    primary_plan: RequestPlan
    probe: Candidate | None = None
    probe_plan: RequestPlan | None = None
    cascade: bool = False
    cascade_ev: dict[str, float] = field(default_factory=dict)
    all_candidates: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    requirements: dict[str, Requirement] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    @property
    def first_model(self) -> str:
        return self.probe.model.id if self.cascade and self.probe else self.primary.model.id

    @property
    def first_plan(self) -> RequestPlan:
        return self.probe_plan if self.cascade and self.probe_plan else self.primary_plan


# ---------------------------------------------------------------- requirements


def derive_requirements(
    kg: KnowledgeGraph, task_type: str, domain: str, signals: list[str]
) -> tuple[dict[str, Requirement], list[str]]:
    """Walk REQUIRES from the task, then apply every ELEVATES that fires.

    The trace it returns is the audit answer to "why did this query need a
    better model than the same query yesterday".
    """
    task_node = f"task:{task_type}"
    if not kg.has(task_node):
        raise KeyError(f"unknown task type {task_type!r}")

    reqs: dict[str, Requirement] = {}
    trace: list[str] = []

    for cap_node, data in kg.out_edges(task_node, EdgeType.REQUIRES):
        cap = cap_node.removeprefix("cap:")
        reqs[cap] = Requirement(cap, float(data["min_level"]), [f"{task_node} REQUIRES {cap_node}"])
        trace.append(f"{task_node} -REQUIRES {data['min_level']:.2f}-> {cap_node}")

    elevators = [f"domain:{domain}"] + [f"signal:{s}" for s in signals]
    for node in elevators:
        if not kg.has(node):
            continue
        for cap_node, data in kg.out_edges(node, EdgeType.ELEVATES):
            cap = cap_node.removeprefix("cap:")
            delta = float(data["delta"])
            if cap not in reqs:
                # An elevator can introduce a requirement the base task didn't
                # have — e.g. `citations_required` pulling in long_context.
                base = 0.55
                reqs[cap] = Requirement(cap, base, [f"introduced by {node}"])
                trace.append(f"{node} -ELEVATES-> {cap_node} (new requirement, base {base:.2f})")
            before = reqs[cap].level
            reqs[cap].elevate(delta, node)
            trace.append(
                f"{node} -ELEVATES +{delta:.2f}-> {cap_node}: {before:.2f} -> {reqs[cap].level:.2f}"
            )

    return reqs, trace


# ------------------------------------------------------------------- scoring


def _quality(
    kg: KnowledgeGraph,
    spec: ModelSpec,
    reqs: dict[str, Requirement],
    task_type: str,
    learning: Any,
    prior_strength: int,
) -> tuple[float, float, float | None, float, dict[str, float], str, dict[str, str]]:
    """Returns (expected, seed, learned_mean, evidence_weight, margins, binding_cap, inferred).

    Both `seed` and `margins` read the immutable seed_level, deliberately.

    Learning must not move the *eligibility* gate. Capability levels are a
    structural claim about a model; a run of verifier rejections is noisy
    evidence about a workload. Letting drift revoke eligibility produced a
    measured cost regression: cheap models that failed a few deliberately
    cheap probes were struck from the candidate set entirely, so every
    subsequent query went to the flagship and spend went UP.

    Evidence belongs in `expected_quality` instead, where it raises the
    modelled failure probability and makes the cascade EV gate stop probing
    that model on its own — the same adaptation, reached by lowering spend
    rather than raising it.
    """
    margins: dict[str, float] = {}
    inferred: dict[str, str] = {}
    num = den = 0.0
    for cap, req in reqs.items():
        resolved = kg.resolve_level(spec.id, cap, seed=True)
        provided = resolved.level
        if not resolved.declared:
            inferred[cap] = resolved.explain()
        margins[cap] = round(provided - req.level, 4)
        # Weight by how demanding the requirement is: clearing a 0.95 bar says
        # more about a model than clearing a 0.40 one.
        w = req.level ** 2
        num += provided * w
        den += w
    seed = num / den if den else 0.0
    binding = min(margins, key=lambda c: margins[c]) if margins else ""

    learned_mean: float | None = None
    weight = 0.0
    expected = seed
    if learning is not None:
        post = learning.posterior(spec.id, task_type, seed)
        if post:
            learned_mean, n = post
            weight = n / (n + prior_strength)
            expected = learned_mean
    return (round(expected, 4), round(seed, 4), learned_mean, round(weight, 3),
            margins, binding, inferred)


def build_candidates(
    kg: KnowledgeGraph,
    fabric_cfg: dict[str, Any],
    *,
    reqs: dict[str, Requirement],
    features: Any,
    policy: dict[str, Any],
    slo: dict[str, Any],
    output_class_meta: dict[str, Any],
    learning: Any = None,
) -> list[Candidate]:
    modifiers = fabric_cfg["modifiers"]
    prior_strength = int(fabric_cfg["learning"]["prior_strength"])
    expected_out = int(output_class_meta["expected_output_tokens"])

    out: list[Candidate] = []
    for spec in kg.models():
        reasons: list[str] = []
        hard_ok = True

        # --- hard constraint: context window ---
        needed_ctx = features.input_tokens + expected_out
        if needed_ctx > spec.context_window:
            hard_ok = False
            reasons.append(
                f"context {needed_ctx:,} tok exceeds {spec.context_window:,} window"
            )

        # --- hard constraint: data residency / retention ---
        # Claude Fable 5 is not available under zero data retention; a request
        # from a ZDR org returns 400. That is a compliance boundary, so it is
        # enforced here rather than discovered at call time.
        if getattr(features, "zero_data_retention", False) and spec.config_surface.get(
            "requires_data_retention_30d"
        ):
            hard_ok = False
            reasons.append("unavailable under zero data retention")

        # --- hard constraint: latency SLO ---
        rel = float(spec.nfr.get("relative_latency", 1.0))
        if rel > float(slo.get("max_relative_latency", 1.0)):
            hard_ok = False
            reasons.append(
                f"relative latency {rel:.2f} exceeds SLO ceiling "
                f"{slo['max_relative_latency']:.2f}"
            )

        expected, seed, lmean, lweight, margins, binding, inferred = _quality(
            kg, spec, reqs, features.task_type, learning, prior_strength
        )
        for cap, how in inferred.items():
            reasons.append(f"{cap}: {how}")

        # --- capability bar (soft: may be accepted under protest) ---
        eligible = hard_ok
        short = {c: m for c, m in margins.items() if m < 0}
        if short:
            eligible = False
            worst = min(short, key=lambda c: short[c])
            reasons.append(
                f"below bar on {worst} by {abs(short[worst]):.2f} "
                f"(needs {reqs[worst].level:.2f}, "
                f"{kg.resolve_level(spec.id, worst, seed=True).explain()})"
            )

        plan = plan_request(
            spec,
            required={c: r.level for c, r in reqs.items()},
            output_class_meta=output_class_meta,
            policy=policy,
            slo=slo,
            stable_context_tokens=features.stable_context_tokens,
            effort_multipliers=modifiers["effort_output_multiplier"],
        )
        fresh = max(0, features.input_tokens - (features.stable_context_tokens if plan.use_cache else 0))
        cost = project_cost(
            spec,
            modifiers,
            fresh_input_tokens=fresh,
            output_tokens=expected_out,
            cache_write_tokens=features.stable_context_tokens if plan.use_cache else 0,
            effort=plan.cost_effort_key(),
            batch=plan.batch,
            fast_mode=plan.fast_mode,
        )
        if lmean is not None:
            reasons.append(f"seed {seed:.2f} -> posterior {lmean:.2f} (evidence weight {lweight:.2f})")

        out.append(
            Candidate(
                model=spec,
                expected_quality=expected,
                seed_quality=seed,
                learned_quality=lmean,
                learned_weight=lweight,
                margins=margins,
                binding_capability=binding,
                est_cost_usd=cost.total_usd,
                value_score=round(expected / max(cost.total_usd, 1e-9), 2),
                hard_ok=hard_ok,
                eligible=eligible,
                reasons=reasons,
            )
        )
    return out


# ------------------------------------------------------------------ selection


def select(
    kg: KnowledgeGraph,
    fabric_cfg: dict[str, Any],
    *,
    candidates: list[Candidate],
    reqs: dict[str, Requirement],
    features: Any,
    policy: dict[str, Any],
    slo: dict[str, Any],
    output_class_meta: dict[str, Any],
    trace: list[str],
) -> Selection:
    floor = float(policy["quality_floor"])
    modifiers = fabric_cfg["modifiers"]

    passing = [c for c in candidates if c.eligible and c.expected_quality >= floor]
    rejected = [c for c in candidates if c not in passing]

    if not passing:
        # Nothing clears the floor. Fall back to the strongest model that still
        # satisfies the HARD constraints -- never to one the context window or
        # latency SLO already ruled out. A quality shortfall degrades the
        # answer; violating the SLO breaks the caller's contract, and silently
        # trading the second for the first is not the router's call to make.
        pool = [c for c in candidates if c.eligible]
        note = "no candidate clears policy floor"
        if not pool:
            pool = [c for c in candidates if c.hard_ok]
            note = "no candidate clears the capability bar"
        if not pool:
            pool = candidates
            primary = max(pool, key=lambda c: c.expected_quality)
            trace.append(
                f"INFEASIBLE: no model satisfies this request's hard constraints "
                f"(context window / latency SLO). Falling back to "
                f"{primary.model.id} — the caller's constraints cannot all be met "
                f"and the SLO will be missed."
            )
            primary.reasons.append("selected despite failing a hard constraint")
        else:
            primary = max(pool, key=lambda c: c.expected_quality)
            trace.append(
                f"{note} {floor:.2f}; taking the strongest constraint-satisfying "
                f"model ({primary.model.id}, q={primary.expected_quality:.2f})"
            )
    else:
        # Cheapest model that clears the bar. Ties -> faster, then better.
        primary = min(
            passing,
            key=lambda c: (
                round(c.est_cost_usd, 8),
                float(c.model.nfr.get("relative_latency", 1.0)),
                -c.expected_quality,
            ),
        )
        trace.append(
            f"{len(passing)} candidate(s) clear floor {floor:.2f}; cheapest is "
            f"{primary.model.id} at ${primary.est_cost_usd:.5f} "
            f"(binding capability: {primary.binding_capability}, "
            f"margin {primary.margins.get(primary.binding_capability, 0):+.2f})"
        )

    primary_plan = plan_request(
        primary.model,
        required={c: r.level for c, r in reqs.items()},
        output_class_meta=output_class_meta,
        policy=policy,
        slo=slo,
        stable_context_tokens=features.stable_context_tokens,
        effort_multipliers=modifiers["effort_output_multiplier"],
    )

    sel = Selection(
        primary=primary, primary_plan=primary_plan, all_candidates=list(candidates),
        rejected=rejected, requirements=reqs, trace=trace,
    )

    # --- cascade: is a cheaper probe worth trying under verification? ---
    if policy.get("cascade") and int(policy.get("max_escalations", 0)) > 0:
        probes = [
            c for c in candidates
            if c.eligible
            and c.est_cost_usd < primary.est_cost_usd
            and c.expected_quality >= floor - CASCADE_TOLERANCE
        ]
        if probes:
            probe = min(probes, key=lambda c: c.est_cost_usd)
            verify_cost = _verifier_cost(kg, fabric_cfg, output_class_meta)
            p_fail = probability_of_failure(
                probe.expected_quality,
                float(policy.get("accept_threshold", policy["quality_floor"])),
                float(fabric_cfg["learning"].get("quality_sigma", 0.10)),
            )
            ev = (
                probe.est_cost_usd
                + verify_cost
                + p_fail * primary.est_cost_usd
            )
            worth_it = ev < primary.est_cost_usd
            sel.cascade_ev = {
                "probe_cost": round(probe.est_cost_usd, 6),
                "verify_cost": round(verify_cost, 6),
                "p_fail": p_fail,
                "expected_cascade_cost": round(ev, 6),
                "direct_cost": round(primary.est_cost_usd, 6),
                "expected_saving": round(primary.est_cost_usd - ev, 6),
            }
            if worth_it:
                sel.cascade, sel.probe, sel.probe_plan = True, probe, plan_request(
                    probe.model,
                    required={c: r.level for c, r in reqs.items()},
                    output_class_meta=output_class_meta,
                    policy=policy,
                    slo=slo,
                    stable_context_tokens=features.stable_context_tokens,
                    effort_multipliers=modifiers["effort_output_multiplier"],
                )
                trace.append(
                    f"cascade ON: probe {probe.model.id} ${probe.est_cost_usd:.5f} + verify "
                    f"${verify_cost:.5f} + {p_fail:.0%} x ${primary.est_cost_usd:.5f} = "
                    f"${ev:.5f} < ${primary.est_cost_usd:.5f} direct"
                )
            else:
                trace.append(
                    f"cascade OFF: expected cascade cost ${ev:.5f} >= direct "
                    f"${primary.est_cost_usd:.5f} (p_fail {p_fail:.0%})"
                )
        else:
            trace.append("cascade OFF: no cheaper candidate within tolerance of the floor")
    else:
        trace.append(f"cascade OFF: policy '{policy['name']}' forbids it")

    return sel


def probability_of_failure(expected_quality: float, accept_threshold: float, sigma: float) -> float:
    """P(realised quality < acceptance bar), assuming quality ~ N(expected, sigma).

    The crude `1 - expected_quality` this replaced is not a probability of
    anything: it says a model scored 0.80 fails 20% of the time regardless of
    how low the acceptance bar is, which makes the cascade gate refuse probes
    that are obviously worth running.
    """
    if sigma <= 0:
        return 0.0 if expected_quality >= accept_threshold else 1.0
    z = (accept_threshold - expected_quality) / sigma
    return round(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 4)


def _verifier_cost(kg: KnowledgeGraph, fabric_cfg: dict[str, Any], output_class_meta: dict) -> float:
    """What one verification pass costs — it must be counted, not waved away."""
    vid = fabric_cfg["meta_models"]["verifier"]
    if not kg.has(vid):
        return 0.0
    spec = kg.model(vid)
    judged_tokens = int(output_class_meta["expected_output_tokens"]) + 300
    return project_cost(
        spec, fabric_cfg["modifiers"],
        fresh_input_tokens=judged_tokens, output_tokens=120, effort="none",
    ).total_usd
