"""The pipeline: perceive -> ground -> constrain -> score -> configure ->
execute -> verify -> learn.

`Router.route()` is the only entry point most callers need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import classifier, features as feat, reasoner, verifier
from .config_planner import RequestPlan, plan_request
from .executor import ExecutionResult, Executor
from .learning import LearningStore
from .ontology import Candidate, EdgeType, Requirement
from .pricing import CostBreakdown, actual_cost, project_cost
from .seed import Fabric
from .telemetry import RouteRecord, Telemetry


@dataclass
class Attempt:
    model_id: str
    plan: RequestPlan
    result: ExecutionResult
    verdict: Any
    cost: CostBreakdown
    role: str  # probe | primary | escalation


@dataclass
class RoutingDecision:
    query: str
    features: feat.QueryFeatures
    policy: dict[str, Any]
    slo_name: str
    output_class: str
    requirements: dict[str, Requirement]
    selection: reasoner.Selection
    attempts: list[Attempt] = field(default_factory=list)
    overhead_usd: float = 0.0          # classifier + verifier — the router's own cost
    total_usd: float = 0.0
    baseline_usd: float = 0.0
    executed: bool = False
    dry_run: bool = True
    route_id: int | None = None
    drift: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    # ---- convenience ----
    @property
    def chosen_model(self) -> str:
        return self.attempts[-1].model_id if self.attempts else self.selection.first_model

    @property
    def answer(self) -> str:
        return self.attempts[-1].result.text if self.attempts else ""

    @property
    def escalated(self) -> bool:
        return len(self.attempts) > 1

    @property
    def saved_usd(self) -> float:
        return round(self.baseline_usd - self.total_usd, 6)

    @property
    def saved_pct(self) -> float:
        return round(100 * self.saved_usd / self.baseline_usd, 2) if self.baseline_usd else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query[:200],
            "task_type": self.features.task_type,
            "domain": self.features.domain,
            "signals": self.features.signals,
            "policy": self.policy["name"],
            "slo": self.slo_name,
            "requirements": {c: round(r.level, 3) for c, r in self.requirements.items()},
            "plan": self.selection.first_plan.summary(),
            "cascade": self.selection.cascade,
            "cascade_ev": self.selection.cascade_ev,
            "chosen_model": self.chosen_model,
            "escalated": self.escalated,
            "attempts": [
                {"model": a.model_id, "role": a.role, "plan": a.plan.summary(),
                 "accepted": bool(getattr(a.verdict, "accepted", None)),
                 "score": getattr(a.verdict, "score", None),
                 "usd": round(a.cost.total_usd, 6)}
                for a in self.attempts
            ],
            "overhead_usd": round(self.overhead_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "baseline_usd": round(self.baseline_usd, 6),
            "saved_usd": self.saved_usd,
            "saved_pct": self.saved_pct,
            "dry_run": self.dry_run,
            "trace": self.trace,
        }


class Router:
    def __init__(
        self,
        config_dir: Path | str | None = None,
        *,
        db_path: Path | str = "./decision_fabric.db",
        dry_run: bool | None = None,
        use_llm_classifier: bool = True,
    ) -> None:
        self.fabric = Fabric(config_dir) if config_dir else Fabric()
        self.kg = self.fabric.kg
        self.telemetry = Telemetry(db_path)
        self.learning = LearningStore(self.kg, self.telemetry, self.fabric.learning_cfg)
        self.executor = Executor(dry_run=dry_run)
        self.use_llm_classifier = use_llm_classifier
        # Replay stored evidence so a restarted process keeps what it learned.
        self.startup_drift = self.learning.rehydrate()

    @property
    def dry_run(self) -> bool:
        return self.executor.dry_run

    def close(self) -> None:
        self.telemetry.close()

    # ------------------------------------------------------------------ route

    def route(
        self,
        query: str,
        *,
        execute: bool = True,
        policy: str | None = None,
        latency_slo: str = "interactive",
        domain: str | None = None,
        task_type: str | None = None,
        context_tokens: int = 0,
        stable_context_tokens: int = 0,
        zero_data_retention: bool = False,
        system: Any = None,
        learn: bool = True,
    ) -> RoutingDecision:
        cfg = {
            "modifiers": self.fabric.modifiers,
            "learning": self.fabric.learning_cfg,
            "meta_models": self.fabric.meta_models,
        }
        trace: list[str] = []
        overhead = 0.0

        # --- 1. Perceive -------------------------------------------------
        f = feat.extract(
            query, self.fabric.signals,
            context_tokens=context_tokens,
            stable_context_tokens=stable_context_tokens,
            scope_cfg=self.fabric.scope_signals,
            latency_slo=latency_slo,
            zero_data_retention=zero_data_retention,
            domain=domain,
            task_type=task_type,
        )
        trace.append(
            f"perceive: task={f.task_type} (conf {f.task_confidence:.2f}, {f.source}), "
            f"domain={f.domain}, signals={f.signals or '-'}, ~{f.input_tokens} input tokens"
        )

        depth_elevation = 0.0
        refine, gate_reason = classifier.should_refine(f, self.use_llm_classifier, self.fabric)
        if refine:
            trace.append(
                f"perceive: gate says escalate — {gate_reason}; paying for LLM classification"
            )
            f, ccost = classifier.refine(
                f, self.executor, self.kg.model(self.fabric.meta_models["classifier"]),
                self.fabric.modifiers,
                task_types=sorted(self.fabric.ontology["task_types"]),
                domains=sorted(self.fabric.ontology["domains"]),
                output_classes=sorted(self.fabric.output_classes),
            )
            overhead += ccost
            for n in f.notes:
                trace.append(f"perceive: {n}")
                if n.startswith("depth_elevation="):
                    depth_elevation = float(n.split("=", 1)[1])
        else:
            trace.append(f"perceive: gate says commit — {gate_reason}; no LLM call")

        # --- 2. Ground: policy, SLO, output class ------------------------
        pol = self.fabric.effective_policy(policy, f.domain)
        if pol.get("forced_by"):
            trace.append(
                f"policy: '{pol['overrode']}' overridden to '{pol['name']}' by {pol['forced_by']}"
            )
        else:
            trace.append(f"policy: {pol['name']} (quality floor {pol['quality_floor']:.2f})")

        slo = {"name": f.latency_slo, **self.fabric.latency_slos[f.latency_slo]}
        out_class = f.output_class or next(
            (d.removeprefix("out:") for d, _ in self.kg.out_edges(f"task:{f.task_type}", EdgeType.PRODUCES)),
            "short",
        )
        # A caller-supplied or classifier-supplied output_class is authoritative;
        # the scope shift only adjusts the task's structural default.
        if f.output_class is None and f.output_class_shift:
            ladder = self.fabric.output_class_ladder
            if out_class in ladder:
                i = ladder.index(out_class)
                shifted = ladder[max(0, min(len(ladder) - 1, i + f.output_class_shift))]
                if shifted != out_class:
                    trace.append(
                        f"ground: scope {f.scope_signals} shifts output class "
                        f"{out_class} -> {shifted} ({f.output_class_shift:+d} rung)"
                    )
                    out_class = shifted

        oc_meta = self.fabric.output_classes[out_class]
        trace.append(
            f"ground: output class '{out_class}' "
            f"(~{oc_meta['expected_output_tokens']} tok, max_tokens {oc_meta['max_tokens']})"
        )

        # --- 3. Derive requirements by edge propagation ------------------
        reqs, req_trace = reasoner.derive_requirements(self.kg, f.task_type, f.domain, f.signals)
        trace.extend(f"graph: {t}" for t in req_trace)
        if depth_elevation:
            r = reqs.get("deep_reasoning")
            if r:
                before = r.level
                r.elevate(depth_elevation, "classifier:reasoning_depth")
                trace.append(
                    f"graph: classifier depth -> deep_reasoning {before:.2f} -> {r.level:.2f}"
                )

        # --- 4/5. Constrain, score, configure ----------------------------
        cands = reasoner.build_candidates(
            self.kg, cfg, reqs=reqs, features=f, policy=pol, slo=slo,
            output_class_meta=oc_meta, learning=self.learning,
        )
        for c in cands:
            trace.append(
                f"score: {c.model.id} q={c.expected_quality:.3f} "
                f"${c.est_cost_usd:.5f} value={c.value_score:.0f} "
                f"{'OK' if c.eligible else 'REJECTED'}"
                + (f" [{'; '.join(c.reasons)}]" if c.reasons else "")
            )
        sel = reasoner.select(
            self.kg, cfg, candidates=cands, reqs=reqs, features=f, policy=pol,
            slo=slo, output_class_meta=oc_meta, trace=trace,
        )

        decision = RoutingDecision(
            query=query, features=f, policy=pol, slo_name=slo["name"],
            output_class=out_class, requirements=reqs, selection=sel,
            overhead_usd=overhead, dry_run=self.dry_run, trace=trace,
        )

        if not execute:
            decision.total_usd = round(overhead + _est(sel), 8)
            decision.baseline_usd = self._baseline_cost(f, oc_meta)
            return decision

        # --- 6/7. Execute, verify, escalate ------------------------------
        self._run_ladder(decision, cfg, oc_meta)

        # --- accounting ---
        decision.executed = True
        decision.total_usd = round(
            decision.overhead_usd + sum(a.cost.total_usd for a in decision.attempts), 8
        )
        decision.baseline_usd = self._baseline_cost(f, oc_meta, decision=decision)
        trace.append(
            f"cost: routed ${decision.total_usd:.6f} vs baseline ${decision.baseline_usd:.6f} "
            f"({decision.saved_pct:+.1f}%)"
        )

        # --- 8. Learn -----------------------------------------------------
        decision.route_id = self._record(decision)
        if learn:
            self._learn(decision)
        return decision

    # ---------------------------------------------------------------- helpers

    def _run_ladder(self, d: RoutingDecision, cfg: dict, oc_meta: dict) -> None:
        sel = d.selection
        pol = d.policy
        # Acceptance is an absolute shippability test; selection headroom is a
        # separate, higher bar handled during scoring.
        accept = float(pol.get("accept_threshold", pol["quality_floor"]))
        max_esc = int(pol.get("max_escalations", 0))
        judge_spec = (
            self.kg.model(self.fabric.meta_models["verifier"])
            if pol.get("verifier") == "llm" else None
        )
        messages = [{"role": "user", "content": d.query}]

        queue: list[tuple[Candidate, RequestPlan, str]] = []
        if sel.cascade and sel.probe and sel.probe_plan:
            queue.append((sel.probe, sel.probe_plan, "probe"))
        queue.append((sel.primary, sel.primary_plan, "primary"))

        # Escalation rungs are drawn from candidates that already cleared the
        # HARD constraints (context window, latency SLO). Walking ESCALATES_TO
        # blindly would escalate into a model this request may not use — e.g. a
        # slow model under a realtime SLO, which is a correctness bug, not a
        # cost one.
        slo = {"name": d.slo_name, **self.fabric.latency_slos[d.slo_name]}
        for c in sorted(sel.all_candidates, key=lambda c: c.model.rung):
            if c.eligible and c.model.rung > sel.primary.model.rung:
                queue.append((
                    c,
                    plan_request(
                        c.model,
                        required={cap: r.level for cap, r in d.requirements.items()},
                        output_class_meta=oc_meta,
                        policy=pol,
                        slo=slo,
                        stable_context_tokens=d.features.stable_context_tokens,
                        effort_multipliers=self.fabric.modifiers["effort_output_multiplier"],
                    ),
                    "escalation",
                ))
        if len(queue) == (2 if sel.cascade else 1):
            d.trace.append(
                f"no eligible rung above {sel.primary.model.id} — it is the ceiling for this "
                f"request's constraints"
            )

        escalations = 0
        for cand, plan, role in queue:
            spec = self.kg.model(plan.model_id)
            result = self.executor.execute(
                plan, spec, messages, None,
                model_quality=cand.expected_quality if cand else 0.8,
            )
            cost = actual_cost(
                spec, self.fabric.modifiers, result.usage or {},
                batch=plan.batch, fast_mode=plan.fast_mode,
            )
            v = verifier.verify(
                result, mode=str(pol.get("verifier", "heuristic")), threshold=accept,
                executor=self.executor, task_type=d.features.task_type, query=d.query,
                judge_spec=judge_spec, modifiers=self.fabric.modifiers,
            )
            d.overhead_usd += v.cost_usd
            d.attempts.append(Attempt(plan.model_id, plan, result, v, cost, role))
            d.trace.append(
                f"execute[{role}]: {plan.model_id} -> {result.usage.get('output_tokens', 0)} out tok, "
                f"${cost.total_usd:.6f}, verdict {v.verifier} {v.score:.2f} "
                f"{'ACCEPT' if v.accepted else 'REJECT'} ({'; '.join(v.reasons)})"
            )
            if v.accepted:
                return
            if role != "probe":
                escalations += 1
            if escalations > max_esc:
                d.trace.append(
                    f"escalation budget ({max_esc}) exhausted — returning best-effort answer "
                    f"from {plan.model_id}"
                )
                return
            d.trace.append(f"verification failed on {plan.model_id} -> escalating")

    def _baseline_cost(
        self, f: feat.QueryFeatures, oc_meta: dict, decision: RoutingDecision | None = None
    ) -> float:
        """Projected spend if every query went to the flagship at default effort.

        This is a *projection*, not a measurement — we do not actually run the
        baseline model. Input tokens are the real ones; output tokens are the
        task's expected length at the baseline effort multiplier.
        """
        b = self.fabric.baseline()
        spec = self.kg.model(b["model"])
        in_tok = f.input_tokens
        if decision and decision.attempts:
            u = decision.attempts[0].result.usage or {}
            in_tok = max(in_tok, u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0))
        return round(
            project_cost(
                spec, self.fabric.modifiers,
                fresh_input_tokens=in_tok,
                output_tokens=int(oc_meta["expected_output_tokens"]),
                effort=b.get("effort", "high"),
            ).total_usd,
            8,
        )

    def _record(self, d: RoutingDecision) -> int:
        first = d.attempts[0] if d.attempts else None
        last = d.attempts[-1] if d.attempts else None
        usage = (last.result.usage if last else {}) or {}
        return self.telemetry.record_route(RouteRecord(
            query=d.query,
            task_type=d.features.task_type,
            domain=d.features.domain,
            signals=d.features.signals,
            policy=d.policy["name"],
            slo=d.slo_name,
            first_model=first.model_id if first else d.selection.first_model,
            final_model=d.chosen_model,
            plan=(last.plan.summary() if last else d.selection.first_plan.summary()),
            cascade=d.selection.cascade,
            escalated=d.escalated,
            accepted=bool(last and getattr(last.verdict, "accepted", False)),
            est_cost_usd=round(d.selection.primary.est_cost_usd, 8),
            actual_cost_usd=d.total_usd,
            baseline_cost_usd=d.baseline_usd,
            input_tokens=int(usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            mode="dry-run" if d.dry_run else "live",
            trace=d.trace,
        ))

    def _learn(self, d: RoutingDecision) -> None:
        n = len(d.attempts)
        for i, a in enumerate(d.attempts):
            accepted = bool(getattr(a.verdict, "accepted", False))
            # A model that got escalated past is a failure observation for that
            # model on this task type — that is the signal that reshapes routing.
            success = accepted and i == n - 1
            self.learning.observe(
                a.model_id, d.features.task_type, success,
                source="verifier" if accepted else "escalation", route_id=d.route_id,
            )
        d.drift = self.learning.apply_drift(d.features.task_type)
        for c in d.drift:
            d.trace.append(f"learn: {c}")


def _est(sel: reasoner.Selection) -> float:
    c = sel.probe if (sel.cascade and sel.probe) else sel.primary
    return c.est_cost_usd
