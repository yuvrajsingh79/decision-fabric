"""Stage 8 — Learn.

Two feedback channels, both writing back into the graph:

1. Beta posteriors per (model, task_type). These blend with the seed capability
   prior at scoring time, weighted by observation count.
2. Capability drift. Sustained failures on a task pull down the model's PROVIDES
   levels for that task's required capabilities (bounded), so the *structure*
   adapts, not just a scalar. A model that keeps getting escalated on debugging
   stops being offered for debugging-shaped work — including for task types that
   share the same capabilities.

Nothing here is magic; it is a Beta-Bernoulli update with a drift cap. The point
is that the adaptation lives on graph edges, so it is inspectable and revertible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph import KnowledgeGraph
from .ontology import EdgeType, NodeType
from .telemetry import Telemetry


@dataclass
class Posterior:
    mean: float
    n: int
    successes: int


class LearningStore:
    def __init__(self, kg: KnowledgeGraph, telemetry: Telemetry, cfg: dict[str, Any]) -> None:
        self.kg = kg
        self.tel = telemetry
        self.prior_strength = int(cfg.get("prior_strength", 8))
        self.max_drift = float(cfg.get("max_level_drift", 0.15))
        self.escalation_is_failure = bool(cfg.get("escalation_counts_as_failure", True))

    # ---------- read side (used by the reasoner) ----------

    def posterior(
        self, model_id: str, task_type: str, prior_mean: float
    ) -> tuple[float, int] | None:
        """Beta posterior mean anchored on the seed prior.

        The prior is Beta(prior_mean*k, (1-prior_mean)*k) with k = prior_strength,
        so with zero observations the posterior *is* the seed level and the first
        observation nudges rather than overwrites. Anchoring matters: a flat
        Beta(1,1) would drop a 0.94-capability model to 0.67 after one success,
        which is worse than not learning at all.
        """
        succ, n = self.tel.counts(model_id, task_type)
        if n == 0:
            return None
        k = self.prior_strength
        mean = (prior_mean * k + succ) / (k + n)
        return (round(max(0.0, min(1.0, mean)), 4), n)

    def posteriors(self) -> list[tuple[str, str, Posterior]]:
        """Raw observed rates per pair — for inspection, not for scoring."""
        out = []
        for model_id, task, succ, n in self.tel.all_pairs():
            out.append((model_id, task, Posterior(mean=round(succ / n, 4), n=n, successes=succ)))
        return sorted(out, key=lambda t: (t[0], t[1]))

    # ---------- write side ----------

    def observe(
        self, model_id: str, task_type: str, success: bool, source: str, route_id: int | None = None
    ) -> None:
        self.tel.record_observation(model_id, task_type, success, source, route_id)

    def apply_drift(self, task_type: str) -> list[str]:
        """Recompute levels after new evidence landed on `task_type`.

        Capability levels are recomputed globally rather than per task, because
        capabilities are shared: `deep_reasoning` is required by debugging,
        code_review, data_analysis and more. Updating it from one task's
        posterior in isolation means whichever task ran most recently wins, and
        the level oscillates instead of converging. Evidence from every task
        that depends on a capability is pooled instead.
        """
        return self.recompute_levels()

    def recompute_levels(self) -> list[str]:
        # capability -> [(task_type, required_level)] across the whole ontology
        cap_tasks: dict[str, list[tuple[str, float]]] = {}
        for task_node in self.kg.nodes_of(NodeType.TASK_TYPE):
            task = task_node.removeprefix("task:")
            for cap, level in self.kg.requirements_of(task).items():
                cap_tasks.setdefault(cap, []).append((task, level))

        changes: list[str] = []
        for spec in self.kg.models():
            for cap, tasks in cap_tasks.items():
                edge = self.kg.edge(spec.id, f"cap:{cap}", EdgeType.PROVIDES)
                if edge is None:
                    continue
                seed = float(edge["seed_level"])

                num = den = 0.0
                total_obs = 0
                for task, req_level in tasks:
                    reqs = self.kg.requirements_of(task)
                    seed_task = self.seed_quality(spec.id, reqs)
                    post = self.posterior(spec.id, task, seed_task)
                    if not post:
                        continue
                    mean, n = post
                    # Weight by evidence volume AND by how much this task leans
                    # on the capability -- a task that barely needs it should
                    # not dominate the estimate.
                    w = n * (req_level ** 2)
                    num += (mean - seed_task) * w
                    den += w
                    total_obs += n
                if den == 0:
                    new = seed
                else:
                    delta = num / den
                    new = max(seed - self.max_drift, min(seed + self.max_drift, seed + delta))
                new = round(max(0.0, min(0.99, new)), 4)

                if abs(new - float(edge["level"])) >= 0.001:
                    changes.append(
                        f"{spec.id} PROVIDES {cap}: {edge['level']:.3f} -> {new:.3f} "
                        f"(seed {seed:.3f}, {total_obs} pooled obs)"
                    )
                    self.kg.set_edge_attr(
                        spec.id, f"cap:{cap}", EdgeType.PROVIDES,
                        level=new, observations=total_obs,
                    )
        return changes

    def seed_quality(self, model_id: str, reqs: dict[str, float]) -> float:
        """Seed-prior quality of a model for a requirement set (difficulty-weighted)."""
        num = den = 0.0
        for cap, level in reqs.items():
            edge = self.kg.edge(model_id, f"cap:{cap}", EdgeType.PROVIDES)
            provided = float(edge["seed_level"]) if edge else 0.0
            w = level ** 2
            num += provided * w
            den += w
        return num / den if den else 0.0

    def rehydrate(self) -> list[str]:
        """Replay all stored observations into the graph. Call on startup."""
        return self.recompute_levels()
