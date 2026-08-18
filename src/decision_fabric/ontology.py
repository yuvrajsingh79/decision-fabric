"""Node and edge types for the decision graph.

The graph is deliberately small and typed. Everything the router decides is
derived by walking these edges, so every decision has a citable path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    CAPABILITY = "Capability"
    TASK_TYPE = "TaskType"
    DOMAIN = "Domain"
    SIGNAL = "Signal"
    MODEL = "Model"
    POLICY = "Policy"
    SLO = "Slo"
    OUTPUT_CLASS = "OutputClass"
    EVIDENCE = "Evidence"


class EdgeType(str, Enum):
    REQUIRES = "REQUIRES"        # TaskType -> Capability {min_level}
    PROVIDES = "PROVIDES"        # Model -> Capability {level, seed_level}
    ELEVATES = "ELEVATES"        # Domain|Signal -> Capability {delta}
    PRODUCES = "PRODUCES"        # TaskType -> OutputClass
    ESCALATES_TO = "ESCALATES_TO"  # Model -> Model
    GOVERNS = "GOVERNS"          # Policy -> TaskType|Domain
    OBSERVED = "OBSERVED"        # Evidence -> Model {task_type, success}


@dataclass
class Requirement:
    """A capability bar a candidate model must clear, plus where it came from."""
    capability: str
    level: float
    sources: list[str] = field(default_factory=list)

    def elevate(self, delta: float, source: str) -> None:
        """Raise the bar toward 1.0 with diminishing returns.

        Additive stacking is wrong here: three signals of +0.10 on a task that
        already requires 0.84 would demand 1.14, which no model can meet, so
        every multi-signal query would pin to the top of the ladder — the exact
        spend the fabric exists to prevent. Elevating by `delta` of the
        *remaining headroom* keeps signals meaningful without saturating.
        """
        if delta > 0:
            self.level = round(self.level + delta * (1.0 - self.level), 4)
        elif delta < 0:
            self.level = round(max(0.0, self.level + delta * self.level), 4)
        self.level = min(self.level, 0.99)
        if source not in self.sources:
            self.sources.append(source)


@dataclass
class ModelSpec:
    """A Model node's payload."""
    id: str
    display_name: str
    rung: int
    context_window: int
    max_output_tokens: int
    pricing: dict[str, Any]
    nfr: dict[str, Any]
    config_surface: dict[str, Any]
    provides: dict[str, float]

    def supports(self, knob: str) -> bool:
        return bool(self.config_surface.get(f"supports_{knob}", False))


@dataclass
class Candidate:
    """A model that survived hard constraints, with its scored justification."""
    model: ModelSpec
    expected_quality: float
    seed_quality: float
    learned_quality: float | None
    learned_weight: float
    margins: dict[str, float]          # capability -> provided - required
    binding_capability: str            # the capability with the thinnest margin
    est_cost_usd: float
    value_score: float
    # `hard_ok`: passes the inviolable constraints (context window, latency SLO).
    # `eligible`: hard_ok AND clears every capability bar.
    # They are separate because a quality shortfall may be accepted under
    # protest, while blowing the context window or the SLO may not.
    hard_ok: bool
    eligible: bool
    reasons: list[str] = field(default_factory=list)
