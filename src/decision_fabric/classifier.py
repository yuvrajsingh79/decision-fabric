"""Optional LLM refinement of the heuristic feature extraction.

This runs *only* when the free heuristic is unsure. That gate matters: a router
that pays for a classification call on every query has already spent part of
what it is trying to save. On Haiku with a cached system prefix the refinement
costs roughly $0.0003 — worth it when it prevents one misrouted Opus call, not
worth it 100% of the time.
"""
from __future__ import annotations

import json
from typing import Any

from .executor import Executor
from .features import QueryFeatures
from .ontology import ModelSpec
from .pricing import actual_cost, project_cost

# Legacy fixed threshold. Retained only so tooling that reports a single
# number still has one; the real decision is made by measured calibration
# (Fabric.may_commit). Held-out measurement showed a fixed threshold cannot be
# sound here: the heuristic classifies natural phrasing at ~14% accuracy, and
# any single cut-off either trusts it far too often or is arbitrary.
REFINE_BELOW_CONFIDENCE = 0.55


def _schema(task_types: list[str], domains: list[str], output_classes: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "enum": task_types},
            "domain": {"type": "string", "enum": domains},
            "output_class": {"type": "string", "enum": output_classes},
            "reasoning_depth": {
                "type": "string",
                "enum": ["trivial", "shallow", "moderate", "deep", "research"],
            },
            "confidence": {"type": "number"},
        },
        "required": ["task_type", "domain", "output_class", "reasoning_depth", "confidence"],
        "additionalProperties": False,
    }


SYSTEM = (
    "You are a routing classifier inside a cost-control layer. Classify the request. "
    "You are NOT answering it — never attempt the task.\n"
    "`reasoning_depth` is how much deliberation the request genuinely needs, not how "
    "long the answer should be. A hard question with a one-word answer is 'deep'.\n"
    "`confidence` is 0..1 for your task_type choice. Return JSON only."
)

# Depth -> extra pressure on deep_reasoning, applied as a synthetic signal.
DEPTH_ELEVATION = {"trivial": -0.10, "shallow": -0.05, "moderate": 0.0, "deep": 0.08, "research": 0.15}


def should_refine(
    features: QueryFeatures, enabled: bool, fabric: Any = None
) -> tuple[bool, str]:
    """Should we pay for LLM classification? Returns (yes, reason).

    Default is YES. The heuristic earns the right to skip the call only where
    its confidence band has demonstrated precision on held-out development
    data. Misrouting costs far more than classifying.
    """
    if features.source != "heuristic":
        return False, f"task type came from {features.source}, not the heuristic"
    if not enabled:
        return False, "LLM classifier disabled by caller"
    if fabric is None:
        allowed = features.task_confidence >= REFINE_BELOW_CONFIDENCE
        return (not allowed), f"no calibration available; fixed threshold {REFINE_BELOW_CONFIDENCE}"
    allowed, reason = fabric.may_commit(features.task_confidence)
    return (not allowed), reason


def refine(
    features: QueryFeatures,
    executor: Executor,
    spec: ModelSpec,
    modifiers: dict[str, Any],
    *,
    task_types: list[str],
    domains: list[str],
    output_classes: list[str],
) -> tuple[QueryFeatures, float]:
    """Returns (possibly updated features, classifier cost in USD)."""
    schema = _schema(task_types, domains, output_classes)
    prompt = f"Request to classify:\n\n{features.text[:6000]}"

    if executor.dry_run:
        cost = project_cost(
            spec, modifiers,
            fresh_input_tokens=len(SYSTEM) // 4 + len(prompt) // 4,
            output_tokens=80, effort="none",
        ).total_usd
        features.notes.append(
            f"classifier refinement simulated (would cost ${cost:.6f}); "
            f"heuristic guess '{features.task_type}' kept"
        )
        return features, cost

    try:
        msg = executor.client.messages.create(
            model=spec.id,
            max_tokens=300,
            # Frozen system + schema = a stable cacheable prefix across every
            # classification. Volatile content (the query) comes after it.
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except Exception as e:
        features.notes.append(f"classifier unavailable ({type(e).__name__}); heuristic kept")
        return features, 0.0

    text = next((b.text for b in msg.content if b.type == "text"), "{}")
    cost = actual_cost(spec, modifiers, msg.usage).total_usd
    executor.record_spend("classifier", spec.id, msg.usage, cost, features.text[:80])
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        features.notes.append("classifier returned unparseable JSON; heuristic kept")
        return features, cost

    prev = features.task_type
    features.task_type = data.get("task_type", features.task_type)
    features.domain = data.get("domain", features.domain)
    features.output_class = data.get("output_class", features.output_class)
    features.task_confidence = float(data.get("confidence", features.task_confidence))
    features.source = "llm"
    depth = data.get("reasoning_depth", "moderate")
    features.notes.append(
        f"classifier: {prev} -> {features.task_type} "
        f"(conf {features.task_confidence:.2f}, depth {depth})"
    )
    features.notes.append(f"depth_elevation={DEPTH_ELEVATION.get(depth, 0.0):+.2f}")
    return features, cost
