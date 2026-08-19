"""Stage 7 — Verify.

A cascade is only worth running if the check that guards it is cheap and
honest. Two verifiers:

* `heuristic` — free. Catches the failure modes that actually dominate cheap-
  model output: truncation, refusal, punting ("I don't have enough context"),
  empty or stub answers, and format non-compliance.
* `llm` — a Haiku judge with a strict JSON schema. Costs ~$0.0002 per check,
  which is still two orders of magnitude below an Opus retry.

Both return a 0..1 score compared against the policy's quality floor.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .executor import ExecutionResult, Executor
from .ontology import ModelSpec
from .pricing import actual_cost, project_cost

PUNT_PATTERNS = [
    r"\bI (don'?t|do not) have (enough|access to|the) ",
    r"\bI'?m (not able|unable) to\b",
    r"\bcannot (determine|assist|help) (with|from)\b",
    r"\bwould need more (context|information|details)\b",
    r"\bas an AI\b",
    r"\bplease (provide|share) (the|more)\b",
]

STUB_PATTERNS = [
    r"\b(TODO|FIXME|implementation goes here|your code here|\.\.\.\s*$)",
    r"^\s*(here'?s a (rough|basic|simple) (outline|sketch))",
]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answers_the_question": {"type": "boolean"},
        "complete": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "score": {"type": "number"},
        "failure_reason": {"type": "string"},
    },
    "required": ["answers_the_question", "complete", "grounded", "score", "failure_reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You grade whether a response is good enough to ship, not whether it is perfect.\n"
    "Score 0..1. Score below 0.6 only when the response is truncated, refuses, punts for "
    "more information it was already given, is a stub, or contradicts the supplied context.\n"
    "Do not penalise brevity if the question was simple. Return JSON only."
)


@dataclass
class Verdict:
    accepted: bool
    score: float
    verifier: str
    cost_usd: float = 0.0
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


def verify(
    result: ExecutionResult,
    *,
    mode: str,
    threshold: float,
    executor: Executor,
    task_type: str,
    query: str,
    judge_spec: ModelSpec | None = None,
    modifiers: dict[str, Any] | None = None,
) -> Verdict:
    if result.error:
        return Verdict(False, 0.0, "error", reasons=[f"execution failed: {result.error}"])
    if mode == "none":
        return Verdict(True, 1.0, "none", reasons=["verification disabled by policy"])

    hv = _heuristic(result, task_type, threshold)
    if mode == "heuristic" or hv.score < 0.35 or judge_spec is None:
        # A hard heuristic failure needs no second opinion — don't pay for one.
        return hv
    return _llm(result, query, task_type, threshold, executor, judge_spec, modifiers or {}, hv)


def _heuristic(result: ExecutionResult, task_type: str, threshold: float) -> Verdict:
    reasons: list[str] = []
    text = result.text or ""

    if result.simulated and result.sim_quality is not None:
        score = result.sim_quality
        reasons.append(f"simulated answer quality {score:.2f}")
        return Verdict(score >= threshold, score, "heuristic(sim)", reasons=reasons)

    score = 0.85
    if not text.strip():
        return Verdict(False, 0.0, "heuristic", reasons=["empty response"])
    if result.stop_reason == "max_tokens":
        score -= 0.45
        reasons.append("truncated at max_tokens")
    if result.stop_reason == "refusal":
        return Verdict(False, 0.0, "heuristic", reasons=["model refused"])
    for pat in PUNT_PATTERNS:
        if re.search(pat, text, re.I):
            score -= 0.35
            reasons.append(f"punt marker: {pat}")
            break
    for pat in STUB_PATTERNS:
        if re.search(pat, text, re.I | re.M):
            score -= 0.30
            reasons.append(f"stub marker: {pat}")
            break
    # Task-shaped floors: a 20-token "answer" to a design question is a failure.
    min_words = {"architecture_design": 120, "code_review": 80, "debugging": 60,
                 "code_gen": 40, "math_proof": 80}.get(task_type, 0)
    if min_words and len(text.split()) < min_words:
        score -= 0.30
        reasons.append(f"{len(text.split())} words is below the {min_words}-word floor for {task_type}")

    score = round(max(0.0, min(1.0, score)), 3)
    if not reasons:
        reasons.append("no failure markers")
    return Verdict(score >= threshold, score, "heuristic", reasons=reasons)


def _llm(
    result: ExecutionResult,
    query: str,
    task_type: str,
    threshold: float,
    executor: Executor,
    judge_spec: ModelSpec,
    modifiers: dict[str, Any],
    heuristic_verdict: Verdict,
) -> Verdict:
    prompt = (
        f"TASK TYPE: {task_type}\n\nUSER REQUEST:\n{query[:4000]}\n\n"
        f"RESPONSE TO GRADE:\n{(result.text or '')[:8000]}"
    )
    if executor.dry_run:
        # Keep the judge in the loop for cost accounting even when simulating.
        est = project_cost(
            judge_spec, modifiers,
            fresh_input_tokens=len(prompt) // 4, output_tokens=120, effort="none",
        ).total_usd
        v = heuristic_verdict
        return Verdict(v.accepted, v.score, "llm(sim)", cost_usd=est,
                       reasons=v.reasons + ["judge simulated"])

    try:
        msg = executor.client.messages.create(
            model=judge_spec.id,
            max_tokens=512,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
    except Exception as e:
        return Verdict(
            heuristic_verdict.accepted, heuristic_verdict.score, "heuristic(judge-failed)",
            reasons=heuristic_verdict.reasons + [f"judge unavailable: {type(e).__name__}"],
        )

    text = next((b.text for b in msg.content if b.type == "text"), "{}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Verdict(heuristic_verdict.accepted, heuristic_verdict.score,
                       "heuristic(judge-unparseable)", reasons=heuristic_verdict.reasons)

    cost = actual_cost(judge_spec, modifiers, msg.usage).total_usd
    executor.record_spend("verifier", judge_spec.id, msg.usage, cost, task_type)
    score = float(data.get("score", 0.0))
    reasons = [f"judge score {score:.2f}"]
    if data.get("failure_reason"):
        reasons.append(str(data["failure_reason"])[:200])
    for flag in ("answers_the_question", "complete", "grounded"):
        if not data.get(flag, True):
            reasons.append(f"judge: not {flag}")
    return Verdict(score >= threshold, round(score, 3), "llm", cost_usd=cost,
                   reasons=reasons, raw=data)
