"""Stage 1 — Perceive.

Turn a raw request into the small feature vector the graph can bind to:
task type, domain, complexity signals, size, and non-functional requirements.

The heuristic extractor is free and runs always. The LLM classifier
(classifier.py) refines it only when the heuristic is unsure — that gate is the
difference between a router that saves money and one that doubles your calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Rough chars-per-token for English prose + code. Replaced by the real
# count_tokens call when a live client is available (see executor.estimate_tokens).
CHARS_PER_TOKEN = 3.8

LONG_INPUT_TOKEN_THRESHOLD = 20_000

# Total pattern weight at which the heuristic is considered fully evidenced.
# Below it, confidence is scaled down no matter how uncontested the winner is.
STRONG_EVIDENCE = 4.0

# task_type -> (regex, weight). Deliberately coarse: this is a prior, not a verdict.
TASK_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "chitchat": [(r"^\s*(hi|hey|hello|thanks|thank you|good (morning|evening))\b", 3.0)],
    "classification": [(r"\b(classify|categori[sz]e|label|tag|is this (a|an)|sentiment)\b", 2.0),
                       (r"\b(spam or|positive or negative|which category)\b", 2.5)],
    "extraction": [(r"\b(extract|pull out|parse|fields?|into json|structured output)\b", 2.0),
                   (r"\blist (all |the |every |each )?(documented|stated|named|the )?\w+", 1.5),
                   (r"\b(do not infer|only what is stated|verbatim)\b", 1.5),
                   (r"\b(invoice|receipt|resume|cv|intake|form)\b", 1.0)],
    "summarization_short": [(r"\b(summari[sz]e|tl;?dr|key points|recap|digest)\b", 2.0)],
    "summarization_long": [(r"\b(summari[sz]e).{0,40}\b(all|entire|corpus|these \d+|every)\b", 3.0),
                           (r"\b(across (all|the) (documents|files|tickets|transcripts))\b", 2.5)],
    "rag_qa": [(r"\b(according to|based on the (attached|document|context)|in the (doc|pdf|contract))\b", 2.5),
               (r"\b(cite|quote) (the )?(source|section|clause)\b", 2.0),
               (r"\bagainst the (checklist|policy|standard|requirements|narrative)\b", 2.5),
               (r"\bflag any (gaps|issues|discrepancies|omissions)\b", 2.0)],
    "factual_qa": [(r"^\s*(what|who|when|where|which|how many|is|are|does)\b.{0,90}\?\s*$", 1.5),
                   (r"\b(define|meaning of|difference between)\b", 1.2)],
    "code_gen": [(r"\b(write|implement|create|build|generate|add|make|code)\s+(me\s+)?(a|an|the)?\s*(\w+\s+){0,2}(function|method|class|script|endpoint|component|test|migration|query|parser|handler|project|app|application|program|package|module|library|cli|service|game|bot|scraper|dashboard)\b", 2.5),
                 (r"\b(in (python|typescript|javascript|java|go|rust|sql|c\+\+)|snippet|boilerplate|regex)\b", 1.0)],
    "code_review": [(r"\b(review|critique|audit) (this |the |my )?(code|diff|pr|pull request|patch)\b", 3.0),
                    (r"\bis (this|the) \w+( \w+)? vulnerable\b", 3.0),
                    (r"\b(session fixation|sql injection|xss|csrf|auth bypass|privilege escalation|timing attack)\b", 2.5),
                    (r"\b(code smell|anti-?pattern|refactor this)\b", 1.5)],
    "debugging": [(r"\b(debug|traceback|stack ?trace|exception|segfault|core dump)\b", 3.0),
                  (r"\bwhy (is|does|do|are|am|can'?t|won'?t|isn'?t|aren'?t)\b", 2.0),
                  (r"\b(intermittent|flaky|throwing|throws|thrown|hangs?|deadlock|memory leak|race condition)\b", 2.0),
                  (r"\b(5\d\d|4\d\d) (error|response|status)?s?\b", 1.5),
                  (r"\b(not working|broken|regression|reproduce|fails? in production)\b", 1.2)],
    "data_analysis": [(r"\b(analy[sz]e|trend|correlat|cohort|breakdown|anomal|forecast|metrics?)\b", 2.0),
                      (r"\b(npv|irr|payback|sensitivity|scenarios?|assumptions?|regression|variance)\b", 2.0),
                      (r"\b(calculate|compute)\b", 1.2),
                      (r"\b(csv|spreadsheet|dataset|table of|dashboard|retention|churn)\b", 1.2)],
    "math_proof": [(r"\b(prove|proof|derive|theorem|lemma|closed form|optimi[sz]ation problem)\b", 3.0)],
    "agentic_multistep": [(r"\b(then|after that|finally)\b.{0,140}\b(then|finally|and)\b", 1.2),
                          (r"\b(orchestrate|multi-?step|for each .* (call|file|create)|workflow that)\b", 2.5),
                          (r"\bend-?to-?end pipeline\b", 2.5),
                          (r"\b(pulls?|fetch(es)?|reads?) .{0,60}(and|then) .{0,40}(classif|file|create|post|writes?|sends?)\b", 2.5),
                          (r"\b(nightly run|scheduled run|set (this|it) up to run)\b", 2.0),
                          (r"\b(use the .* tool|call the api and then)\b", 2.0)],
    "architecture_design": [(r"\b(architect(ure|ing)?|system design|design (a|an|the)|high level design|hld|scalab|should we use)\b", 2.5),
                            (r"\b(tradeoffs?|trade-offs?)\b", 1.5),
                            (r"\b(p99|multi-?region|failover|throughput|capacity plan)\b", 1.0)],
    "creative_writing": [(r"\b(write|draft|compose)\s+(a|an|the|\w+)?\s*(\w+\s+){0,2}(blog|post|email|newsletter|tagline|headline|subject lines?|story|poem|copy|announcement|variants?)\b", 2.5),
                         (r"\b(brand voice|tone of voice|marketing|campaign)\b", 1.5)],
    "translation": [(r"\b(translate|in (spanish|french|german|japanese|hindi|mandarin|portuguese))\b", 3.0)],
}

DOMAIN_PATTERNS: dict[str, list[str]] = {
    "finance": [r"\b(revenue|ebitda|invoice|gaap|ledger|p&l|forecast|arr|churn|npv|irr|audit|10-?k)\b"],
    "legal": [r"\b(contract|clause|indemnit|liabilit|nda|msa|terms of service|counsel|jurisdiction|breach)\b"],
    "healthcare": [r"\b(patient|clinical|diagnos|icd-?10|hipaa|ehr|prescri|dosage|medical)\b"],
    "security": [r"\b(vulnerab|cve|exploit|xss|sql injection|threat model|pentest|auth bypass|owasp)\b"],
    "engineering": [r"\b(deploy|kubernetes|latency|database|api|repo|microservice|ci/cd|refactor|code)\b"],
    "support": [r"\b(customer (asked|reported)|ticket|escalation|refund|sla breach|complaint)\b"],
}


@dataclass
class QueryFeatures:
    text: str
    task_type: str
    task_confidence: float
    domain: str
    signals: list[str] = field(default_factory=list)
    input_tokens: int = 0
    context_tokens: int = 0          # attached docs / retrieved chunks
    stable_context_tokens: int = 0   # reusable prefix -> caching candidate
    latency_slo: str = "interactive"
    zero_data_retention: bool = False
    output_class: str | None = None  # None -> take the task type's default
    scope_signals: list[str] = field(default_factory=list)
    output_class_shift: int = 0      # rungs to move along the output-class ladder
    source: str = "heuristic"        # heuristic | llm | caller
    # True when the task type is trustworthy: it came from the caller, from the
    # LLM classifier, or from a heuristic band with measured precision.
    verified: bool = False
    alternates: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["text"] = self.text[:200] + ("…" if len(self.text) > 200 else "")
        return d


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _score_patterns(text: str, table: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for key, pats in table.items():
        for pat, w in pats:
            if re.search(pat, text, re.I | re.S):
                scores[key] = scores.get(key, 0.0) + w
    return sorted(scores.items(), key=lambda kv: -kv[1])


def detect_scope(text: str, scope_cfg: dict[str, Any]) -> tuple[list[str], int]:
    """Shifts along the output-class ladder, independent of task type.

    Capped at +/-1 so three overlapping phrases cannot jump a short answer
    two rungs into a 32K ceiling.
    """
    hits, shift = [], 0
    for name, meta in (scope_cfg or {}).items():
        for pat in meta.get("patterns") or []:
            if re.search(pat, text, re.I | re.S):
                hits.append(name)
                shift += int(meta.get("output_class_shift", 0))
                break
    return sorted(set(hits)), max(-1, min(1, shift))


def detect_signals(text: str, signal_cfg: dict[str, Any], input_tokens: int) -> list[str]:
    hits = []
    for name, meta in signal_cfg.items():
        for pat in meta.get("patterns") or []:
            if re.search(pat, text, re.I | re.S):
                hits.append(name)
                break
    if input_tokens >= LONG_INPUT_TOKEN_THRESHOLD and "long_input" in signal_cfg:
        hits.append("long_input")
    return sorted(set(hits))


def extract(
    text: str,
    signal_cfg: dict[str, Any],
    *,
    scope_cfg: dict[str, Any] | None = None,
    context_tokens: int = 0,
    stable_context_tokens: int = 0,
    latency_slo: str = "interactive",
    zero_data_retention: bool = False,
    domain: str | None = None,
    task_type: str | None = None,
    output_class: str | None = None,
) -> QueryFeatures:
    """Free, deterministic first pass. Confidence drives whether we pay for more."""
    input_tokens = estimate_tokens(text) + context_tokens

    ranked = _score_patterns(text, TASK_PATTERNS)
    if task_type:
        chosen, confidence, source = task_type, 1.0, "caller"
    elif not ranked:
        chosen, confidence, source = "factual_qa", 0.25, "heuristic"
    else:
        chosen = ranked[0][0]
        top = ranked[0][1]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        # Confidence = how far the winner is clear of the runner-up, scaled by
        # absolute evidence. A 3.0-vs-2.8 win is not a confident classification.
        # Confidence must be gated by ABSOLUTE evidence, not just by the winner's
        # margin. Weighting separation alone meant a single weak 1.5-point match
        # with nothing competing scored 0.82 — "nothing else matched" read as
        # "I am sure". Measured on held-out data, that band was the source of
        # every silent misroute: stated 0.83, actual 0.60. Separation now only
        # amplifies evidence it already has, so it can never manufacture it.
        separation = (top - runner_up) / top if top else 0.0
        evidence = min(top / STRONG_EVIDENCE, 1.0)
        confidence = round(
            min(0.95, 0.20 + 0.45 * evidence + 0.35 * separation * evidence), 3
        )
        source = "heuristic"

    if domain is None:
        dom_hits = [
            (d, sum(1 for p in pats if re.search(p, text, re.I)))
            for d, pats in DOMAIN_PATTERNS.items()
        ]
        dom_hits = [(d, n) for d, n in dom_hits if n]
        domain = max(dom_hits, key=lambda kv: kv[1])[0] if dom_hits else "general"

    scope_hits, shift = detect_scope(text, scope_cfg or {})

    return QueryFeatures(
        text=text,
        task_type=chosen,
        task_confidence=confidence,
        domain=domain,
        signals=detect_signals(text, signal_cfg, input_tokens),
        input_tokens=input_tokens,
        context_tokens=context_tokens,
        stable_context_tokens=stable_context_tokens,
        latency_slo=latency_slo,
        zero_data_retention=zero_data_retention,
        output_class=output_class,
        scope_signals=scope_hits,
        output_class_shift=shift,
        source=source,
        alternates=[(k, round(v, 2)) for k, v in ranked[1:4]],
    )
