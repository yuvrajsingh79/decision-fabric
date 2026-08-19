"""Safety properties of the classification gate.

These are the guardrails, not accuracy tests. The heuristic's accuracy is poor
and that is a known, measured, accepted fact (see eval/). What must never
regress is the gate's *soundness*: when it commits, it must be right, because
nothing downstream re-checks a committed classification.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_fabric import features as F  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402

EVAL = ROOT / "eval"


@pytest.fixture(scope="module")
def fabric():
    return Fabric()


@pytest.fixture(scope="module")
def sealed(fabric):
    path = EVAL / "test.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return path, rows


def test_sealed_test_set_is_intact(sealed):
    """A silently edited test set turns every reported number into fiction."""
    path, _ = sealed
    expected = (EVAL / "test.sha256").read_text().strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        "eval/test.jsonl was modified. Re-seal deliberately and retire any "
        "result previously reported from it."
    )


def test_gate_never_commits_a_wrong_classification(fabric, sealed):
    """Silent errors are the only classification failures that reach production
    unchecked — everything the gate escalates gets a second opinion."""
    _, rows = sealed
    silent = []
    for r in rows:
        f = F.extract(r["q"], fabric.signals, scope_cfg=fabric.scope_signals)
        if fabric.may_commit(f.task_confidence)[0] and f.task_type != r["label"]:
            silent.append((r["q"], r["label"], f.task_type, f.task_confidence))
    assert not silent, "gate committed to a wrong task type:\n" + "\n".join(
        f"  conf={c:.2f} {exp} -> {got}: {q[:70]}" for q, exp, got, c in silent
    )


def test_uncalibrated_confidence_never_commits(fabric):
    """Absence of evidence is not evidence of reliability: a band with no
    measurements must escalate, not commit."""
    for band in fabric.calibration.get("bands", []):
        if band.get("n", 0) == 0 or band.get("precision") is None:
            mid = (float(band["lo"]) + float(band["hi"])) / 2
            allowed, reason = fabric.may_commit(mid)
            assert not allowed, f"committed on unmeasured band {band}: {reason}"


def test_committing_bands_meet_the_declared_precision_floor(fabric):
    floor = float(fabric.classification_cfg["min_commit_precision"])
    support = int(fabric.classification_cfg["min_band_support"])
    for band in fabric.calibration.get("bands", []):
        if band.get("commit"):
            assert band["precision"] >= floor, f"{band} below floor {floor}"
            assert band["n"] >= support, f"{band} has less than {support} observations"


def test_gate_defaults_to_paying_when_uncertain(fabric):
    """The expensive failure is misrouting, not classifying."""
    from decision_fabric.classifier import should_refine
    f = F.extract("orders api started returning empty arrays", fabric.signals,
                  scope_cfg=fabric.scope_signals)
    refine, reason = should_refine(f, True, fabric)
    assert refine, f"should have escalated: {reason}"


def test_unverified_classification_routes_conservatively(tmp_path):
    """If the gate wanted an LLM opinion and never got one, the task type is a
    guess the system already called unreliable. Routing cheaply on it is how a
    cost optimiser silently degrades answers while reporting a saving."""
    from decision_fabric.router import Router
    r = Router(db_path=tmp_path / "u.db", dry_run=True, use_llm_classifier=True)
    try:
        d = r.route("how do I stop my kubernetes pods from getting evicted?",
                    execute=False, learn=False)
        assert d.features.verified is False
        base = r.fabric.policy("balanced")["quality_floor"]
        premium = r.fabric.classification_cfg["unverified_classification"]["quality_floor_premium"]
        assert d.policy["quality_floor"] == pytest.approx(base + premium)
        assert d.policy["cascade"] is False
        assert d.policy["classification_unverified"] is True
        assert any("UNVERIFIED" in t for t in d.trace)
    finally:
        r.close()


def test_caller_supplied_task_type_is_trusted(tmp_path):
    """An explicit task type from the caller needs no classification at all."""
    from decision_fabric.router import Router
    r = Router(db_path=tmp_path / "v.db", dry_run=True)
    try:
        d = r.route("anything at all", task_type="code_gen", execute=False, learn=False)
        assert d.features.verified is True
        assert not d.policy.get("classification_unverified")
    finally:
        r.close()
