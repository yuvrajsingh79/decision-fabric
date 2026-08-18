"""Tests for the parts where a bug would silently cost money or send an
invalid request. Run: PYTHONPATH=src python -m pytest tests/ -q
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric.config_planner import plan_request  # noqa: E402
from decision_fabric.ontology import EdgeType, Requirement  # noqa: E402
from decision_fabric.pricing import project_cost  # noqa: E402
from decision_fabric.reasoner import derive_requirements, probability_of_failure  # noqa: E402
from decision_fabric.router import Router  # noqa: E402
from decision_fabric.seed import Fabric, effective_pricing  # noqa: E402


@pytest.fixture(scope="module")
def fabric():
    return Fabric()


@pytest.fixture
def router(tmp_path):
    r = Router(db_path=tmp_path / "t.db", dry_run=True, use_llm_classifier=False)
    yield r
    r.close()


# ----------------------------------------------------------------- graph seed

def test_graph_builds_and_is_connected(fabric):
    stats = fabric.kg.stats()
    assert stats["Model"] == 4
    assert stats["REQUIRES"] > 20
    assert fabric.kg.ladder() == [
        "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5", "claude-fable-5"
    ]


def test_every_task_requires_a_real_capability(fabric):
    for task in fabric.ontology["task_types"]:
        reqs = fabric.kg.requirements_of(task)
        assert reqs, f"{task} has no requirements"
        for cap in reqs:
            assert fabric.kg.has(f"cap:{cap}")


# ------------------------------------------------------- requirement derivation

def test_signals_elevate_but_never_saturate(fabric):
    """Stacked signals must not push a requirement to an unreachable level."""
    reqs, _ = derive_requirements(
        fabric.kg, "debugging", "engineering",
        ["adversarial", "correctness_critical", "multi_step", "novelty", "ambiguity"],
    )
    assert reqs["deep_reasoning"].level < 0.99
    # ...and must still be strictly higher than the unelevated baseline.
    plain, _ = derive_requirements(fabric.kg, "debugging", "general", [])
    assert reqs["deep_reasoning"].level > plain["deep_reasoning"].level


def test_regulated_domain_raises_the_bar(fabric):
    general, _ = derive_requirements(fabric.kg, "extraction", "general", [])
    health, _ = derive_requirements(fabric.kg, "extraction", "healthcare", [])
    assert health["domain_precision"].level > general.get(
        "domain_precision", Requirement("domain_precision", 0.0)
    ).level


# ------------------------------------------------------------ config validity

@pytest.mark.parametrize("model_id", ["claude-haiku-4-5", "claude-sonnet-5",
                                      "claude-opus-5", "claude-fable-5"])
def test_plan_never_emits_a_knob_the_model_rejects(fabric, model_id):
    """The whole point of encoding the config surface in the graph: an invalid
    knob is a 400, not a worse answer."""
    spec = fabric.kg.model(model_id)
    plan = plan_request(
        spec,
        required={"deep_reasoning": 0.95, "code_synthesis": 0.9},
        output_class_meta=fabric.output_classes["medium"],
        policy=fabric.policy("balanced"),
        slo={"name": "background", **fabric.latency_slos["background"]},
    )
    kw = plan.to_request_kwargs([{"role": "user", "content": "hi"}])
    cs = spec.config_surface

    if not cs.get("supports_effort"):
        assert "output_config" not in kw, f"{model_id} has no effort knob"
    if cs.get("thinking_style") == "always_on":
        assert "thinking" not in kw, "Fable 5 rejects any explicit thinking config"
    if cs.get("thinking_style") == "adaptive":
        assert kw["thinking"] == {"type": "adaptive"}
    if cs.get("thinking_style") == "budget_tokens" and "thinking" in kw:
        assert kw["thinking"]["type"] == "enabled"
        assert kw["thinking"]["budget_tokens"] >= cs["thinking_min_budget"]
        assert kw["thinking"]["budget_tokens"] < kw["max_tokens"], "budget must be < max_tokens"
    assert "temperature" not in kw
    assert kw["max_tokens"] <= spec.max_output_tokens


def test_max_tokens_leaves_room_for_the_effort_it_chose(fabric):
    """A ceiling below what the chosen effort will generate truncates the answer."""
    spec = fabric.kg.model("claude-opus-5")
    plan = plan_request(
        spec,
        required={"deep_reasoning": 0.95},
        output_class_meta=fabric.output_classes["medium"],
        policy=fabric.policy("balanced"),
        slo={"name": "background", **fabric.latency_slos["background"]},
        effort_multipliers=fabric.modifiers["effort_output_multiplier"],
    )
    mult = fabric.modifiers["effort_output_multiplier"][plan.effort]
    assert plan.max_tokens >= fabric.output_classes["medium"]["expected_output_tokens"] * mult


def test_tiny_stable_prefix_does_not_enable_caching(fabric):
    """Below the ~1024-token minimum, cache_control only costs the write premium."""
    spec = fabric.kg.model("claude-sonnet-5")
    plan = plan_request(
        spec, required={"summarization": 0.7},
        output_class_meta=fabric.output_classes["short"],
        policy=fabric.policy("balanced"),
        slo={"name": "interactive", **fabric.latency_slos["interactive"]},
        stable_context_tokens=400,
    )
    assert not plan.use_cache


def test_batch_and_fast_mode_are_mutually_exclusive(fabric):
    spec = fabric.kg.model("claude-opus-5")
    plan = plan_request(
        spec, required={"deep_reasoning": 0.8},
        output_class_meta=fabric.output_classes["short"],
        policy=fabric.policy("quality"),
        slo={"name": "batch", **fabric.latency_slos["batch"]},
    )
    assert not (plan.batch and plan.fast_mode)


# ------------------------------------------------------------------- pricing

def test_intro_pricing_expires(fabric):
    from datetime import date
    spec = fabric.kg.model("claude-sonnet-5")
    assert effective_pricing(spec, date(2026, 8, 1)) == (2.00, 10.00)
    assert effective_pricing(spec, date(2026, 9, 1)) == (3.00, 15.00)


def test_batch_halves_the_bill(fabric):
    spec = fabric.kg.model("claude-haiku-4-5")
    kw = dict(fresh_input_tokens=10_000, output_tokens=1_000, effort="none")
    normal = project_cost(spec, fabric.modifiers, **kw).total_usd
    batched = project_cost(spec, fabric.modifiers, batch=True, **kw).total_usd
    assert batched == pytest.approx(normal * 0.5)


def test_cache_read_is_cheaper_than_fresh_input(fabric):
    spec = fabric.kg.model("claude-opus-5")
    fresh = project_cost(spec, fabric.modifiers, fresh_input_tokens=50_000,
                         output_tokens=100, effort="none").total_usd
    cached = project_cost(spec, fabric.modifiers, fresh_input_tokens=0,
                          cache_read_tokens=50_000, output_tokens=100, effort="none").total_usd
    assert cached < fresh * 0.2


def test_effort_dominates_output_cost(fabric):
    spec = fabric.kg.model("claude-opus-5")
    lo = project_cost(spec, fabric.modifiers, fresh_input_tokens=1000,
                      output_tokens=2000, effort="low").total_usd
    hi = project_cost(spec, fabric.modifiers, fresh_input_tokens=1000,
                      output_tokens=2000, effort="max").total_usd
    assert hi > lo * 4


# ------------------------------------------------------------ failure model

def test_probability_of_failure_is_monotonic():
    assert probability_of_failure(0.9, 0.55, 0.1) < probability_of_failure(0.6, 0.55, 0.1)
    assert probability_of_failure(0.5, 0.55, 0.1) > 0.5
    assert 0.0 <= probability_of_failure(0.99, 0.2, 0.1) <= 1.0


# -------------------------------------------------------------------- routing

def test_trivial_query_does_not_reach_the_flagship(router):
    d = router.route("Hi, thanks for the help earlier!", policy="economy")
    assert d.chosen_model == "claude-haiku-4-5"
    assert d.saved_usd > 0


def test_hard_regulated_query_is_not_routed_cheap(router):
    d = router.route(
        "Prove that this rebalancing algorithm terminates and derive its worst-case complexity.",
        policy="critical", latency_slo="background",
    )
    assert router.kg.model(d.chosen_model).rung >= 2


def test_domain_floor_overrides_a_cheaper_request(router):
    d = router.route(
        "Given this patient intake summary, list the documented allergies and medications.",
        policy="economy", latency_slo="background",
    )
    assert d.policy["name"] == "quality"
    assert d.policy.get("forced_by") == "domain:healthcare"


def test_realtime_slo_excludes_slow_models(router):
    d = router.route("Prove this theorem and derive the complexity.",
                     policy="balanced", latency_slo="realtime", execute=False)
    chosen = router.kg.model(d.selection.primary.model.id)
    assert chosen.nfr["relative_latency"] <= router.fabric.latency_slos["realtime"]["max_relative_latency"]


def test_escalation_never_targets_an_ineligible_model(router):
    """Escalating into a model the SLO already excluded is a correctness bug."""
    d = router.route("Debug this intermittent production deadlock.",
                     policy="balanced", latency_slo="realtime")
    ceiling = router.fabric.latency_slos["realtime"]["max_relative_latency"]
    for a in d.attempts:
        assert router.kg.model(a.model_id).nfr["relative_latency"] <= ceiling


def test_context_window_excludes_models_that_cannot_hold_it(router):
    d = router.route("Summarize the attached corpus.", context_tokens=400_000,
                     latency_slo="batch", execute=False)
    assert router.kg.model(d.selection.primary.model.id).context_window >= 400_000
    haiku = next(c for c in d.selection.all_candidates if c.model.id == "claude-haiku-4-5")
    assert not haiku.eligible


def test_decision_is_reproducible(router):
    q = "Extract the vendor and total from this invoice."
    a = router.route(q, learn=False, execute=False)
    b = router.route(q, learn=False, execute=False)
    assert a.selection.first_plan.summary() == b.selection.first_plan.summary()


def test_every_decision_carries_an_audit_trail(router):
    d = router.route("Review this pull request for security issues.", policy="quality")
    assert d.trace and any("REQUIRES" in t for t in d.trace)
    assert d.requirements
    assert d.to_dict()["chosen_model"] == d.chosen_model


# ------------------------------------------------------------------- learning

def test_success_never_lowers_a_capability_level(router):
    """A flat Beta(1,1) prior would drop a 0.94 model to 0.67 on one success."""
    kg = router.kg
    before = {m.id: kg.provided_level(m.id, "deep_reasoning") for m in kg.models()}
    for _ in range(5):
        router.learning.observe("claude-opus-5", "debugging", True, "verifier")
    router.learning.recompute_levels()
    assert kg.provided_level("claude-opus-5", "deep_reasoning") >= before["claude-opus-5"]


def test_drift_is_bounded(router):
    for _ in range(200):
        router.learning.observe("claude-sonnet-5", "code_gen", False, "escalation")
    router.learning.recompute_levels()
    cap = "code_synthesis"
    seed = router.kg.seed_level("claude-sonnet-5", cap)
    cur = router.kg.provided_level("claude-sonnet-5", cap)
    assert abs(cur - seed) <= router.fabric.learning_cfg["max_level_drift"] + 1e-6


def test_learning_cannot_revoke_eligibility(router):
    """Evidence adjusts scoring, never the structural capability gate — the
    regression that turned +28% savings into -26%."""
    q = "According to the attached contract, what is the termination notice period?"
    before = router.route(q, execute=False, learn=False)
    elig_before = {c.model.id for c in before.selection.all_candidates if c.eligible}
    for _ in range(50):
        router.learning.observe("claude-sonnet-5", "rag_qa", False, "escalation")
    router.learning.recompute_levels()
    after = router.route(q, execute=False, learn=False)
    elig_after = {c.model.id for c in after.selection.all_candidates if c.eligible}
    assert elig_before == elig_after


def test_evidence_moves_expected_quality(router):
    q = "Extract the vendor and total from this invoice."
    before = router.route(q, execute=False, learn=False)
    q_before = next(c.expected_quality for c in before.selection.all_candidates
                    if c.model.id == "claude-haiku-4-5")
    for _ in range(30):
        router.learning.observe("claude-haiku-4-5", "extraction", True, "verifier")
    router.learning.recompute_levels()
    after = router.route(q, execute=False, learn=False)
    q_after = next(c.expected_quality for c in after.selection.all_candidates
                   if c.model.id == "claude-haiku-4-5")
    assert q_after > q_before


def test_telemetry_reports_savings(router):
    for _ in range(3):
        router.route("Classify this ticket as billing or technical.", policy="economy")
    rep = router.telemetry.savings_report()
    assert rep["routes"] == 3
    assert rep["baseline_usd"] > rep["actual_usd"]


def test_zero_data_retention_excludes_models_that_require_retention(router):
    """Fable 5 is unavailable under ZDR — a compliance boundary, not a preference."""
    q = "Prove that this rebalancing algorithm terminates and derive its complexity."
    normal = router.route(q, policy="critical", latency_slo="background", execute=False)
    zdr = router.route(q, policy="critical", latency_slo="background",
                       zero_data_retention=True, execute=False)
    assert any(c.model.id == "claude-fable-5" and c.eligible
               for c in normal.selection.all_candidates)
    fable = next(c for c in zdr.selection.all_candidates if c.model.id == "claude-fable-5")
    assert not fable.hard_ok
    assert zdr.selection.primary.model.id != "claude-fable-5"


def test_credential_detection_ignores_an_empty_config_dir(tmp_path, monkeypatch):
    """`ant auth status` creates ~/.config/anthropic before any login happens;
    treating that as 'live' turns a clean dry-run into a run of auth failures."""
    from decision_fabric.executor import Executor
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    home = tmp_path / "home"
    (home / ".config" / "anthropic" / "credentials").mkdir(parents=True)
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(home)))
    assert Executor._credentials_available() is False

    (home / ".config" / "anthropic" / "credentials" / "default.json").write_text(
        '{"access_token": "x"}'
    )
    assert Executor._credentials_available() is True


def test_dotenv_rejects_placeholder_keys(tmp_path, monkeypatch):
    """A placeholder key outranks an `ant auth login` profile, so loading one
    silently disables real credentials."""
    from decision_fabric.env import load_dotenv
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=sk-ant-your-real-key\n")
    assert load_dotenv(f) == []
    import os
    assert "ANTHROPIC_API_KEY" not in os.environ

    f.write_text("ANTHROPIC_API_KEY=sk-ant-api03-" + "a" * 80 + "\n")
    assert load_dotenv(f) == ["ANTHROPIC_API_KEY"]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
