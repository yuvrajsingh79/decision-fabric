"""Knowledge-graph mechanics beyond the flat capability join.

Four properties: derived task similarity, evidence reachable by traversal,
task definitions that learn, and escalation as a lattice rather than a chain.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric.ontology import EdgeType, NodeType  # noqa: E402
from decision_fabric.seed import Fabric, capability_similarity  # noqa: E402


@pytest.fixture
def fab():
    return Fabric()


@pytest.fixture
def router(tmp_path):
    from decision_fabric.router import Router
    r = Router(db_path=tmp_path / "kg.db", dry_run=True, use_llm_classifier=False)
    yield r
    r.close()


# ------------------------------------------------------------ SIMILAR_TO

def test_similarity_is_symmetric_and_bounded(fab):
    for src, dst, data in [(s, d, a) for s, d, k, a in fab.kg.g.edges(keys=True, data=True)
                           if a.get("etype") == EdgeType.SIMILAR_TO.value]:
        back = fab.kg.edge(dst, src, EdgeType.SIMILAR_TO)
        assert back is not None, f"{src}->{dst} has no reverse edge"
        assert back["similarity"] == data["similarity"]
        assert 0.0 <= data["similarity"] <= 1.0


def test_similarity_weights_by_demanded_level(fab):
    """Presence alone is too coarse — two tasks are similar when they are hard
    in the same places."""
    identical = capability_similarity({"a": 0.8, "b": 0.6}, {"a": 0.8, "b": 0.6})
    same_caps_diff_levels = capability_similarity({"a": 0.9}, {"a": 0.3})
    assert identical == pytest.approx(1.0)
    assert same_caps_diff_levels < 0.5


def test_no_task_is_similar_to_itself(fab):
    for node in fab.kg.nodes_of(NodeType.TASK_TYPE):
        targets = [d for d, _ in fab.kg.out_edges(node, EdgeType.SIMILAR_TO)]
        assert node not in targets


def test_evidence_is_borrowed_from_similar_tasks(router):
    seed = router.learning.seed_quality("claude-sonnet-5", router.kg.requirements_of("debugging"))
    assert router.learning.posterior("claude-sonnet-5", "debugging", seed) is None
    for _ in range(10):
        router.learning.observe("claude-sonnet-5", "code_review", True, "verifier")
    b_s, b_n, sources = router.learning.borrowed_evidence("claude-sonnet-5", "debugging")
    assert b_n > 0 and sources
    post = router.learning.posterior("claude-sonnet-5", "debugging", seed)
    assert post is not None and post[0] > seed
    assert post[1] == 0, "borrowed evidence must not inflate the DIRECT observation count"


def test_borrowed_evidence_is_capped(router):
    for task in ("code_review", "code_gen", "architecture_design"):
        for _ in range(200):
            router.learning.observe("claude-sonnet-5", task, True, "verifier")
    _, b_n, _ = router.learning.borrowed_evidence("claude-sonnet-5", "debugging")
    assert b_n <= router.learning.max_borrowed_weight + 1e-6


# -------------------------------------------------------------- evidence

def test_evidence_becomes_traversable(router):
    for _ in range(3):
        router.learning.observe("claude-opus-5", "debugging", True, "verifier")
    router.learning.sync_evidence()
    about_model = router.kg.evidence_about_model("claude-opus-5")
    about_task = router.kg.evidence_about_task("debugging")
    assert any(e["task_type"] == "debugging" and e["trials"] == 3 for e in about_model)
    assert any(e["model_id"] == "claude-opus-5" for e in about_task)


def test_evidence_is_aggregated_not_one_node_per_event(router):
    for _ in range(50):
        router.learning.observe("claude-opus-5", "debugging", True, "verifier")
    router.learning.sync_evidence()
    n = len(router.kg.nodes_of(NodeType.EVIDENCE))
    assert n == 1, f"50 observations produced {n} nodes; must aggregate per (model, task)"


def test_evidence_node_updates_in_place(router):
    router.learning.observe("claude-opus-5", "debugging", True, "verifier")
    router.learning.sync_evidence()
    router.learning.observe("claude-opus-5", "debugging", False, "escalation")
    router.learning.sync_evidence()
    e = router.kg.evidence_about_model("claude-opus-5")[0]
    assert e["trials"] == 2 and e["successes"] == 1
    assert len(router.kg.nodes_of(NodeType.EVIDENCE)) == 1


# ---------------------------------------------------- REQUIRES learning

def test_thin_margin_failures_raise_the_bar(router):
    task, cap = "code_gen", "code_synthesis"
    authored = router.kg.authored_requirement(task, cap)
    for _ in range(6):
        router.telemetry.record_requirement_feedback(task, cap, "claude-sonnet-5",
                                                     margin=0.03, required=authored, success=False)
    changes = router.learning.recompute_requirements()
    assert changes
    assert router.kg.requirements_of(task)[cap] > authored


def test_raising_is_faster_than_lowering(router):
    """Asymmetric on purpose: a thin margin plus a failure is attributable
    evidence; success with headroom never proves less would have sufficed."""
    up_task, up_cap = "code_gen", "code_synthesis"
    dn_task, dn_cap = "extraction", "structured_extraction"
    up0 = router.kg.authored_requirement(up_task, up_cap)
    dn0 = router.kg.authored_requirement(dn_task, dn_cap)
    for _ in range(8):
        router.telemetry.record_requirement_feedback(up_task, up_cap, "m", 0.03, up0, False)
        router.telemetry.record_requirement_feedback(dn_task, dn_cap, "m", 0.35, dn0, True)
    router.learning.recompute_requirements()
    up_delta = router.kg.requirements_of(up_task)[up_cap] - up0
    dn_delta = dn0 - router.kg.requirements_of(dn_task)[dn_cap]
    assert up_delta > 0 and dn_delta > 0
    assert up_delta > dn_delta * 2, f"raise {up_delta:.3f} not clearly faster than lower {dn_delta:.3f}"


def test_requirement_drift_is_bounded(router):
    task, cap = "code_gen", "code_synthesis"
    authored = router.kg.authored_requirement(task, cap)
    for _ in range(500):
        router.telemetry.record_requirement_feedback(task, cap, "m", 0.01, authored, False)
    for _ in range(20):
        router.learning.recompute_requirements()
    drift = abs(router.kg.requirements_of(task)[cap] - authored)
    assert drift <= router.learning.req_cfg["max_drift"] + 1e-6


def test_authored_requirement_never_moves(router):
    task, cap = "code_gen", "code_synthesis"
    authored = router.kg.authored_requirement(task, cap)
    for _ in range(10):
        router.telemetry.record_requirement_feedback(task, cap, "m", 0.02, authored, False)
    router.learning.recompute_requirements()
    assert router.kg.authored_requirement(task, cap) == authored


def test_insufficient_evidence_changes_nothing(router):
    before = dict(router.kg.requirements_of("code_gen"))
    router.telemetry.record_requirement_feedback("code_gen", "code_synthesis", "m", 0.02, 0.8, False)
    assert router.learning.recompute_requirements() == []
    assert router.kg.requirements_of("code_gen") == before


# ------------------------------------------------------------- lattice

def test_default_ladder_is_the_full_chain(fab):
    assert fab.kg.ladder() == ["claude-haiku-4-5", "claude-sonnet-5",
                               "claude-opus-5", "claude-fable-5"]


def test_task_path_can_skip_a_rung(fab):
    assert fab.kg.ladder("math_proof") == ["claude-haiku-4-5", "claude-opus-5", "claude-fable-5"]


def test_task_path_terminates(fab):
    """A declared path must not fall through to the default chain past its end."""
    assert fab.kg.ladder("classification") == ["claude-haiku-4-5", "claude-sonnet-5"]
    assert fab.kg.ladder("chitchat") == ["claude-haiku-4-5"]


def test_task_path_can_start_above_the_bottom(fab):
    assert fab.kg.ladder("agentic_multistep")[0] == "claude-sonnet-5"


def test_parallel_escalation_edges_coexist(fab):
    """The default haiku->sonnet hop and a task-specific one must both survive;
    keying edges by type alone silently dropped one."""
    outs = list(fab.kg.out_edges("claude-haiku-4-5", EdgeType.ESCALATES_TO))
    tasks = {a.get("for_task") for _, a in outs}
    assert None in tasks and len(tasks) > 1


def test_escalation_respects_the_task_path(router):
    d = router.route("classify this ticket", task_type="classification",
                     latency_slo="background", policy="economy")
    path = router.kg.ladder("classification")
    for a in d.attempts:
        assert a.model_id in path, f"escalated to {a.model_id}, off the declared path {path}"


def test_unknown_task_in_an_escalation_path_fails_at_boot(tmp_path):
    import shutil, yaml
    cfg = tmp_path / "config"
    shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "config", cfg)
    (cfg / "extensions.yaml").unlink(missing_ok=True)
    cat = yaml.safe_load((cfg / "models.yaml").read_text())
    cat["escalation_paths"]["no_such_task"] = ["claude-haiku-4-5", "claude-opus-5"]
    (cfg / "models.yaml").write_text(yaml.safe_dump(cat))
    with pytest.raises(ValueError, match="unknown task"):
        Fabric(cfg)


def test_non_monotonic_escalation_path_fails_at_boot(tmp_path):
    import shutil, yaml
    cfg = tmp_path / "config"
    shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "config", cfg)
    (cfg / "extensions.yaml").unlink(missing_ok=True)
    cat = yaml.safe_load((cfg / "models.yaml").read_text())
    cat["escalation_paths"]["debugging"] = ["claude-opus-5", "claude-haiku-4-5"]
    (cfg / "models.yaml").write_text(yaml.safe_dump(cat))
    with pytest.raises(ValueError, match="monotonic"):
        Fabric(cfg)
