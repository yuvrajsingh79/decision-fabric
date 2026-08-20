"""Capability subsumption and open-world extension.

Two properties matter here. Inheritance must make a newly added capability
usable rather than catastrophic — before it existed, adding one zeroed every
model in the catalog and forced the top rung. And extension must be atomic: a
rejected extension has to leave the graph exactly as it was, because a
half-applied one still routes.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric.ontology import EdgeType, ModelSpec  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402


@pytest.fixture
def kg():
    return Fabric().kg


# ---------------------------------------------------------------- taxonomy

def test_taxonomy_is_acyclic(kg):
    for node in kg.nodes_of_capability() if hasattr(kg, "nodes_of_capability") else []:
        pass
    from decision_fabric.ontology import NodeType
    for node in kg.nodes_of(NodeType.CAPABILITY):
        cap = node.removeprefix("cap:")
        assert cap not in kg.ancestors_of(cap), f"{cap} subsumes itself"


def test_declared_levels_are_never_inherited(kg):
    """Inheritance must be inert wherever a model has declared a level —
    otherwise adding a taxonomy silently reprices the whole catalog."""
    for spec in kg.models():
        for cap in spec.provides:
            r = kg.resolve_level(spec.id, cap)
            assert r.declared, f"{spec.id}/{cap} resolved as {r.source}"
            assert r.level == pytest.approx(spec.provides[cap])


def test_undeclared_capability_inherits_from_ancestor(kg):
    kg.add_capability("statute_recall", parent="domain_precision")
    for spec in kg.models():
        r = kg.resolve_level(spec.id, "statute_recall")
        assert r.source == "inherited"
        assert r.via == "domain_precision"
        assert r.hops == 1
        parent = spec.provides["domain_precision"]
        assert r.level == pytest.approx(round(parent * kg.inheritance_discount, 4))


def test_inheritance_is_discounted_per_hop(kg):
    kg.add_capability("mid", parent="deep_reasoning")
    kg.add_capability("leaf", parent="mid")
    r = kg.resolve_level("claude-opus-5", "leaf")
    assert r.hops == 2
    assert r.discount == pytest.approx(kg.inheritance_discount ** 2)
    assert r.level < kg.resolve_level("claude-opus-5", "mid").level


def test_inheritance_is_bounded_by_max_hops(kg):
    kg.max_inheritance_hops = 2
    kg.add_capability("a1", parent="deep_reasoning")
    kg.add_capability("a2", parent="a1")
    kg.add_capability("a3", parent="a2")
    assert kg.resolve_level("claude-opus-5", "a3").source == "unknown"


def test_orphan_capability_resolves_unknown_not_wrong(kg):
    """No ancestor is not the same as a level of zero being *declared* — the
    distinction has to survive into the explanation."""
    kg.add_capability("freefloating")
    r = kg.resolve_level("claude-opus-5", "freefloating")
    assert r.source == "unknown" and r.level == 0.0
    assert "no ancestor" in r.explain()


def test_inheritance_never_exceeds_the_source_level(kg):
    kg.add_capability("derived", parent="deep_reasoning")
    for spec in kg.models():
        assert kg.resolve_level(spec.id, "derived").level <= spec.provides["deep_reasoning"]


# --------------------------------------------------------------- extension

def test_extension_is_atomic_on_failure(kg):
    before = (kg.stats()["_nodes"], kg.stats()["_edges"])
    for bad in [
        lambda: kg.add_capability("deep_reasoning"),
        lambda: kg.add_capability("x", parent="does_not_exist"),
        lambda: kg.add_task_type("t", requires={"nope": 0.5}, output_class="short"),
        lambda: kg.add_task_type("t", requires={}, output_class="short"),
        lambda: kg.add_task_type("t", requires={"deep_reasoning": 1.5}, output_class="short"),
        lambda: kg.add_task_type("t", requires={"deep_reasoning": 0.5}, output_class="nope"),
    ]:
        with pytest.raises(ValueError):
            bad()
    assert (kg.stats()["_nodes"], kg.stats()["_edges"]) == before


def test_self_parenting_is_rejected(kg):
    """A capability cannot subsume itself; the guard must fire and leave no node."""
    before = kg.stats()["_nodes"]
    with pytest.raises(ValueError):
        kg.add_capability("loopy", parent="loopy")
    assert not kg.has("cap:loopy")
    assert kg.stats()["_nodes"] == before


def test_ancestor_walk_terminates_on_a_manual_cycle(kg):
    """Even if a cycle is introduced out-of-band, resolution must terminate
    rather than hang — the walk is cycle-guarded, not merely depth-limited."""
    kg.add_capability("c1", parent="deep_reasoning")
    kg.add_capability("c2", parent="c1")
    kg.g.add_edge("cap:c2", "cap:deep_reasoning", key=EdgeType.SUBSUMES.value,
                  etype=EdgeType.SUBSUMES.value)
    chain = kg.ancestors_of("c1")
    assert len(chain) <= kg.max_inheritance_hops
    assert len(set(chain)) == len(chain), "walk revisited a node"


def test_added_task_type_routes_immediately(kg, tmp_path):
    from decision_fabric.router import Router
    r = Router(db_path=tmp_path / "x.db", dry_run=True, use_llm_classifier=False)
    try:
        r.kg.add_capability("statute_recall", parent="domain_precision")
        r.kg.add_task_type("compliance_mapping",
                           requires={"statute_recall": 0.70, "long_context": 0.70},
                           output_class="medium")
        d = r.route("map these controls to the framework", task_type="compliance_mapping",
                    latency_slo="background", execute=False, learn=False)
        assert "statute_recall" in d.requirements
        assert d.selection.primary.model.id in {m.id for m in r.kg.models()}
    finally:
        r.close()


def test_added_model_splices_into_the_ladder(kg):
    spec = ModelSpec(
        id="test-model-x", display_name="Test X", rung=1,
        context_window=200_000, max_output_tokens=8_000,
        pricing={"input_per_mtok": 2.0, "output_per_mtok": 8.0},
        nfr={"relative_latency": 0.3},
        config_surface={"thinking_style": "adaptive", "supports_effort": True,
                        "supports_batch": True, "supports_caching": True},
        provides={"instruction_following": 0.8, "deep_reasoning": 0.7, "long_context": 0.75},
    )
    kg.add_model(spec, after="claude-haiku-4-5")
    ladder = kg.ladder()
    assert ladder.index("test-model-x") == ladder.index("claude-haiku-4-5") + 1
    assert "claude-sonnet-5" in ladder and len(ladder) == 5


def test_model_declaring_unknown_capability_is_rejected(kg):
    spec = ModelSpec(id="bad", display_name="Bad", rung=0, context_window=1000,
                     max_output_tokens=100, pricing={"input_per_mtok": 1, "output_per_mtok": 1},
                     nfr={}, config_surface={}, provides={"telepathy": 0.9})
    with pytest.raises(ValueError, match="unknown capabilities"):
        kg.add_model(spec)
    assert not kg.has("bad")


def test_undeclared_capabilities_report(kg):
    kg.add_capability("statute_recall", parent="domain_precision")
    gaps = kg.undeclared_capabilities("claude-opus-5")
    assert "statute_recall" in gaps
    assert gaps["statute_recall"].source == "inherited"


# -------------------------------------------------------------- persistence

def test_extensions_survive_a_restart(tmp_path, monkeypatch):
    """An extension that vanishes on restart is not an extension."""
    import shutil
    from decision_fabric.seed import Fabric
    cfg = tmp_path / "config"
    shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "config", cfg)
    (cfg / "extensions.yaml").unlink(missing_ok=True)

    f1 = Fabric(cfg)
    f1.extend_capability("statute_recall", parent="domain_precision")
    f1.extend_task_type("compliance_mapping",
                        requires={"statute_recall": 0.7}, output_class="medium")
    nodes = f1.kg.stats()["_nodes"]

    f2 = Fabric(cfg)                      # a fresh process would do exactly this
    assert f2.kg.stats()["_nodes"] == nodes
    assert f2.kg.has("cap:statute_recall")
    assert f2.kg.requirements_of("compliance_mapping") == {"statute_recall": 0.7}
    assert f2.applied_extensions == ["capability:statute_recall",
                                     "task_type:compliance_mapping"]


def test_overlay_rejects_a_duplicate_entry(tmp_path):
    import shutil
    from decision_fabric.seed import Fabric
    cfg = tmp_path / "config"
    shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "config", cfg)
    (cfg / "extensions.yaml").unlink(missing_ok=True)
    f = Fabric(cfg)
    f.extend_capability("dup_cap", parent="deep_reasoning")
    with pytest.raises(ValueError):
        f.extend_capability("dup_cap", parent="deep_reasoning")


def test_a_broken_overlay_fails_loudly_at_boot(tmp_path):
    """Silently skipping a bad extension would route traffic against an
    ontology nobody has seen."""
    import shutil
    from decision_fabric.seed import Fabric
    cfg = tmp_path / "config"
    shutil.copytree(pathlib.Path(__file__).resolve().parents[1] / "config", cfg)
    (cfg / "extensions.yaml").write_text(
        "task_types:\n- name: broken\n  requires: {no_such_capability: 0.5}\n"
        "  output_class: medium\n")
    with pytest.raises(ValueError, match="unknown capabilities"):
        Fabric(cfg)
