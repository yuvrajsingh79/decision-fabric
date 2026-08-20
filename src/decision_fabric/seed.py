"""Load the YAML ontology + model catalog into a KnowledgeGraph."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .graph import KnowledgeGraph
from .ontology import EdgeType, ModelSpec, NodeType

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


class Fabric:
    """The loaded graph plus the raw config the reasoner needs alongside it."""

    def __init__(self, config_dir: Path | str = CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir)
        self.ontology = _load(self.config_dir / "graph.yaml")
        self.catalog = _load(self.config_dir / "models.yaml")
        self.policies_cfg = _load(self.config_dir / "policies.yaml")
        cal_name = self.policies_cfg.get("classification", {}).get(
            "calibration_file", "calibration.yaml"
        )
        cal_path = self.config_dir / cal_name
        self.calibration = _load(cal_path) if cal_path.is_file() else {"bands": []}
        self.kg = build_graph(self.ontology, self.catalog, self.policies_cfg)
        # Extensions added at runtime are recorded in an overlay rather than
        # written back into the hand-authored config. The base ontology stays
        # reviewable, and everything an operator added is visible in one file
        # and revertible by deleting it.
        self.extensions_path = self.config_dir / "extensions.yaml"
        self.applied_extensions = apply_extensions(self.kg, self.extensions_path)

    # convenience passthroughs used all over the reasoner
    @property
    def modifiers(self) -> dict[str, Any]:
        return self.catalog["modifiers"]

    @property
    def output_classes(self) -> dict[str, Any]:
        return self.ontology["output_classes"]

    @property
    def latency_slos(self) -> dict[str, Any]:
        return self.ontology["latency_slos"]

    @property
    def signals(self) -> dict[str, Any]:
        return self.ontology["complexity_signals"]

    @property
    def classification_cfg(self) -> dict[str, Any]:
        return self.policies_cfg.get("classification", {})

    def may_commit(self, confidence: float) -> tuple[bool, str]:
        """May the heuristic be trusted at this confidence, per measured precision?

        Returns (allowed, reason). Unmeasured or under-supported bands never
        commit: absence of evidence is not evidence of reliability.
        """
        for b in self.calibration.get("bands") or []:
            if float(b["lo"]) <= confidence < float(b["hi"]):
                if b.get("commit"):
                    return True, (
                        f"band [{b['lo']:.2f},{b['hi']:.2f}) has measured precision "
                        f"{b['precision']:.2f} over n={b['n']}"
                    )
                p = b.get("precision")
                return False, (
                    f"band [{b['lo']:.2f},{b['hi']:.2f}) precision "
                    f"{'unmeasured' if p is None else f'{p:.2f}'} over n={b['n']} "
                    f"— below the commit floor"
                )
        return False, f"confidence {confidence:.2f} falls in no calibrated band"

    @property
    def scope_signals(self) -> dict[str, Any]:
        return self.ontology.get("output_scope_signals", {})

    @property
    def output_class_ladder(self) -> list[str]:
        return self.ontology.get(
            "output_class_ladder", ["tiny", "short", "medium", "long", "xlong"]
        )

    @property
    def max_signal_output_class(self) -> str:
        return self.ontology.get("max_signal_output_class", "long")

    @property
    def learning_cfg(self) -> dict[str, Any]:
        return self.policies_cfg["learning"]

    @property
    def meta_models(self) -> dict[str, str]:
        return self.policies_cfg["meta_models"]

    def policy(self, name: str | None) -> dict[str, Any]:
        name = name or self.policies_cfg["default_policy"]
        if name not in self.policies_cfg["policies"]:
            raise KeyError(f"unknown policy {name!r}")
        return {"name": name, **self.policies_cfg["policies"][name]}

    def effective_policy(self, requested: str | None, domain: str) -> dict[str, Any]:
        """Domain floors can override a caller that asked for something cheaper."""
        pol = self.policy(requested)
        floor_name = self.policies_cfg.get("policy_floors_by_domain", {}).get(domain)
        if not floor_name:
            return pol
        floor = self.policy(floor_name)
        if floor["quality_floor"] > pol["quality_floor"]:
            floor["forced_by"] = f"domain:{domain}"
            floor["overrode"] = pol["name"]
            return floor
        return pol

    def extend_capability(self, name: str, *, parent: str | None = None, desc: str = "",
                          persist: bool = True) -> None:
        """Add a capability to the live graph, and to the overlay by default."""
        self.kg.add_capability(name, parent=parent, desc=desc)
        if persist:
            record_extension(self.extensions_path, "capabilities",
                             {"name": name, "parent": parent, "desc": desc})

    def extend_task_type(self, name: str, *, requires: dict[str, float], output_class: str,
                         desc: str = "", persist: bool = True) -> None:
        self.kg.add_task_type(name, requires=requires, output_class=output_class, desc=desc)
        if persist:
            record_extension(self.extensions_path, "task_types",
                             {"name": name, "requires": requires,
                              "output_class": output_class, "desc": desc})

    def baseline(self) -> dict[str, Any]:
        return self.catalog["baseline"]


def build_graph(ontology: dict, catalog: dict, policies: dict) -> KnowledgeGraph:
    kg = KnowledgeGraph()

    # --- Capability nodes, then the SUBSUMES taxonomy over them ---
    for cap, meta in ontology["capabilities"].items():
        kg.add_node(f"cap:{cap}", NodeType.CAPABILITY, name=cap, desc=meta.get("desc", ""))

    tax = ontology.get("capability_taxonomy") or {}
    kg.inheritance_discount = float(tax.get("inheritance_discount", 0.90))
    kg.max_inheritance_hops = int(tax.get("max_hops", 3))
    for child, parent in (tax.get("parents") or {}).items():
        for c in (child, parent):
            if not kg.has(f"cap:{c}"):
                raise ValueError(f"taxonomy references unknown capability {c!r}")
        kg.add_edge(f"cap:{parent}", f"cap:{child}", EdgeType.SUBSUMES)
    for child in (tax.get("parents") or {}):
        if child in kg.ancestors_of(child):
            raise ValueError(f"subsumption cycle involving {child!r}")

    # --- OutputClass nodes ---
    for oc, meta in ontology["output_classes"].items():
        kg.add_node(f"out:{oc}", NodeType.OUTPUT_CLASS, name=oc, **meta)

    # --- TaskType nodes + REQUIRES / PRODUCES edges ---
    for task, meta in ontology["task_types"].items():
        tid = kg.add_node(f"task:{task}", NodeType.TASK_TYPE, name=task, desc=meta.get("desc", ""))
        for cap, lvl in meta["requires"].items():
            if not kg.has(f"cap:{cap}"):
                raise ValueError(f"task {task!r} requires unknown capability {cap!r}")
            # Same two-copy discipline as PROVIDES: `min_level` is what the
            # router reads and what learning may move; `authored_min_level`
            # never changes, so drift is always measurable against intent.
            kg.add_edge(tid, f"cap:{cap}", EdgeType.REQUIRES,
                        min_level=float(lvl), authored_min_level=float(lvl), observations=0)
        kg.add_edge(tid, f"out:{meta['output_class']}", EdgeType.PRODUCES)

    # --- Domain nodes + ELEVATES edges ---
    for dom, meta in ontology["domains"].items():
        did = kg.add_node(f"domain:{dom}", NodeType.DOMAIN, name=dom)
        for cap, delta in (meta.get("elevates") or {}).items():
            kg.add_edge(did, f"cap:{cap}", EdgeType.ELEVATES, delta=float(delta))

    # --- Signal nodes + ELEVATES edges ---
    for sig, meta in ontology["complexity_signals"].items():
        sid = kg.add_node(
            f"signal:{sig}", NodeType.SIGNAL, name=sig, patterns=meta.get("patterns") or []
        )
        for cap, delta in (meta.get("elevates") or {}).items():
            kg.add_edge(sid, f"cap:{cap}", EdgeType.ELEVATES, delta=float(delta))

    # --- SLO nodes ---
    for slo, meta in ontology["latency_slos"].items():
        kg.add_node(f"slo:{slo}", NodeType.SLO, name=slo, **meta)

    # --- Model nodes + PROVIDES edges ---
    for m in catalog["models"]:
        spec = ModelSpec(
            id=m["id"],
            display_name=m["display_name"],
            rung=int(m["rung"]),
            context_window=int(m["context_window"]),
            max_output_tokens=int(m["max_output_tokens"]),
            pricing=m["pricing"],
            nfr=m.get("nfr", {}),
            config_surface=m.get("config_surface", {}),
            provides=dict(m["provides"]),
        )
        kg.add_node(spec.id, NodeType.MODEL, name=spec.display_name, spec=spec)
        for cap, lvl in spec.provides.items():
            if not kg.has(f"cap:{cap}"):
                raise ValueError(f"model {spec.id} provides unknown capability {cap!r}")
            # `level` is mutable (learning writes here); `seed_level` never moves.
            kg.add_edge(
                spec.id, f"cap:{cap}", EdgeType.PROVIDES,
                level=float(lvl), seed_level=float(lvl), observations=0,
            )

    # --- SIMILAR_TO: derived, not authored ---
    #
    # Similarity is computed from what two tasks actually demand rather than
    # from a hand-written list, so it stays correct when a task's requirements
    # change and cannot drift out of sync with the ontology.
    sim_cfg = (policies.get("learning") or {}).get("similarity") or {}
    min_sim = float(sim_cfg.get("min_similarity", 0.35))
    tasks = [n.removeprefix("task:") for n in kg.nodes_of(NodeType.TASK_TYPE)]
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            sim = capability_similarity(kg.requirements_of(a), kg.requirements_of(b))
            if sim >= min_sim:
                # Symmetric relation, stored as two directed edges so a walk
                # from either end finds it.
                kg.add_edge(f"task:{a}", f"task:{b}", EdgeType.SIMILAR_TO, similarity=sim)
                kg.add_edge(f"task:{b}", f"task:{a}", EdgeType.SIMILAR_TO, similarity=sim)

    # --- ESCALATES_TO chain ---
    ladder = catalog["ladder"]
    for lo, hi in zip(ladder, ladder[1:]):
        kg.add_edge(lo, hi, EdgeType.ESCALATES_TO, for_task=None)

    # Task-specific paths turn the chain into a lattice. Edges carry the task
    # they apply to, so `ladder(task)` resolves a different route through the
    # same node set.
    for task, path in (catalog.get("escalation_paths") or {}).items():
        if not kg.has(f"task:{task}"):
            raise ValueError(f"escalation path for unknown task {task!r}")
        unknown = [m for m in path if not kg.has(m)]
        if unknown:
            raise ValueError(f"escalation path {task!r} names unknown models: {unknown}")
        rungs = [kg.model(m).rung for m in path]
        if rungs != sorted(rungs):
            raise ValueError(f"escalation path {task!r} is not monotonically increasing")
        kg.attrs(f"task:{task}")["escalation_path"] = list(path)
        for lo, hi in zip(path, path[1:]):
            kg.add_edge(lo, hi, EdgeType.ESCALATES_TO, qualifier=task, for_task=task)

    # --- Policy nodes ---
    for name, meta in policies["policies"].items():
        kg.add_node(f"policy:{name}", NodeType.POLICY, name=name, **meta)
    for dom, pol in (policies.get("policy_floors_by_domain") or {}).items():
        kg.add_edge(f"policy:{pol}", f"domain:{dom}", EdgeType.GOVERNS, kind="floor")

    return kg


def capability_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Weighted Jaccard over two tasks' capability requirements.

    Presence alone is too coarse: `chitchat` and `math_proof` both require
    instruction_following, at 0.40 and 0.85. Weighting by the demanded level
    means two tasks count as similar when they are hard in the same places,
    which is what makes borrowed evidence meaningful.
    """
    caps = set(a) | set(b)
    if not caps:
        return 0.0
    inter = sum(min(a.get(c, 0.0), b.get(c, 0.0)) for c in caps)
    union = sum(max(a.get(c, 0.0), b.get(c, 0.0)) for c in caps)
    return round(inter / union, 4) if union else 0.0


def apply_extensions(kg: KnowledgeGraph, path: Path) -> list[str]:
    """Replay an extensions overlay onto a freshly built graph.

    Order matters: capabilities before the task types that require them.
    A failure here is fatal by design — silently skipping a bad extension
    would route traffic against an ontology nobody has actually seen.
    """
    if not path.is_file():
        return []
    data = _load(path) or {}
    applied: list[str] = []
    for cap in data.get("capabilities") or []:
        kg.add_capability(cap["name"], parent=cap.get("parent"), desc=cap.get("desc", ""))
        applied.append(f"capability:{cap['name']}")
    for task in data.get("task_types") or []:
        kg.add_task_type(task["name"], requires=task["requires"],
                         output_class=task["output_class"], desc=task.get("desc", ""))
        applied.append(f"task_type:{task['name']}")
    return applied


def record_extension(path: Path, kind: str, entry: dict[str, Any]) -> None:
    """Append one extension to the overlay so it survives a restart."""
    data = _load(path) if path.is_file() else {}
    data = data or {}
    bucket = data.setdefault(kind, [])
    if any(e.get("name") == entry["name"] for e in bucket):
        raise ValueError(f"{kind[:-1]} {entry['name']!r} is already in the overlay")
    bucket.append(entry)
    with path.open("w") as f:
        f.write("# GENERATED by runtime extension. Reviewable, and revertible by\n"
                "# deleting an entry. The hand-authored ontology lives in graph.yaml.\n")
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def effective_pricing(spec: ModelSpec, on: date | None = None) -> tuple[float, float]:
    """(input $/MTok, output $/MTok), honouring time-boxed introductory rates."""
    p = spec.pricing
    inp, out = float(p["input_per_mtok"]), float(p["output_per_mtok"])
    until = p.get("intro_until")
    if until is not None and "intro_input_per_mtok" in p:
        if isinstance(until, str):
            until = date.fromisoformat(until)
        if (on or date.today()) <= until:
            inp = float(p["intro_input_per_mtok"])
            out = float(p["intro_output_per_mtok"])
    return inp, out
