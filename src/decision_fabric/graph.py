"""The knowledge graph itself: a NetworkX MultiDiGraph plus typed accessors.

The store is deliberately behind a thin interface (`KnowledgeGraph`) so a Neo4j
or RDF backend can replace it without touching the reasoner. Everything the
reasoner needs is expressed as edge traversals, not table lookups.
"""
from __future__ import annotations

from typing import Any, Iterator

import networkx as nx

from .ontology import EdgeType, ModelSpec, NodeType, ResolvedLevel


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        # Populated from config/graph.yaml -> capability_taxonomy.
        self.inheritance_discount = 0.90
        self.max_inheritance_hops = 3

    # ---------- writes ----------

    def add_node(self, node_id: str, ntype: NodeType, **attrs: Any) -> str:
        self.g.add_node(node_id, ntype=ntype.value, **attrs)
        return node_id

    def add_edge(self, src: str, dst: str, etype: EdgeType,
                 qualifier: str | None = None, **attrs: Any) -> None:
        """Add a typed edge.

        `qualifier` distinguishes parallel edges of the SAME type between the
        same pair — needed for the escalation lattice, where a default
        haiku->sonnet hop and a task-specific one must coexist. Without it the
        second silently replaced the first, because the edge key was the type
        alone.
        """
        key = etype.value if qualifier is None else f"{etype.value}#{qualifier}"
        self.g.add_edge(src, dst, key=key, etype=etype.value, **attrs)

    # ---------- reads ----------

    def nodes_of(self, ntype: NodeType) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("ntype") == ntype.value]

    def attrs(self, node_id: str) -> dict[str, Any]:
        if node_id not in self.g:
            raise KeyError(f"no such node: {node_id!r}")
        return self.g.nodes[node_id]

    def has(self, node_id: str) -> bool:
        return node_id in self.g

    def out_edges(self, node_id: str, etype: EdgeType) -> Iterator[tuple[str, dict]]:
        """Yield (target, edge_attrs) for one edge type, including qualified
        parallel edges — matching on the stored `etype` rather than the key."""
        if node_id not in self.g:
            return
        for _, dst, _key, data in self.g.out_edges(node_id, keys=True, data=True):
            if data.get("etype") == etype.value:
                yield dst, data

    def edge(self, src: str, dst: str, etype: EdgeType) -> dict[str, Any] | None:
        try:
            return self.g.edges[src, dst, etype.value]
        except KeyError:
            return None

    def set_edge_attr(self, src: str, dst: str, etype: EdgeType, **attrs: Any) -> None:
        e = self.edge(src, dst, etype)
        if e is None:
            raise KeyError(f"no {etype.value} edge {src} -> {dst}")
        e.update(attrs)

    # ---------- typed convenience ----------

    def model(self, model_id: str) -> ModelSpec:
        a = self.attrs(model_id)
        if a.get("ntype") != NodeType.MODEL.value:
            raise TypeError(f"{model_id!r} is not a Model node")
        return a["spec"]

    def models(self) -> list[ModelSpec]:
        specs = [self.attrs(n)["spec"] for n in self.nodes_of(NodeType.MODEL)]
        return sorted(specs, key=lambda s: s.rung)

    # ---------- capability resolution (subsumption) ----------

    def ancestors_of(self, capability: str) -> list[str]:
        """Capabilities that subsume this one, nearest first.

        Walks SUBSUMES edges upward. Cycle-guarded, and bounded by
        `max_inheritance_hops` so a deep or accidentally-recursive taxonomy
        cannot make resolution unbounded.
        """
        chain: list[str] = []
        seen = {capability}
        cur = capability
        for _ in range(self.max_inheritance_hops):
            parent = next(
                (src.removeprefix("cap:")
                 for src, _, _k, d in self.g.in_edges(f"cap:{cur}", keys=True, data=True)
                 if d.get("etype") == EdgeType.SUBSUMES.value),
                None,
            )
            if parent is None or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            cur = parent
        return chain

    def resolve_level(self, model_id: str, capability: str, *, seed: bool = True) -> ResolvedLevel:
        """A model's level for a capability, inheriting from ancestors if undeclared.

        This is the multi-hop step. A model that declares `deep_reasoning` but
        has never been scored on `math_symbolic` is presumed competent at it,
        discounted once per hop -- because a capability added to the ontology
        tomorrow must not silently zero every model in the catalog.
        """
        attr = "seed_level" if seed else "level"
        e = self.edge(model_id, f"cap:{capability}", EdgeType.PROVIDES)
        if e is not None:
            return ResolvedLevel(float(e[attr]), "declared")

        for hops, ancestor in enumerate(self.ancestors_of(capability), start=1):
            ae = self.edge(model_id, f"cap:{ancestor}", EdgeType.PROVIDES)
            if ae is None:
                continue
            discount = self.inheritance_discount ** hops
            return ResolvedLevel(
                round(float(ae[attr]) * discount, 4),
                "inherited", via=ancestor, hops=hops, discount=round(discount, 4),
            )
        return ResolvedLevel(0.0, "unknown")

    def provided_level(self, model_id: str, capability: str) -> float:
        """Learning-adjusted level, with inheritance. Reporting and scoring."""
        return self.resolve_level(model_id, capability, seed=False).level

    def seed_level(self, model_id: str, capability: str) -> float:
        """Immutable seed prior, with inheritance. Eligibility is judged on this."""
        return self.resolve_level(model_id, capability, seed=True).level

    def requirements_of(self, task_type: str) -> dict[str, float]:
        return {
            dst.removeprefix("cap:"): float(d["min_level"])
            for dst, d in self.out_edges(f"task:{task_type}", EdgeType.REQUIRES)
        }

    def authored_requirement(self, task_type: str, capability: str) -> float:
        """The requirement as written in config, before any learning."""
        e = self.edge(f"task:{task_type}", f"cap:{capability}", EdgeType.REQUIRES)
        return float(e["authored_min_level"]) if e else 0.0

    def elevations_of(self, node_id: str) -> dict[str, float]:
        return {
            dst.removeprefix("cap:"): float(d["delta"])
            for dst, d in self.out_edges(node_id, EdgeType.ELEVATES)
        }

    def ladder(self, task_type: str | None = None) -> list[str]:
        """Escalation order, resolved by walking ESCALATES_TO.

        With a task type, edges tagged for that task win over the default chain,
        so the same node set yields a different route — a lattice rather than a
        single line. Falls back to the default chain wherever no task-specific
        edge exists.
        """
        models = self.models()
        if not models:
            return []

        # A declared path is authoritative and TERMINATES. Deriving the walk
        # from tagged edges alone cannot express that: a two-model path leaves
        # the second model with no tagged outgoing edge, and the walk would
        # fall through to the default chain and escalate past the path's end.
        # The edges still exist so the lattice is queryable and drawable; the
        # node attribute is what stops the walk.
        if task_type and self.has(f"task:{task_type}"):
            declared = self.attrs(f"task:{task_type}").get("escalation_path")
            if declared:
                return list(declared)

        cur = models[0].id
        order = [cur]
        while True:
            outs = list(self.out_edges(cur, EdgeType.ESCALATES_TO))
            nxt = next((d for d, a in outs if a.get("for_task") is None), None)
            if nxt is None or nxt in order:
                break
            order.append(nxt)
            cur = nxt
        return order

    # ---------- open-world extension ----------
    #
    # The graph is authored from YAML at boot, but it must also be extensible at
    # runtime: a new capability, task type or model should be addable without a
    # restart and without a code change. Every method here validates first and
    # raises rather than leaving the graph half-modified -- a partially applied
    # extension is worse than a rejected one, because it routes.

    def add_capability(self, name: str, *, parent: str | None = None, desc: str = "") -> str:
        node = f"cap:{name}"
        if self.has(node):
            raise ValueError(f"capability {name!r} already exists")
        if parent is not None and not self.has(f"cap:{parent}"):
            raise ValueError(f"parent capability {parent!r} does not exist")
        self.add_node(node, NodeType.CAPABILITY, name=name, desc=desc)
        if parent is not None:
            self.add_edge(f"cap:{parent}", node, EdgeType.SUBSUMES)
            # Reject a cycle by checking reachability *after* wiring, then undo.
            if name in self.ancestors_of(parent) or parent == name:
                self.g.remove_edge(f"cap:{parent}", node, EdgeType.SUBSUMES.value)
                self.g.remove_node(node)
                raise ValueError(f"{parent!r} -> {name!r} would create a subsumption cycle")
        return node

    def add_task_type(
        self, name: str, *, requires: dict[str, float], output_class: str, desc: str = ""
    ) -> str:
        node = f"task:{name}"
        if self.has(node):
            raise ValueError(f"task type {name!r} already exists")
        if not requires:
            raise ValueError(f"task type {name!r} must require at least one capability")
        missing = [c for c in requires if not self.has(f"cap:{c}")]
        if missing:
            raise ValueError(f"unknown capabilities for {name!r}: {missing}")
        if not self.has(f"out:{output_class}"):
            raise ValueError(f"unknown output class {output_class!r}")
        bad = {c: lv for c, lv in requires.items() if not 0.0 < float(lv) <= 0.99}
        if bad:
            raise ValueError(f"required levels must be in (0, 0.99]: {bad}")

        self.add_node(node, NodeType.TASK_TYPE, name=name, desc=desc)
        for cap, lv in requires.items():
            self.add_edge(node, f"cap:{cap}", EdgeType.REQUIRES, min_level=float(lv))
        self.add_edge(node, f"out:{output_class}", EdgeType.PRODUCES)
        return node

    def add_model(self, spec: ModelSpec, *, after: str | None = None) -> str:
        """Insert a model, optionally splicing it into the escalation ladder
        directly after `after`."""
        if self.has(spec.id):
            raise ValueError(f"model {spec.id!r} already exists")
        missing = [c for c in spec.provides if not self.has(f"cap:{c}")]
        if missing:
            raise ValueError(f"model {spec.id} declares unknown capabilities: {missing}")
        if after is not None and not self.has(after):
            raise ValueError(f"cannot splice after unknown model {after!r}")

        self.add_node(spec.id, NodeType.MODEL, name=spec.display_name, spec=spec)
        for cap, lv in spec.provides.items():
            self.add_edge(spec.id, f"cap:{cap}", EdgeType.PROVIDES,
                          level=float(lv), seed_level=float(lv), observations=0)
        if after is not None:
            nxt = next((d for d, _ in self.out_edges(after, EdgeType.ESCALATES_TO)), None)
            if nxt is not None:
                self.g.remove_edge(after, nxt, EdgeType.ESCALATES_TO.value)
                self.add_edge(spec.id, nxt, EdgeType.ESCALATES_TO)
            self.add_edge(after, spec.id, EdgeType.ESCALATES_TO)
        return spec.id

    # ---------- evidence ----------

    def upsert_evidence(self, model_id: str, task_type: str, successes: int, trials: int) -> str:
        """Materialise an aggregate Evidence node linking a model to a task.

        One node per (model, task) pair, updated in place — never one node per
        observation. The durable append-only log stays in SQLite where it
        belongs; the graph holds the aggregate, so questions like "what have we
        actually observed about this model?" become traversals instead of
        queries against a table the graph knows nothing about.
        """
        node = f"evidence:{model_id}|{task_type}"
        mean = round(successes / trials, 4) if trials else 0.0
        if self.has(node):
            a = self.attrs(node)
            a.update(successes=successes, trials=trials, mean=mean)
        else:
            self.add_node(node, NodeType.EVIDENCE, model_id=model_id, task_type=task_type,
                          successes=successes, trials=trials, mean=mean)
            self.add_edge(node, model_id, EdgeType.OBSERVED, task_type=task_type)
            if self.has(f"task:{task_type}"):
                self.add_edge(node, f"task:{task_type}", EdgeType.OBSERVED, model_id=model_id)
        for _, data in [(None, self.edge(node, model_id, EdgeType.OBSERVED) or {})]:
            data.update(successes=successes, trials=trials, mean=mean)
        return node

    def evidence_about_model(self, model_id: str) -> list[dict[str, Any]]:
        """Every aggregate observation concerning a model, via OBSERVED edges."""
        out = []
        for src, _, _k, d in self.g.in_edges(model_id, keys=True, data=True):
            if d.get("etype") == EdgeType.OBSERVED.value:
                out.append(dict(self.attrs(src)))
        return sorted(out, key=lambda d: -d.get("trials", 0))

    def evidence_about_task(self, task_type: str) -> list[dict[str, Any]]:
        out = []
        for src, _, _k, d in self.g.in_edges(f"task:{task_type}", keys=True, data=True):
            if d.get("etype") == EdgeType.OBSERVED.value:
                out.append(dict(self.attrs(src)))
        return sorted(out, key=lambda d: -d.get("trials", 0))

    def undeclared_capabilities(self, model_id: str) -> dict[str, ResolvedLevel]:
        """Capabilities this model has no declared level for, and how each
        currently resolves. The operational view of taxonomy coverage."""
        out = {}
        for node in self.nodes_of(NodeType.CAPABILITY):
            cap = node.removeprefix("cap:")
            if self.edge(model_id, node, EdgeType.PROVIDES) is None:
                out[cap] = self.resolve_level(model_id, cap)
        return out

    # ---------- introspection ----------

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            counts[d.get("ntype", "?")] = counts.get(d.get("ntype", "?"), 0) + 1
        for _, _, _k, d in self.g.edges(keys=True, data=True):
            et = d.get("etype", "?")
            counts[et] = counts.get(et, 0) + 1
        counts["_nodes"] = self.g.number_of_nodes()
        counts["_edges"] = self.g.number_of_edges()
        return counts

    def to_dot(self) -> str:
        """Graphviz dump — useful for showing stakeholders the actual fabric."""
        colors = {
            NodeType.CAPABILITY.value: "lightblue",
            NodeType.TASK_TYPE.value: "lightgreen",
            NodeType.MODEL.value: "gold",
            NodeType.DOMAIN.value: "lightpink",
            NodeType.SIGNAL.value: "lavender",
            NodeType.POLICY.value: "wheat",
        }
        lines = ["digraph DecisionFabric {", '  rankdir=LR; node [style=filled shape=box];']
        for n, d in self.g.nodes(data=True):
            c = colors.get(d.get("ntype", ""), "white")
            lines.append(f'  "{n}" [fillcolor={c}];')
        for s, t, k, d in self.g.edges(keys=True, data=True):
            label = k
            if "min_level" in d:
                label += f' {d["min_level"]:.2f}'
            elif "level" in d:
                label += f' {d["level"]:.2f}'
            elif "delta" in d:
                label += f' +{d["delta"]:.2f}'
            lines.append(f'  "{s}" -> "{t}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)
