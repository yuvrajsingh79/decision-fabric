"""The knowledge graph itself: a NetworkX MultiDiGraph plus typed accessors.

The store is deliberately behind a thin interface (`KnowledgeGraph`) so a Neo4j
or RDF backend can replace it without touching the reasoner. Everything the
reasoner needs is expressed as edge traversals, not table lookups.
"""
from __future__ import annotations

from typing import Any, Iterator

import networkx as nx

from .ontology import EdgeType, ModelSpec, NodeType


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ---------- writes ----------

    def add_node(self, node_id: str, ntype: NodeType, **attrs: Any) -> str:
        self.g.add_node(node_id, ntype=ntype.value, **attrs)
        return node_id

    def add_edge(self, src: str, dst: str, etype: EdgeType, **attrs: Any) -> None:
        self.g.add_edge(src, dst, key=etype.value, etype=etype.value, **attrs)

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
        """Yield (target, edge_attrs) for one edge type."""
        if node_id not in self.g:
            return
        for _, dst, key, data in self.g.out_edges(node_id, keys=True, data=True):
            if key == etype.value:
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

    def provided_level(self, model_id: str, capability: str) -> float:
        """Learning-adjusted level. For reporting and export, not for gating."""
        e = self.edge(model_id, f"cap:{capability}", EdgeType.PROVIDES)
        return float(e["level"]) if e else 0.0

    def seed_level(self, model_id: str, capability: str) -> float:
        """Immutable seed prior. This is what eligibility is judged against."""
        e = self.edge(model_id, f"cap:{capability}", EdgeType.PROVIDES)
        return float(e["seed_level"]) if e else 0.0

    def requirements_of(self, task_type: str) -> dict[str, float]:
        return {
            dst.removeprefix("cap:"): float(d["min_level"])
            for dst, d in self.out_edges(f"task:{task_type}", EdgeType.REQUIRES)
        }

    def elevations_of(self, node_id: str) -> dict[str, float]:
        return {
            dst.removeprefix("cap:"): float(d["delta"])
            for dst, d in self.out_edges(node_id, EdgeType.ELEVATES)
        }

    def ladder(self) -> list[str]:
        """Escalation order derived by walking ESCALATES_TO from the bottom rung."""
        models = self.models()
        if not models:
            return []
        cur = models[0].id
        order = [cur]
        while True:
            nxt = next((d for d, _ in self.out_edges(cur, EdgeType.ESCALATES_TO)), None)
            if nxt is None or nxt in order:
                break
            order.append(nxt)
            cur = nxt
        return order

    # ---------- introspection ----------

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            counts[d.get("ntype", "?")] = counts.get(d.get("ntype", "?"), 0) + 1
        for _, _, k in self.g.edges(keys=True):
            counts[k] = counts.get(k, 0) + 1
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
