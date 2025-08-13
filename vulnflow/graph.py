from __future__ import annotations

import networkx as nx
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GraphNode:
    id: str
    label: str
    file: str
    line: int
    kind: str  # source | sink | var | call


def new_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.graph["name"] = "vulnflow"
    return g


def add_node(g: nx.DiGraph, node: GraphNode) -> None:
    if node.id not in g:
        g.add_node(node.id, label=node.label, file=node.file, line=node.line, kind=node.kind)


def add_edge(g: nx.DiGraph, src_id: str, dst_id: str, kind: str) -> None:
    g.add_edge(src_id, dst_id, kind=kind)


def write_graph(g: nx.DiGraph, path: str) -> None:
    if path.endswith(".graphml"):
        nx.write_graphml(g, path)
    elif path.endswith(".dot"):
        try:
            import pydot  # noqa: F401
            from networkx.drawing.nx_pydot import write_dot
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Writing DOT requires pydot installed") from e
        write_dot(g, path)
    else:
        nx.write_gexf(g, path)