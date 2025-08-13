from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .graph import GraphNode, add_edge, add_node, new_graph
from .rules import Sink, Source, is_sink, is_source, string_building_is_dynamic


@dataclass
class Finding:
    kind: str
    file: str
    line: int
    message: str
    code: str
    path_nodes: List[str] = field(default_factory=list)
    potential: bool = True


@dataclass
class AnalysisResult:
    graph: nx.DiGraph
    findings: List[Finding]


class TaintTracker(ast.NodeVisitor):
    def __init__(self, file_path: str, graph: nx.DiGraph) -> None:
        self.file_path = file_path
        self.graph = graph
        self.tainted: Set[str] = set()
        self.var_sources: Dict[str, Set[str]] = {}
        self.findings: List[Finding] = []
        self.current_function: Optional[str] = None

    def _node_id(self, label: str, line: int, kind: str) -> str:
        return f"{self.file_path}:{line}:{label}:{kind}"

    def _mark_tainted(self, var: str, src_node_ids: Set[str]) -> None:
        self.tainted.add(var)
        self.var_sources.setdefault(var, set()).update(src_node_ids)

    def _expr_is_tainted(self, node: ast.AST) -> Tuple[bool, Set[str]]:
        # Evaluate if expression is tainted, and collect origin nodes
        if isinstance(node, ast.Name):
            if node.id in self.tainted:
                return True, self.var_sources.get(node.id, set()).copy()
            return False, set()
        if isinstance(node, ast.Call):
            src = is_source(node)
            if src:
                src_id = self._record_source(node, src)
                return True, {src_id}
            # Otherwise, taint if any arg tainted
            origins: Set[str] = set()
            tainted = False
            for arg in list(getattr(node, "args", [])) + [kw.value for kw in getattr(node, "keywords", [])]:
                t, orig = self._expr_is_tainted(arg)
                if t:
                    tainted = True
                    origins.update(orig)
            return tainted, origins
        if isinstance(node, ast.BinOp):
            l_t, l_o = self._expr_is_tainted(node.left)
            r_t, r_o = self._expr_is_tainted(node.right)
            return (l_t or r_t), (l_o | r_o)
        if isinstance(node, ast.JoinedStr):
            origins: Set[str] = set()
            tainted = False
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    t, o = self._expr_is_tainted(v.value)
                    if t:
                        tainted = True
                        origins.update(o)
            return tainted, origins
        if isinstance(node, ast.Attribute):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.Subscript):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.Dict):
            origins: Set[str] = set()
            tainted = False
            for v in node.values:
                t, o = self._expr_is_tainted(v)
                if t:
                    tainted = True
                    origins.update(o)
            return tainted, origins
        if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            origins: Set[str] = set()
            tainted = False
            for v in node.elts:
                t, o = self._expr_is_tainted(v)
                if t:
                    tainted = True
                    origins.update(o)
            return tainted, origins
        return False, set()

    def _record_source(self, node: ast.Call, src: Source) -> str:
        line = getattr(node, "lineno", 0)
        node_id = self._node_id(src.kind, line, "source")
        add_node(self.graph, GraphNode(id=node_id, label=src.description, file=self.file_path, line=line, kind="source"))
        return node_id

    def _record_sink(self, node: ast.Call, sink: Sink, origins: Set[str], detail: str) -> None:
        line = getattr(node, "lineno", 0)
        sink_id = self._node_id(sink.kind, line, "sink")
        add_node(self.graph, GraphNode(id=sink_id, label=sink.description, file=self.file_path, line=line, kind="sink"))
        for o in origins:
            add_edge(self.graph, o, sink_id, kind="taint")
        code = self._get_source_snippet(line)
        self.findings.append(
            Finding(
                kind=sink.kind,
                file=self.file_path,
                line=line,
                message=detail or sink.description,
                code=code,
                path_nodes=[*origins, sink_id],
                potential=True,
            )
        )

    def _get_source_snippet(self, line: int, context: int = 2) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            start = max(1, line - context)
            end = min(len(lines), line + context)
            snippet = "".join(lines[start - 1 : end])
            return textwrap.dedent(snippet)
        except Exception:
            return ""

    # Visitors

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self.current_function
        self.current_function = node.name
        # Reset taint for function scope but keep findings/graph
        saved_tainted = self.tainted.copy()
        saved_sources = {k: v.copy() for k, v in self.var_sources.items()}
        self.tainted = set()
        self.var_sources = {}
        self.generic_visit(node)
        # Restore outer scope
        self.tainted = saved_tainted
        self.var_sources = saved_sources
        self.current_function = prev

    def visit_Assign(self, node: ast.Assign) -> None:
        t, origins = self._expr_is_tainted(node.value)
        direct_source = None
        if isinstance(node.value, ast.Call):
            s = is_source(node.value)
            if s:
                src_id = self._record_source(node.value, s)
                direct_source = {src_id}
                t = True
                origins = direct_source
        if t:
            for target in node.targets:
                for name in self._iter_assigned_names(target):
                    self._mark_tainted(name, origins)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        t, origins = self._expr_is_tainted(node.value)
        if t:
            for name in self._iter_assigned_names(node.target):
                self._mark_tainted(name, origins)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        s = is_source(node)
        if s:
            self._record_source(node, s)
        sink = is_sink(node)
        if sink:
            # If any arg tainted, record finding
            tainted = False
            origins: Set[str] = set()
            for arg in list(getattr(node, "args", [])) + [kw.value for kw in getattr(node, "keywords", [])]:
                t, o = self._expr_is_tainted(arg)
                if t:
                    tainted = True
                    origins.update(o)
            detail = ""
            if sink.kind == "sql":
                # If query is dynamically built, increase confidence
                if node.args:
                    if string_building_is_dynamic(node.args[0]):
                        detail = "Potential SQL injection: dynamic query building with tainted input"
            if sink.kind == "subprocess":
                detail = "Potential command injection: tainted input reaches subprocess call"
            if sink.kind == "llm":
                detail = "Tainted input passed to LLM call (data exfiltration risk)"
            if sink.kind == "fs":
                detail = "Tainted path used in file operation (path traversal risk)"
            if tainted:
                self._record_sink(node, sink, origins, detail)
        self.generic_visit(node)

    def _iter_assigned_names(self, target: ast.AST):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from self._iter_assigned_names(elt)
        elif isinstance(target, ast.Attribute):
            # consider attribute assignments as variable name for local taint purposes
            yield self._attr_to_str(target)

    def _attr_to_str(self, node: ast.Attribute) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


def analyze_path(path: str) -> AnalysisResult:
    graph = new_graph()
    findings: List[Finding] = []
    for file in Path(path).rglob("*.py"):
        try:
            src = file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        tracker = TaintTracker(str(file), graph)
        tracker.visit(tree)
        findings.extend(tracker.findings)
    return AnalysisResult(graph=graph, findings=findings)