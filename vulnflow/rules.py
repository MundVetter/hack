from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Source:
    kind: str  # e.g., request_args, env, input, cli
    description: str


@dataclass(frozen=True)
class Sink:
    kind: str  # e.g., sql, llm, subprocess, fs
    description: str


def _attr_chain(node: ast.AST) -> Tuple[str, ...]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        parts.append(node.func.id)
    return tuple(reversed(parts))


def is_source(node: ast.AST) -> Optional[Source]:
    # Call-based sources
    if isinstance(node, ast.Call):
        # input()
        if isinstance(node.func, ast.Name) and node.func.id == "input":
            return Source("stdin", "Built-in input()")
        # os.getenv / environ.get
        if isinstance(node.func, ast.Attribute):
            chain = _attr_chain(node.func)
            if chain in [("os", "getenv")]:
                return Source("env", "Environment variable via os.getenv")
            if len(chain) >= 3 and chain[-3:] in [("os", "environ", "get")]:
                return Source("env", "Environment variable via os.environ.get")
        # flask/django request.*.get
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            chain = _attr_chain(node.func.value)
            # request.args.get / request.form.get / request.values.get / request.headers.get
            if len(chain) >= 2 and chain[-2:] in [
                ("request", "args"),
                ("request", "form"),
                ("request", "values"),
                ("request", "headers"),
            ]:
                return Source("http_request", f"HTTP user input via {'.'.join(chain)}.get")
    # Name-based conventional sources (function params often considered sources in handlers)
    return None


def is_sink(call: ast.Call) -> Optional[Sink]:
    # SQL: cursor.execute / connection.execute / session.execute / text()
    if isinstance(call.func, ast.Attribute):
        chain = _attr_chain(call.func)
        if chain and chain[-1] in ("execute", "executemany"):
            return Sink("sql", f"Database execution via {'.'.join(chain)}")
        # subprocess.* with shell or command string
        if len(chain) >= 2 and chain[:1] == ("subprocess",):
            return Sink("subprocess", f"Subprocess call via {'.'.join(chain)}")
        # requests to LLM-like endpoints (heuristic)
        if len(chain) >= 2 and chain[:1] == ("requests",):
            return Sink("http", f"HTTP request via {'.'.join(chain)}")
        # open() for path traversal risk
        if chain in [("open",)]:
            return Sink("fs", "File open")
    # openai LLM calls new/old SDKs
    if isinstance(call.func, ast.Attribute):
        chain = _attr_chain(call.func)
        # openai.ChatCompletion.create / client.chat.completions.create
        if (
            (len(chain) >= 3 and chain[-3:] == ("ChatCompletion", "create") and chain[0] == "openai")
            or (len(chain) >= 4 and chain[-4:] == ("chat", "completions", "create"))
        ):
            return Sink("llm", f"LLM call via {'.'.join(chain)}")
    if isinstance(call.func, ast.Name) and call.func.id == "open":
        return Sink("fs", "File open")
    return None


def string_building_is_dynamic(node: ast.AST) -> bool:
    # Detect string building that likely incorporates variables
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        # str.format / f"" already covered by JoinedStr, '%' via BinOp Mod
        if node.func.attr == "format":
            return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return True
    return False