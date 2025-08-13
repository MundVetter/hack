from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

from .analyzer import AnalysisResult, analyze_path
from .graph import write_graph
from .utils import cleanup_dir, clone_repo_shallow
from .llm_detect import detect_vulnerabilities_with_openai

app = typer.Typer(add_completion=False, help="Build dataflow graph and flag potential vulnerabilities in a codebase")


def _explain_with_openai(findings_json: str, api_key: Optional[str]) -> Optional[str]:
    try:
        from openai import OpenAI
    except Exception:
        return None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a security assistant. Given these static analysis findings in JSON, group similar issues, "
        "rate severity (low/med/high) and briefly explain fix guidance. Keep it concise.\n\n" + findings_json
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise, actionable security summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


@app.command()
def analyze(
    path: Optional[str] = typer.Option(None, help="Path to local repository to analyze"),
    repo: Optional[str] = typer.Option(None, help="GitHub repo URL to shallow clone and analyze"),
    graph_out: Optional[str] = typer.Option(None, help="Path to write graph (.graphml or .dot or .gexf)"),
    json_out: Optional[str] = typer.Option(None, help="Write findings to this JSON file"),
    json_print: bool = typer.Option(False, "--json", help="Print findings as JSON"),
    openai_explain: bool = typer.Option(False, help="Summarize findings with OpenAI if API key set"),
    openai_api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY", help="OpenAI API key"),
    use_modal: bool = typer.Option(False, "--modal", help="Run analysis remotely on Modal"),
    openai_detect: bool = typer.Option(False, help="Use OpenAI to do an additional pass to detect vulnerabilities"),
    show_graph: bool = typer.Option(False, help="Render a compact ASCII view of the graph in the terminal"),
):
    """Analyze a codebase, build a dataflow graph, and report potential vulnerabilities."""
    if not path and not repo:
        rprint("[bold red]Error:[/] Provide --path or --repo")
        raise typer.Exit(code=2)

    tmp_dir = None
    target = path
    if repo:
        try:
            tmp_dir = clone_repo_shallow(repo)
            target = tmp_dir
        except Exception as e:
            rprint(f"[bold red]Clone failed:[/] {e}")
            raise typer.Exit(code=1)

    if use_modal:
        try:
            import modal
            from .modal_app import analyze_remote, app as modal_app
        except Exception as e:
            rprint(f"[bold red]Modal not available:[/] {e}")
            if tmp_dir:
                cleanup_dir(tmp_dir)
            raise typer.Exit(code=1)
        r = analyze_remote.call(repo_url=repo, path=path)
        try:
            payload = json.loads(r)
        except Exception:
            payload = {"findings": []}
        findings = payload.get("findings", [])
        if graph_out:
            rprint("[yellow]Graph export not supported in remote mode; run locally for graph output.[/]")
        _print_findings(findings, json_print, json_out)
        if openai_explain:
            msg = _explain_with_openai(json.dumps(findings, indent=2), openai_api_key)
            if msg:
                rprint("\n[bold]LLM summary:[/]")
                rprint(msg)
        if tmp_dir:
            cleanup_dir(tmp_dir)
        raise typer.Exit(0)

    assert target is not None
    result = analyze_path(target)

    if graph_out:
        try:
            write_graph(result.graph, graph_out)
            rprint(f"[green]Graph written:[/] {graph_out}")
        except Exception as e:
            rprint(f"[bold red]Graph export failed:[/] {e}")

    findings = [
        {
            "kind": f.kind,
            "file": f.file,
            "line": f.line,
            "message": f.message,
            "code": f.code,
        }
        for f in result.findings
    ]

    if openai_detect:
        llm_findings = detect_vulnerabilities_with_openai(target, api_key=openai_api_key)
        if llm_findings:
            rprint("[yellow]OpenAI additional findings added.[/]")
            findings.extend(llm_findings)

    _print_findings(findings, json_print, json_out)

    if show_graph:
        _print_ascii_graph(result)

    if openai_explain:
        msg = _explain_with_openai(json.dumps(findings, indent=2), openai_api_key)
        if msg:
            rprint("\n[bold]LLM summary:[/]")
            rprint(msg)

    if tmp_dir:
        cleanup_dir(tmp_dir)


def _print_findings(findings: list[dict], json_print: bool, json_out: Optional[str]):
    if json_out:
        Path(json_out).write_text(json.dumps(findings, indent=2))
        rprint(f"[green]Findings JSON written:[/] {json_out}")
    if json_print or not findings:
        rprint(json.dumps(findings, indent=2))
        return
    table = Table(title="Potential vulnerabilities")
    table.add_column("Kind", style="cyan")
    table.add_column("Location", style="magenta")
    table.add_column("Message", style="white")
    for f in findings:
        loc = f"{f['file']}:{f['line']}"
        msg = (f.get("message") or "").strip().splitlines()[0][:120]
        table.add_row(f.get("kind", ""), loc, msg)
    rprint(table)


def _print_ascii_graph(result: AnalysisResult):
    from collections import defaultdict

    g = result.graph
    by_kind = defaultdict(list)
    for node_id, data in g.nodes(data=True):
        by_kind[data.get("kind", "var")].append((node_id, data))

    console = Console()
    sections = []
    for kind in ("source", "sink", "call", "var"):
        nodes = by_kind.get(kind, [])
        if not nodes:
            continue
        lines = []
        for node_id, data in nodes[:50]:
            label = data.get("label", node_id)
            loc = f"{data.get('file','')}:{data.get('line','')}"
            lines.append(f"- {label} ({loc})")
        sections.append(Panel("\n".join(lines), title=f"{kind.upper()} ({len(nodes)})"))
    if sections:
        console.print(*sections)

    # Show a few taint edges
    edge_lines = []
    for i, (u, v, ed) in enumerate(g.edges(data=True)):
        if i >= 100:
            break
        edge_lines.append(f"{u} -> {v} [{ed.get('kind','')}]")
    if edge_lines:
        console.print(Panel("\n".join(edge_lines), title="EDGES (sample)"))


def main():  # entry point
    app()