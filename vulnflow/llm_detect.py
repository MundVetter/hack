from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any


def _call_openai(messages: list[dict], api_key: Optional[str], model: str) -> Optional[str]:
    try:
        from openai import OpenAI
    except Exception:
        return None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def detect_vulnerabilities_with_openai(
    root_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_files: int = 100,
    max_chars_per_file: int = 12000,
) -> List[Dict[str, Any]]:
    """Return additional findings as a list of dicts: kind, file, line, message, code."""
    findings: List[Dict[str, Any]] = []
    files = [p for p in Path(root_path).rglob("*.py")][:max_files]
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file]
        system = {
            "role": "system",
            "content": (
                "You are a security static analysis assistant. Given a Python file, identify likely vulnerabilities. "
                "Focus on SQL injection, command injection, path traversal, SSRF, unsafe deserialization, secrets exposure, and dangerous LLM usage. "
                "Return a JSON array of objects with keys: kind (one of sql, subprocess, fs, http, llm, secret, deserialization, crypto), line (int best guess), message (concise), code (short snippet). "
                "Do not include any commentary outside JSON."
            ),
        }
        user = {
            "role": "user",
            "content": f"File: {file}\n\n```python\n{text}\n```\n\nJSON only, array of findings.",
        }
        content = _call_openai([system, user], api_key, model)
        if not content:
            continue
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for f in parsed:
                    if not isinstance(f, dict):
                        continue
                    findings.append(
                        {
                            "kind": str(f.get("kind", "llm")),
                            "file": str(file),
                            "line": int(f.get("line") or 0),
                            "message": str(f.get("message", "LLM detected issue")),
                            "code": str(f.get("code", "")),
                        }
                    )
        except Exception:
            # Best-effort: try to extract JSON between first [ and last ]
            try:
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(content[start : end + 1])
                    if isinstance(parsed, list):
                        for f in parsed:
                            if not isinstance(f, dict):
                                continue
                            findings.append(
                                {
                                    "kind": str(f.get("kind", "llm")),
                                    "file": str(file),
                                    "line": int(f.get("line") or 0),
                                    "message": str(f.get("message", "LLM detected issue")),
                                    "code": str(f.get("code", "")),
                                }
                            )
            except Exception:
                continue
    return findings