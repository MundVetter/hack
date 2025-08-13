from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def clone_repo_shallow(repo_url: str, dest_dir: Optional[str] = None) -> str:
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="vulnflow_")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], check=True)
    return dest_dir


def iter_source_files(root: str):
    root_path = Path(root)
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".pyw"}:
            yield str(path)


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def cleanup_dir(path: str) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        pass