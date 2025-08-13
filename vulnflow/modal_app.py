from __future__ import annotations

import json
import sys
import modal

# Build an image that has git and Python deps
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "networkx>=3.3",
    )
)

app = modal.App("vulnflow")


@app.function(
    image=image,
    mounts=[modal.Mount.from_local_dir("/workspace", remote_path="/workspace")],
)
def analyze_remote(repo_url: str | None = None, path: str | None = None) -> str:
    # Ensure our local package is importable in the remote worker
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")

    from vulnflow.analyzer import analyze_path
    from vulnflow.utils import clone_repo_shallow, cleanup_dir

    tmp_dir = None
    try:
        if repo_url:
            tmp_dir = clone_repo_shallow(repo_url)
            target = tmp_dir
        elif path:
            target = path
        else:
            raise ValueError("repo_url or path required")
        result = analyze_path(target)
        payload = {
            "findings": [
                {
                    "kind": f.kind,
                    "file": f.file,
                    "line": f.line,
                    "message": f.message,
                    "code": f.code,
                }
                for f in result.findings
            ]
        }
        return json.dumps(payload)
    finally:
        if tmp_dir:
            cleanup_dir(tmp_dir)