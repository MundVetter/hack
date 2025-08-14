#!/usr/bin/env python3
"""
Summarize any Hugging Face dataset as compact JSON for LLM consumption.

Usage:
  python hf_dataset_summary.py --dataset imdb
  python hf_dataset_summary.py --dataset glue --config sst2
  python hf_dataset_summary.py --dataset https://huggingface.co/datasets/allenai/c4 --config en
"""

from __future__ import annotations
import argparse, json, sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# --- deps (give a helpful message if missing) ---
try:
    from datasets import (
        get_dataset_config_names,
        get_dataset_config_info,
        load_dataset,
    )
except Exception as e:
    sys.stderr.write(
        "Missing or incompatible 'datasets' package. Install with:\n"
        "  python -m pip install --upgrade datasets huggingface_hub\n"
    )
    raise

try:
    from huggingface_hub import HfApi
except Exception:
    sys.stderr.write(
        "Missing 'huggingface_hub'. Install with:\n"
        "  python -m pip install --upgrade huggingface_hub\n"
    )
    raise

# ---------- helpers ----------

def _parse_repo_id(name_or_url: str) -> str:
    if name_or_url.startswith("http"):
        path = urlparse(name_or_url).path.strip("/")
        parts = path.split("/")
        # Expect: datasets/<ns>/<name> OR datasets/<name>
        if len(parts) >= 2 and parts[0] == "datasets":
            return "/".join(parts[1:3]) if len(parts) >= 3 else parts[1]
    return name_or_url

def _truncate(val: Any, max_chars: int = 800) -> Any:
    if isinstance(val, str) and len(val) > max_chars:
        return val[:max_chars] + f"... [truncated {len(val)-max_chars} chars]"
    return val

def _normalize_features(features_obj: Any) -> Any:
    # Try converting to plain dict when possible
    try:
        if hasattr(features_obj, "to_dict"):
            return features_obj.to_dict()
        if isinstance(features_obj, dict):
            return features_obj
    except Exception:
        pass
    return str(features_obj) if features_obj is not None else None

def _extract_splits(ci: Any) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Return a dict like {"train": {"num_examples": 67349}, "test": {"num_examples": None}, ...}
    Works across datasets versions where .splits may be:
      - list[SplitInfo] with .name and .num_examples
      - dict[str, SplitInfo]
      - list[str] of split names
      - dict[str, Any] with varying structure
    """
    out: Dict[str, Dict[str, Optional[int]]] = {}
    splits = getattr(ci, "splits", None)
    if not splits:
        return out

    # dict case
    if isinstance(splits, dict):
        for name, v in splits.items():
            num = None
            try:
                num = getattr(v, "num_examples", None)
                if isinstance(v, dict) and num is None:
                    num = v.get("num_examples")
            except Exception:
                pass
            out[str(name)] = {"num_examples": int(num) if isinstance(num, int) else None}
        return out

    # list case
    if isinstance(splits, list):
        for s in splits:
            # SplitInfo-like
            if hasattr(s, "name"):
                name = getattr(s, "name", "unknown")
                num = getattr(s, "num_examples", None)
                out[str(name)] = {"num_examples": int(num) if isinstance(num, int) else None}
            # string split name only
            elif isinstance(s, str):
                out[s] = {"num_examples": None}
            # dict-like
            elif isinstance(s, dict):
                name = s.get("name", "unknown")
                num = s.get("num_examples")
                out[str(name)] = {"num_examples": int(num) if isinstance(num, int) else None}
    return out

def _select_split_for_example(splits: Dict[str, Dict[str, Optional[int]]]) -> str:
    if not splits:
        return "train"
    # prefer the largest if counts exist; otherwise use common preference order
    with_counts = [(n, info.get("num_examples")) for n, info in splits.items() if info.get("num_examples") is not None]
    if with_counts:
        with_counts.sort(key=lambda x: x[1] or 0, reverse=True)
        return with_counts[0][0]
    for preferred in ("train", "validation", "val", "dev", "test"):
        if preferred in splits:
            return preferred
    # else just the first
    return next(iter(splits.keys()))

def summarize_dataset(repo: str, config: Optional[str] = None, max_example_chars: int = 120) -> Dict[str, Any]:
    repo_id = _parse_repo_id(repo)
    api = HfApi()

    # Card / repo info (license, languages, tags, etc.)
    try:
        info = api.dataset_info(repo_id)
        card = info.cardData or {}
        license_ = card.get("license") or getattr(info, "license", None)
        if license_:
            card = dict(card)
            card["license"] = license_
    except Exception:
        info, card = None, {}

    # Configs
    try:
        configs = get_dataset_config_names(repo_id)
        if not configs:
            configs = ["default"]
    except Exception:
        configs = ["default"]

    if config:
        if config not in configs:
            raise SystemExit(f"Config '{config}' not found. Available: {configs}")
        configs = [config]

    by_config: Dict[str, Any] = {}

    for cfg in configs:
        try:
            ci = get_dataset_config_info(repo_id, cfg if cfg != "default" else None)
        except Exception:
            ci = None

        features = _normalize_features(getattr(ci, "features", None)) if ci else None
        splits = _extract_splits(ci) if ci else {}
        size_in_bytes = getattr(ci, "size_in_bytes", None) if ci else None
        download_size = getattr(ci, "download_size", None) if ci else None
        dataset_size = getattr(ci, "dataset_size", None) if ci else None

        # Try streaming one example row
        example: Any = None
        try:
            target_split = _select_split_for_example(splits)
            ds = load_dataset(
                repo_id,
                cfg if cfg != "default" else None,
                split=target_split,
                streaming=True,
            )
            first = next(iter(ds))
            example = {k: _truncate(v, max_example_chars) for k, v in first.items()}
        except Exception:
            example = None

        by_config[cfg] = {
            "features": features,
            "splits": splits,  # {"train": {"num_examples": int|None}, ...}
            "sizes": {
                "size_in_bytes": size_in_bytes,
                "download_size": download_size,
                "dataset_size": dataset_size,
            },
            "example": example,
        }

    out = {
        "repo_id": repo_id,
        "configs": configs,
        "card": {
            # Keep only fields that are most LLM-useful; pass through if present.
            "license": card.get("license"),
            "task_categories": card.get("task_categories"),
            "task_ids": card.get("task_ids"),
            "language": card.get("language"),
            "languages": card.get("languages"),
            "tags": card.get("tags"),
            "pretty_name": card.get("pretty_name"),
            "size_categories": card.get("size_categories"),
            "dataset_summary": card.get("dataset_summary") or card.get("summary"),
            "description": card.get("description"),
        },
        "by_config": by_config,
    }
    return out

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Name or URL, e.g. 'imdb' or 'allenai/c4'")
    p.add_argument("--config", default=None, help="Configuration to summarize (if multi-config)")
    p.add_argument("--max-example-chars", type=int, default=800, help="Truncate long strings in example")
    args = p.parse_args()

    summary = summarize_dataset(args.dataset, args.config, args.max_example_chars)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()