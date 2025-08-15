#!/usr/bin/env python3
"""
Summarize any Hugging Face dataset as compact JSON for LLM consumption.

Usage:
  python hf_dataset_summary.py --dataset imdb
  python hf_dataset_summary.py --dataset glue --config sst2
  python hf_dataset_summary.py --dataset https://huggingface.co/datasets/allenai/c4 --config en
"""

import argparse, json, sys
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from collections import Counter, defaultdict
import re

# --- deps (give a helpful message if missing) ---
from datasets import get_dataset_config_names, get_dataset_config_info, load_dataset
from huggingface_hub import HfApi
from datasets import get_dataset_split_names
import numpy as np

import os
import requests

# ---------- data analysis helpers ----------

def analyze_text_data(text_values: List[str]) -> Dict[str, Any]:
    """Analyze text data and provide summary statistics."""
    if not text_values:
        return {"type": "text", "count": 0}

    # Filter out None/empty values
    valid_texts = [str(t) for t in text_values if t and str(t).strip()]

    if not valid_texts:
        return {"type": "text", "count": 0, "empty_count": len(text_values)}

    lengths = [len(t) for t in valid_texts]
    word_counts = [len(t.split()) for t in valid_texts]

    # Character-level analysis
    all_chars = ''.join(valid_texts)
    char_freq = Counter(all_chars)

    # Word-level analysis
    all_words = []
    for text in valid_texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)

    word_freq = Counter(all_words)

    # Language detection hints
    non_ascii_chars = sum(1 for c in all_chars if ord(c) > 127)
    language_hint = "multilingual" if non_ascii_chars > len(all_chars) * 0.1 else "english"

    return {
        "type": "text",
        "count": len(valid_texts),
        "empty_count": len(text_values) - len(valid_texts),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
            "median": int(np.median(lengths)),
            "std": float(np.std(lengths))
        },
        "word_count_stats": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": float(np.mean(word_counts)),
            "median": int(np.median(word_counts))
        },
        "character_analysis": {
            "total_chars": len(all_chars),
            "unique_chars": len(char_freq),
            "most_common_chars": dict(char_freq.most_common(10)),
            "non_ascii_ratio": float(non_ascii_chars / len(all_chars))
        },
        "word_analysis": {
            "total_words": len(all_words),
            "unique_words": len(word_freq),
            "most_common_words": dict(word_freq.most_common(10)),
            "avg_word_length": float(np.mean([len(w) for w in all_words]))
        },
        "language_hint": language_hint
    }

def analyze_numerical_data(values: List[Union[int, float]]) -> Dict[str, Any]:
    """Analyze numerical data and provide summary statistics."""
    if not values:
        return {"type": "numerical", "count": 0}
    
    # Filter out None values and convert to float
    valid_values = []
    for v in values:
        try:
            if v is not None:
                valid_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not valid_values:
        return {"type": "numerical", "count": 0, "invalid_count": len(values)}
    
    valid_values_array = np.array(valid_values)
    
    # Basic statistics
    stats = {
        "min": float(np.min(valid_values_array)),
        "max": float(np.max(valid_values_array)),
        "mean": float(np.mean(valid_values_array)),
        "median": float(np.median(valid_values_array)),
        "std": float(np.std(valid_values_array)),
        "q25": float(np.percentile(valid_values_array, 25)),
        "q75": float(np.percentile(valid_values_array, 75))
    }
    
    # Detect if data looks like integers
    is_integer_like = all(abs(v - round(v)) < 1e-10 for v in valid_values)
    
    # Histogram bins for visualization
    if len(valid_values) > 1:
        hist, bin_edges = np.histogram(valid_values_array, bins=min(20, len(valid_values)//2))
        histogram = {
            "bin_edges": [float(e) for e in bin_edges],
            "counts": [int(c) for c in hist]
        }
    else:
        histogram = {"bin_edges": [float(valid_values[0])], "counts": [1]}
    
    return {
        "type": "numerical",
        "count": len(valid_values),
        "invalid_count": len(values) - len(valid_values),
        "is_integer_like": is_integer_like,
        "stats": stats,
        "histogram": histogram
    }

def analyze_image_data(values: List[Any]) -> Dict[str, Any]:
    """Analyze image data and provide summary statistics."""
    if not values:
        return {"type": "image", "count": 0}
    
    # Count different image formats/types
    image_types: Counter[str] = Counter()
    dimensions: List[Any] = []
    file_sizes: List[Any] = []
    
    for val in values:
        if val is None:
            continue
            
        val_str = str(val).lower()
        
        # Detect image format
        if any(ext in val_str for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
            image_types['file_path'] += 1
        elif 'pil' in val_str or 'image' in val_str:
            image_types['pil_object'] += 1
        elif 'tensor' in val_str or 'array' in val_str:
            image_types['tensor_array'] += 1
        else:
            image_types['unknown'] += 1
    
    return {
        "type": "image",
        "count": len([v for v in values if v is not None]),
        "null_count": len([v for v in values if v is None]),
        "format_distribution": dict(image_types),
        "note": "Image data detected - actual dimensions and properties would require loading the images"
    }

def analyze_categorical_data(values: List[Any]) -> Dict[str, Any]:
    """Analyze categorical data and provide summary statistics."""
    if not values:
        return {"type": "categorical", "count": 0}
    
    # Filter out None values
    valid_values = [str(v) for v in values if v is not None]
    
    if not valid_values:
        return {"type": "categorical", "count": 0, "null_count": len(values)}
    
    value_counts = Counter(valid_values)
    unique_count = len(value_counts)
    
    # Detect if it's boolean-like
    bool_like = all(v.lower() in ['true', 'false', '1', '0', 'yes', 'no'] for v in valid_values)
    
    return {
        "type": "categorical",
        "count": len(valid_values),
        "null_count": len(values) - len(valid_values),
        "unique_values": unique_count,
        "is_boolean_like": bool_like,
        "most_common": dict(value_counts.most_common(10)),
        "cardinality": "low" if unique_count <= 10 else "medium" if unique_count <= 100 else "high"
    }

def analyze_list_data(values: List[Any]) -> Dict[str, Any]:
    """Analyze list/array data and provide summary statistics."""
    if not values:
        return {"type": "list", "count": 0}
    
    valid_lists = [v for v in values if isinstance(v, (list, tuple)) and v is not None]
    
    if not valid_lists:
        return {"type": "list", "count": 0, "non_list_count": len(values)}
    
    lengths = [len(v) for v in valid_lists]
    
    # Analyze first few elements to understand structure
    sample_elements = []
    for lst in valid_lists[:5]:  # Sample first 5 lists
        if lst:
            sample_elements.append(str(lst[0])[:100])  # First element, truncated
    
    return {
        "type": "list",
        "count": len(valid_lists),
        "non_list_count": len(values) - len(valid_lists),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": float(np.mean(lengths)),
            "median": int(np.median(lengths))
        },
        "sample_first_elements": sample_elements
    }

def analyze_feature_values(feature_name: str, values: List[Any], max_samples: int = 1000) -> Dict[str, Any]:
    """Analyze a single feature and determine its type and statistics."""
    # Sample the data if it's too large
    if len(values) > max_samples:
        sample_indices = np.random.choice(len(values), max_samples, replace=False)
        sample_values = [values[i] for i in sample_indices]
    else:
        sample_values = values
    
    # Determine data type
    if not sample_values:
        return {"type": "unknown", "count": 0}
    
    # Check if all values are None
    if all(v is None for v in sample_values):
        return {"type": "null", "count": len(values), "null_count": len(values)}
    
    # Check if it's text-like
    text_like = all(isinstance(v, str) or v is None for v in sample_values)
    if text_like:
        return analyze_text_data(sample_values)
    
    # Check if it's numerical
    numerical_like = True
    for v in sample_values:
        if v is not None:
            try:
                float(v)
            except (ValueError, TypeError):
                numerical_like = False
                break
    
    if numerical_like:
        return analyze_numerical_data(sample_values)
    
    # Check if it's image-like
    image_like = any('image' in str(v).lower() or 'pil' in str(v).lower() or 
                     any(ext in str(v).lower() for ext in ['.jpg', '.png', '.jpeg']) 
                     for v in sample_values if v is not None)
    if image_like:
        return analyze_image_data(sample_values)
    
    # Check if it's list-like
    list_like = all(isinstance(v, (list, tuple)) or v is None for v in sample_values)
    if list_like:
        return analyze_list_data(sample_values)
    
    # Default to categorical
    return analyze_categorical_data(sample_values)

def create_feature_histogram(feature_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create a histogram summary of feature types and characteristics."""
    type_counts = Counter()
    data_types = []
    
    for feature_name, analysis in feature_analyses.items():
        data_type = analysis.get("type", "unknown")
        type_counts[data_type] += 1
        
        # Add feature info to data types
        data_types.append({
            "name": feature_name,
            "type": data_type,
            "count": analysis.get("count", 0),
            "null_count": analysis.get("null_count", 0),
            "cardinality": analysis.get("cardinality", "unknown") if data_type == "categorical" else None
        })
    
    return {
        "feature_type_distribution": dict(type_counts),
        "total_features": len(feature_analyses),
        "feature_details": data_types
    }

# ---------- existing helpers ----------

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

def get_sample(card: Any, config: str, split: str) -> Any:
    ds=  load_dataset(card.id, config, split=split, streaming=True)
    return next(iter(ds))




def summarize_dataset(repo: str) -> Dict[str, Any]:
    repo_id = _parse_repo_id(repo)
    api = HfApi()
    card =  api.dataset_info(repo_id)
    print(card.id)
    card_str = str(card)
    splits = get_dataset_split_names(card.id)
    configs = get_dataset_config_names(card.id)


    API_TOKEN = os.environ["HF_TOKEN"]
    stats = {}

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    # For each config and split pair, get stats. If too many, sample or skip.
    MAX_TOTAL_REQUESTS = 20
    stats = {}
    config_split_pairs = []
    for config in configs:
        try:
            config_splits = get_dataset_split_names(card.id, config)
        except Exception:
            config_splits = splits  # fallback to global splits if per-config fails
        for split in config_splits:
            config_split_pairs.append((config, split))

    # If too many, sample evenly across configs
    if len(config_split_pairs) > MAX_TOTAL_REQUESTS:
        # Try to sample at least one split per config, then fill up to max
        per_config = max(1, MAX_TOTAL_REQUESTS // max(1, len(configs)))
        sampled_pairs = []
        for config in configs:
            config_splits = [pair for pair in config_split_pairs if pair[0] == config]
            sampled_pairs.extend(config_splits[:per_config])
        # If still not enough, fill up with remaining pairs
        if len(sampled_pairs) < MAX_TOTAL_REQUESTS:
            remaining = [pair for pair in config_split_pairs if pair not in sampled_pairs]
            sampled_pairs.extend(remaining[:MAX_TOTAL_REQUESTS - len(sampled_pairs)])
        config_split_pairs = sampled_pairs

    for config, split in config_split_pairs:
        api_url = (
            f"https://datasets-server.huggingface.co/statistics"
            f"?dataset={card.id}&config={config}&split={split}"
        )
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            stats[split] = response.json()
        except Exception as e:
            stats[split] = {"error": str(e)}



    # info = {
    #     "id": card.id,
    #     "name": card.name,
    #     "author": card.author,
    #     "created_at": card.created_at,
    #     "data"
    #     "description": card.description,
    #     "citation": card.citation,
    #     "license": card.license,
    #     "size_in_bytes": card.size_in_bytes,
    #     "download_size": card.download_size,
    # }

    # provide first row of each split
    examples = {}
    for split in splits:
        examples[split] = str(get_sample(card, config, split))



    out =  {
        "card": card_str,
        "splits": splits,
        "configs": configs,
        "stats": stats,
        "examples": examples,
    }
    return out

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Name or URL, e.g. 'imdb' or 'allenai/c4'")
    # p.add_argument("--config", default=None, help="Configuration to summarize (if multi-config)")
    # p.add_argument("--max-example-chars", type=int, default=800, help="Truncate long strings in example")
    # p.add_argument("--no-feature-analysis", action="store_true", help="Skip feature analysis to save time")
    # p.add_argument("--max-analysis-samples", type=int, default=1000, help="Maximum samples to analyze for features")
    args = p.parse_args()

    summary = summarize_dataset(
        args.dataset,
        # args.config, 
        # args.max_example_chars,
        # analyze_features=not args.no_feature_analysis,
        # max_analysis_samples=args.max_analysis_samples
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
