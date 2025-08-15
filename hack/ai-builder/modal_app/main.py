from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Dict, Any, Optional, List

import modal

APP_NAME = "ai-builder"
VOLUME_NAME = "ai-builder-models"
MODELS_DIR = "/models"
image = modal.Image.debian_slim().pip_install_from_requirements(
    "./requirements.txt"
).pip_install(["fastapi[standard]"]).add_local_python_source("dataset_sumarizer")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
openai_secret = modal.Secret.from_name("openai-secret")


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9-_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "project"


def _find_job_id_for_slug(slug: str) -> Optional[str]:
    try:
        for entry in os.listdir(MODELS_DIR):
            meta_path = os.path.join(MODELS_DIR, entry, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("slug") == slug:
                    return entry
    except FileNotFoundError:
        return None
    return None


ml_image = modal.Image.debian_slim().pip_install([
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "seaborn",
    "transformers",
    "datasets",
    "scikit-learn",
    "pandas",
    "tqdm",
    "huggingface_hub",
    "Pillow",
    "fastapi[standard]",
])


my_dict = modal.Dict.from_name("summary-cache", create_if_missing=True)
job_results = modal.Dict.from_name("job-results", create_if_missing=True)


sandbox_app = modal.App.lookup("sandbox-train", create_if_missing=True)
@app.function(
    image=image, volumes={MODELS_DIR: volume}, timeout=60 * 60, secrets=[openai_secret]
)
async def train(
    job_id: str = "123", code: str = "print('hello')"
) -> Dict[str, Any]:
    """Execute the generated code in a sandboxed environment."""
    
    if not os.path.exists(f"{MODELS_DIR}/{job_id}"):
        os.makedirs(f"{MODELS_DIR}/{job_id}", exist_ok=True)
        with open(f"{MODELS_DIR}/{job_id}/status.json", "w") as f:
            json.dump({"status": "queued", "jobId": job_id}, f)
        volume.commit()

    with modal.enable_output():
        # Execute the generated code in a sandboxed environment
        sb = modal.Sandbox.create(
            app=sandbox_app, image=ml_image, volumes={MODELS_DIR: volume}, gpu="H200", verbose=True, name=job_id
        )
        p = await sb.exec.aio("python", "-c", f"{code}", timeout=150)

        async for line in p.stdout:
            # Avoid double newlines by using end="".
            print("stdout: ", line, end="")

        async for line in p.stderr:
            print("stderr: ", line,  end="")
            
        await p.wait.aio()

    return {"status": "completed", "jobId": job_id}


# @app.function(image=ml_image, volumes={MODELS_DIR: volume}, timeout=120, gpu="H200")
# def (job_or_slug: str= "job-1755178202", payload: Dict[str, Any] = {"text": "i hate this movie"}) -> Dict[str, Any]:
#     """Load a saved local HF model and run inference using transformers pipelines.
#     """

@app.function(image=image, volumes={MODELS_DIR: volume}, timeout=120)
def create_dataset_summary(dataset_id: str) -> Dict[str, Any]:
    """Load a saved local HF model and run inference using transformers pipelines.
    """
    from dataset_sumarizer import summarize_dataset
        # cache
    summary = my_dict.get(dataset_id)
    if not summary:
        summary = summarize_dataset(dataset_id)
        my_dict.put(dataset_id, summary)
    return summary


@app.function(image=ml_image, volumes={MODELS_DIR: volume}, timeout=120, gpu="H200")
def predict_internal(job_or_slug: str= "job-1755178202", payload: Dict[str, Any] = {"text": "i hate this movie"}) -> Dict[str, Any]:
    """Load a saved local HF model and run inference using transformers pipelines.

    job_or_slug: job id or slug
    payload: expects one of keys {"text", "inputs", "pixels", "image_base64", "audio_base64"}
    """
    import json as _json
    from typing import Any as _Any
    from transformers import pipeline, AutoTokenizer, AutoConfig
    import os as _os

    job_id = job_or_slug
    # Map slug->jobId if needed
    if not _os.path.exists(f"{MODELS_DIR}/{job_id}"):
        mapped = _find_job_id_for_slug(job_or_slug)
        if mapped:
            job_id = mapped

    model_dir = f"{MODELS_DIR}/{job_id}/hf_model"
    meta_path = f"{MODELS_DIR}/{job_id}/meta.json"

    # Default task based on input type
    task = "text-classification"
    inputs: _Any = None
    
    # Check for text inputs first
    if payload.get("text") is not None or payload.get("inputs") is not None:
        inputs = payload.get("text") or payload.get("inputs")
        task = "text-classification"
    
    # Check for image inputs
    elif payload.get("pixels") is not None:
        inputs = payload.get("pixels")
        task = "image-classification"
    
    elif payload.get("image_base64") is not None:
        inputs = payload.get("image_base64")
        task = "image-classification"
    
    # Check for audio inputs
    elif payload.get("audio_base64") is not None:
        inputs = payload.get("audio_base64")
        task = "automatic-speech-recognition"  # Use valid transformers task
    
    # Override task from metadata if available
    if _os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                meta = _json.load(mf)
            if isinstance(meta, dict) and meta.get("task"):
                task = str(meta["task"]).strip() or task
        except Exception:
            pass

    if inputs is None:
        return {"error": "Missing inputs"}

    print(f"Running inference with task: {task}, inputs type: {type(inputs)}")

    # Build pipeline based on task
    try:
        if task in ["image-classification", "automatic-speech-recognition"]:
            # Non-text tasks that don't need tokenizer
            nlp = pipeline(task, model=model_dir)  # type: ignore
        else:
            # Text-based tasks
            nlp = pipeline(task, model=model_dir, tokenizer=model_dir)  # type: ignore
    except Exception as e:
        return {"error": f"Failed to create pipeline: {e}"}

    try:
        result = nlp(inputs)
    except Exception as e:
        return {"error": f"inference_failed: {e}"}

    return {"jobId": job_id, "task": task, "result": result}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
@modal.fastapi_endpoint(method="POST")
def start(request: Dict[str, Any]):
    prompt = (request or {}).get("prompt", "")
    if not prompt:
        return {"error": "Missing prompt"}, 400
    job_id = _slugify(f"job-{int(time.time())}")

    os.makedirs(f"{MODELS_DIR}/{job_id}", exist_ok=True)
    with open(f"{MODELS_DIR}/{job_id}/status.json", "w") as f:
        json.dump({"status": "queued", "jobId": job_id}, f)
    volume.commit()

    train.spawn(job_id, prompt)
    return {"jobId": job_id}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
@modal.fastapi_endpoint(method="GET")
def status(jobId: str):
    status_path = f"{MODELS_DIR}/{jobId}/status.json"
    meta_path = f"{MODELS_DIR}/{jobId}/meta.json"
    if not os.path.exists(status_path):
        return {"status": "unknown"}
    with open(status_path) as f:
        data = json.load(f)
    if os.path.exists(meta_path):
        with open(meta_path) as mf:
            meta = json.load(mf)
        data.update({"slug": meta.get("slug"), "metrics": meta.get("metrics")})
    return data


@app.function(image=ml_image, volumes={MODELS_DIR: volume})
@modal.fastapi_endpoint(method="POST")
def predict(request: Dict[str, Any]):
    slug = (request or {}).get("slug") or (request or {}).get("jobId")
    text = (request or {}).get("text") or (request or {}).get("inputs")
    pixels = (request or {}).get("pixels")
    image_base64 = (request or {}).get("image_base64")
    audio_base64 = (request or {}).get("audio_base64")
    if not slug:
        return {"error": "Missing slug"}, 400
    if text is None and pixels is None and image_base64 is None and audio_base64 is None:
        return {"error": "Missing inputs (provide 'text', 'inputs', 'pixels', 'image_base64', or 'audio_base64')"}, 400
    payload: Dict[str, Any] = {}
    if text is not None:
        payload["text"] = text
    if pixels is not None:
        payload["pixels"] = pixels
    if image_base64 is not None:
        payload["image_base64"] = image_base64
    if audio_base64 is not None:
        payload["audio_base64"] = audio_base64
    out = predict_internal.remote(slug, payload)
    return out