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
    job_id: str = "123", prompt: str = "create a sentiment classifier for imdb dataset", dataset_id: str = "imdb"
) -> Dict[str, Any]:
    import openai
    from dataset_sumarizer import summarize_dataset

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    if not os.path.exists(f"{MODELS_DIR}/{job_id}"):
        os.makedirs(f"{MODELS_DIR}/{job_id}", exist_ok=True)
        with open(f"{MODELS_DIR}/{job_id}/status.json", "w") as f:
            json.dump({"status": "queued", "jobId": job_id}, f)
        volume.commit()

    # cache
    summary = my_dict.get(dataset_id)
    if not summary:
        summary = summarize_dataset(dataset_id)
        my_dict.put(dataset_id, summary)

    # Create enhanced input that includes both the prompt and dataset summary
    enhanced_input = f"""
USER PROMPT:
{prompt}

DATASET SUMMARY:
{json.dumps(summary, indent=2)}

Please use the dataset summary above to understand the data structure, features, and available splits. The summary includes:
- Dataset features and their types
- Available splits (train, test, validation) with example counts
- Sample data examples
- Dataset metadata and task information

Use this information to create appropriate data loaders, preprocessing, and model architecture.
"""

    response = client.responses.create(
        model="gpt-5",
        instructions=f"""
# You are an expert machine learning engineer and scientist.
# Given a user prompt describing a machine learning problem AND a dataset summary,
# write Python code that creates a complete ML solution using only the following packages:
# PyTorch, torch.nn, torch.optim, torch.utils.data, numpy, matplotlib, seaborn,
# transformers, datasets, scikit-learn, pandas, tqdm, json, os, time, random,
# and built-in Python libraries.
# CUDA is available on device 'cuda:0' with an NVIDIA H200 GPU.

# IMPORTANT: The input includes a dataset summary that provides:
# - Dataset features and their data types
# - Available splits (train, test, validation) with example counts
# - Sample data examples showing the structure
# - Dataset metadata and task information
# Use this summary to create appropriate data loaders, preprocessing, and model architecture.

# CRITICAL CONSTRAINTS:
# - Training must complete within 2 minutes (120 seconds)
# - Use small model architectures and limited epochs/steps
# - For large datasets, use only a subset of data
# - Use early stopping to prevent overfitting
# - Limit batch size and model complexity

# The code should train a model to solve the user's problem,
# save the trained model weights to {MODELS_DIR}/{job_id}/model.pt,
# The code should be executable from a main block and should run as a script.
# Do not use any packages other except for standard Python libraries.
# Do not include explanations, only the code.
# THE CODE SHOULD WORK WITHOUT MODIFICATIONS.
# Include proper error handling, logging, and validation.
# Use best practices: set random seeds for reproducibility and use a validation set.
# Ensure the code handles both training and evaluation phases properly.
# The model should be saved in a way that it can be loaded and used for inference later.

# LOGGING REQUIREMENTS:
# - Create a CSV file at {MODELS_DIR}/{job_id}/losses.csv
# - Log training loss, validation loss, and validation accuracy every 25 steps
# - CSV should have columns: step, train_loss, val_loss, val_accuracy
# - Update the CSV file after each logging interval
# - Use pandas to write the CSV
# - write the final metricts to a json file at {MODELS_DIR}/{job_id}/metrics.json

# INFERENCE AND ARTIFACT REQUIREMENTS (IMPORTANT):
# - If using Hugging Face Transformers, also save both the model and tokenizer using save_pretrained
#   into the directory {MODELS_DIR}/{job_id}/hf_model (create it if needed).
#   Example:
#       model.save_pretrained(f"{MODELS_DIR}/{job_id}/hf_model")
#       tokenizer.save_pretrained(f"{MODELS_DIR}/{job_id}/hf_model")
# - Write a metadata file at {MODELS_DIR}/{job_id}/meta.json containing at least:
#     {{
#       "jobId": "{job_id}",
#       "task": "text-classification"  # or one of: text-generation, token-classification, summarization, translation, image-classification
#     }}
#   Set the correct task string for the model you trained.
# - Ensure that the code runs end-to-end within 120 seconds and writes all artifacts.

# OUTPUT ONLY THE CODE. DO NOT INCLUDE ANY EXPLANATIONS, COMMENTS, OR ANYTHING ELSE.
# """,
        input=enhanced_input,
        reasoning={
            "effort": "medium"
        }
    )

    output = response.output_text
#     # print("reasoning:")
#     # print(response.reasoning_content)

#     print(output)
    with open(f"{MODELS_DIR}/{job_id}/code.py", "w") as f:
        f.write(output)
    job_results.put(job_id, {"code": output})
    volume.commit()
#     print("Code written to", f"{MODELS_DIR}/{job_id}/code.py")
    # with open(f"{MODELS_DIR}/{job_id}/code.py", "r") as f:
    #     output = f.read()

    with modal.enable_output():
        # now we execute the code in a sandboxed environment
        sb = modal.Sandbox.create(
            app=sandbox_app, image=ml_image, volumes={MODELS_DIR: volume}, gpu="H200", verbose=True, name=job_id
        )
        p = await sb.exec.aio("python", "-c", f"{output}", timeout=150)

        async for line in p.stdout:
            # Avoid double newlines by using end="".
            print("stdout: ", line, end="")

        async for line in p.stderr:
            print("stderr: ", line,  end="")
            
        await p.wait.aio()
        # sb.terminate()

    return {"status": "completed", "jobId": job_id}




@app.function(image=ml_image, volumes={MODELS_DIR: volume}, timeout=120, gpu="H200")
def predict_internal(job_or_slug: str= "job-1755178202", payload: Dict[str, Any] = {"text": "i hate this movie"}) -> Dict[str, Any]:
    """Load a saved local HF model and run inference using transformers pipelines.

    job_or_slug: job id or slug
    payload: expects one of keys {"text", "inputs", "pixels", "image_base64"}
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

    task = "text-classification"
    if _os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                meta = _json.load(mf)
            if isinstance(meta, dict) and meta.get("task"):
                task = str(meta["task"]).strip() or task
        except Exception:
            pass

    inputs: _Any = payload.get("text") or payload.get("inputs")
    print("inputssssaa", inputs)

    # Fallback to pixels (image) if provided
    if inputs is None and payload.get("pixels") is not None:
        task = "image-classification"
        inputs = payload.get("pixels")

    if inputs is None and payload.get("image_base64") is not None:
        task = "image-classification"
        inputs = payload.get("image_base64")

    if inputs is None:
        return {"error": "Missing inputs"}

    # Build pipeline
    nlp = pipeline(task, model=model_dir, tokenizer=model_dir) if task != "image-classification" else pipeline(task, model=model_dir)

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
    if not slug:
        return {"error": "Missing slug"}, 400
    if text is None and pixels is None and image_base64 is None:
        return {"error": "Missing inputs (provide 'text', 'inputs', 'pixels', or 'image_base64')"}, 400
    payload: Dict[str, Any] = {}
    if text is not None:
        payload["text"] = text
    if pixels is not None:
        payload["pixels"] = pixels
    if image_base64 is not None:
        payload["image_base64"] = image_base64
    out = predict_internal.remote(slug, payload)
    return out