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
).add_local_python_source("dataset_sumarizer")

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


ml_image = modal.Image.debian_slim().pip_install(["torch", "torchvision", "numpy", "matplotlib", "seaborn", "transformers", "datasets", "scikit-learn", "pandas", "tqdm"], "huggingface_hub")


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

    summary = summarize_dataset(dataset_id)

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
# CUDA is available on device 'cuda:0' with an NVIDIA A100 GPU.

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
# - Log training loss, validation loss, and validation accuracy every 10 steps
# - CSV should have columns: step, train_loss, val_loss, val_accuracy
# - Update the CSV file after each logging interval
# - Use pandas to write the CSV

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
#     print("Code written to", f"{MODELS_DIR}/{job_id}/code.py")
    # with open(f"{MODELS_DIR}/{job_id}/code.py", "r") as f:
    #     output = f.read()

    with modal.enable_output():
        # now we execute the code in a sandboxed environment
        sb = modal.Sandbox.create(
            app=sandbox_app, image=ml_image, volumes={MODELS_DIR: volume}, gpu="A100", verbose=True
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


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
def predict_internal(
    job_or_slug: str, pixels: Optional[list[float]] = None
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    # Load meta to rebuild the model
    job_id = job_or_slug
    model_dir = f"{MODELS_DIR}/{job_or_slug}"
    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        j = _find_job_id_for_slug(job_or_slug)
        if j:
            model_dir = f"{MODELS_DIR}/{j}"
            job_id = j
        else:
            raise FileNotFoundError("Model not found")

    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    in_ch, H, W = meta.get("input", {}).get("shape", [1, 28, 28])
    num_classes = int(meta.get("num_classes", 10))
    plan = meta.get("plan", {"model": {"layers": []}})

    model = _build_model_from_spec(plan, int(in_ch), int(num_classes), int(H))
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    )
    model.eval()

    if pixels is None:
        raise ValueError("pixels required for image classification demo")

    x = torch.tensor(pixels, dtype=torch.float32).view(1, int(in_ch), int(H), int(W))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy().tolist()[0]
        label = int(logits.argmax(dim=1).item())
    return {"label": str(label), "probs": probs}


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


@app.function(image=image, secrets=[openai_secret])
@modal.fastapi_endpoint(method="POST")
def predict(request: Dict[str, Any]):
    slug = (request or {}).get("slug") or (request or {}).get("jobId")
    pixels = (request or {}).get("pixels")
    if not slug:
        return {"error": "Missing slug"}, 400
    if pixels is None:
        return {"error": "Missing pixels"}, 400
    out = predict_internal.remote(slug, pixels)
    return out
