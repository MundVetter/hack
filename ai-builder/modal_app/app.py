from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Dict, Any, Optional

import modal

APP_NAME = "ai-builder"
VOLUME_NAME = "ai-builder-models"
MODELS_DIR = "/models"

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_app/requirements.txt")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
openai_secret = modal.Secret.from_name("openai")


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


@app.function(image=image, volumes={MODELS_DIR: volume}, timeout=60 * 60, secrets=[openai_secret])
def train(job_id: str, prompt: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    import numpy as np

    os.makedirs(f"{MODELS_DIR}/{job_id}", exist_ok=True)
    status_path = f"{MODELS_DIR}/{job_id}/status.json"
    meta_path = f"{MODELS_DIR}/{job_id}/meta.json"

    def write_status(state: str, extra: Optional[Dict[str, Any]] = None):
        payload = {"status": state, "jobId": job_id}
        if extra:
            payload.update(extra)
        with open(status_path, "w") as f:
            json.dump(payload, f)
        volume.commit()

    write_status("running")

    prompt_l = prompt.lower()
    if "mnist" in prompt_l:
        dataset_name = "mnist"
        input_channels = 1
        num_classes = 10
        slug = _slugify("mnist-classifier")
    else:
        dataset_name = "mnist"
        input_channels = 1
        num_classes = 10
        slug = _slugify("mnist-classifier")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    ds = load_dataset(dataset_name)

    class TorchDS(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self.data = hf_split
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            x = transform(item["image"])  # [1, 28, 28]
            y = int(item["label"])
            return x, y

    train_ds = TorchDS(ds["train"])
    test_ds = TorchDS(ds.get("test", ds["train"]))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    class SimpleCNN(nn.Module):
        def __init__(self, in_ch: int, num_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(input_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / max(total, 1)

    torch.save(model.state_dict(), f"{MODELS_DIR}/{job_id}/model.pt")
    with open(meta_path, "w") as f:
        json.dump({
            "jobId": job_id,
            "slug": slug,
            "dataset": dataset_name,
            "input": {"shape": [1, 28, 28]},
            "metrics": {"accuracy": acc},
        }, f)
    write_status("completed", {"slug": slug})
    return {"jobId": job_id, "slug": slug, "metrics": {"accuracy": acc}}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
def predict_internal(job_or_slug: str, pixels: Optional[list[float]] = None) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            return self.net(x)

    model_dir = f"{MODELS_DIR}/{job_or_slug}"
    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        # try slug lookup
        j = _find_job_id_for_slug(job_or_slug)
        if j:
            model_dir = f"{MODELS_DIR}/{j}"
        else:
            raise FileNotFoundError("Model not found")
    model_path = os.path.join(model_dir, "model.pt")

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    if pixels is None:
        raise ValueError("pixels required for MNIST demo")

    x = (torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy().tolist()[0]
        label = int(logits.argmax(dim=1).item())
    return {"label": str(label), "probs": probs}


@app.function(image=image, secrets=[openai_secret])
@modal.web_endpoint(method="POST")
def start(request: Dict[str, Any]):
    prompt = (request or {}).get("prompt", "")
    if not prompt:
        return {"error": "Missing prompt"}, 400
    job_id = _slugify(f"job-{int(time.time())}")

    with volume.reload():
        os.makedirs(f"{MODELS_DIR}/{job_id}", exist_ok=True)
        with open(f"{MODELS_DIR}/{job_id}/status.json", "w") as f:
            json.dump({"status": "queued", "jobId": job_id}, f)
        volume.commit()

    train.spawn(job_id, prompt)
    return {"jobId": job_id}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
@modal.web_endpoint(method="GET")
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
@modal.web_endpoint(method="POST")
def predict(request: Dict[str, Any]):
    slug = (request or {}).get("slug") or (request or {}).get("jobId")
    pixels = (request or {}).get("pixels")
    if not slug:
        return {"error": "Missing slug"}, 400
    if pixels is None:
        return {"error": "Missing pixels"}, 400
    out = predict_internal.call(slug, pixels)
    return out