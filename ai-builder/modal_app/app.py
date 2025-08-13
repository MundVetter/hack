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


# -------- Planning via OpenAI (GPT-5) --------
@app.function(image=image, secrets=[openai_secret])
def generate_plan(prompt: str) -> Dict[str, Any]:
    """Ask GPT-5 to propose a dataset and a constrained architecture spec.
    The response must be JSON with fields: dataset, input, model, train.
    """
    import os
    from openai import OpenAI

    client = OpenAI()
    system = (
        "You are an assistant that outputs STRICT JSON for ML project planning. "
        "Return a compact JSON object with keys: dataset, input, model, train. "
        "- dataset.name: one of ['mnist','fashion_mnist','cifar10']\n"
        "- input.shape: [channels, height, width] (mnist= [1,28,28], cifar10=[3,32,32])\n"
        "- model.layers: array of layers. Each layer is one of: \n"
        "  {type:'conv', out_channels:int, kernel:int, stride:int, padding:int, activation:'relu'|'none'} |\n"
        "  {type:'maxpool', kernel:int, stride:int} |\n"
        "  {type:'dropout', p:float} |\n"
        "  {type:'flatten'} |\n"
        "  {type:'dense', out_features:int, activation:'relu'|'none'}\n"
        "- train: {epochs:int<=5, batch_size:int<=128, lr:float<=0.01}\n"
        "- num_classes: 10\n"
        "Do not include any text besides the JSON."
    )

    user = f"Prompt: {prompt}"
    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    content = resp.output_text
    try:
        plan = json.loads(content)
    except Exception:
        # Fallback to MNIST minimal plan
        plan = {
            "dataset": {"name": "mnist"},
            "input": {"shape": [1, 28, 28]},
            "num_classes": 10,
            "model": {"layers": [
                {"type": "conv", "out_channels": 32, "kernel": 3, "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "maxpool", "kernel": 2, "stride": 2},
                {"type": "conv", "out_channels": 64, "kernel": 3, "stride": 1, "padding": 1, "activation": "relu"},
                {"type": "maxpool", "kernel": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "out_features": 128, "activation": "relu"},
            ]},
            "train": {"epochs": 2, "batch_size": 64, "lr": 0.001}
        }
    return plan


def _build_model_from_spec(spec: Dict[str, Any], in_ch: int, num_classes: int, image_hw: int) -> Any:
    import torch
    import torch.nn as nn

    layers: List[nn.Module] = []
    current_channels = in_ch
    current_h = image_hw
    current_w = image_hw

    for layer in spec.get("model", {}).get("layers", []):
        t = layer.get("type")
        if t == "conv":
            out_ch = int(layer.get("out_channels", 32))
            k = int(layer.get("kernel", 3))
            s = int(layer.get("stride", 1))
            p = int(layer.get("padding", 1))
            layers.append(nn.Conv2d(current_channels, out_ch, k, stride=s, padding=p))
            if layer.get("activation", "relu") == "relu":
                layers.append(nn.ReLU())
            # update spatial dims
            current_h = (current_h + 2 * p - k) // s + 1
            current_w = (current_w + 2 * p - k) // s + 1
            current_channels = out_ch
        elif t == "maxpool":
            k = int(layer.get("kernel", 2))
            s = int(layer.get("stride", k))
            layers.append(nn.MaxPool2d(k, stride=s))
            current_h = (current_h - k) // s + 1
            current_w = (current_w - k) // s + 1
        elif t == "dropout":
            p = float(layer.get("p", 0.5))
            layers.append(nn.Dropout(p))
        elif t == "flatten":
            layers.append(nn.Flatten())
        elif t == "dense":
            # infer in_features if after flatten, else compute
            if not isinstance(layers[-1], nn.Flatten):
                layers.append(nn.Flatten())
            in_features = current_channels * current_h * current_w
            out_features = int(layer.get("out_features", 128))
            layers.append(nn.Linear(in_features, out_features))
            if layer.get("activation", "relu") == "relu":
                layers.append(nn.ReLU())
            # update feature dims
            current_channels, current_h, current_w = 1, 1, out_features
        else:
            continue

    # Final classifier layer
    if not isinstance(layers[-1], nn.Flatten) and not isinstance(layers[-1], nn.Linear):
        layers.append(nn.Flatten())
    in_features = current_channels * max(current_h, 1) * max(current_w, 1)
    layers.append(nn.Linear(in_features, num_classes))

    return nn.Sequential(*layers)


@app.function(image=image, volumes={MODELS_DIR: volume}, timeout=60 * 60, secrets=[openai_secret])
def train(job_id: str, prompt: str) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset

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

    # Plan with GPT-5
    plan = generate_plan.remote(prompt)

    dataset_name = plan.get("dataset", {}).get("name", "mnist")
    input_shape = plan.get("input", {}).get("shape", [1, 28, 28])
    in_ch, H, W = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
    num_classes = int(plan.get("num_classes", 10))
    train_cfg = plan.get("train", {"epochs": 2, "batch_size": 64, "lr": 0.001})
    epochs = min(int(train_cfg.get("epochs", 2)), 5)
    batch_size = min(int(train_cfg.get("batch_size", 64)), 128)
    lr = float(train_cfg.get("lr", 0.001))

    slug = _slugify(f"{dataset_name}-classifier")

    # Transforms: to tensor, resize if needed
    tfs = [transforms.ToTensor()]
    if H != 28 or W != 28:
        tfs.insert(0, transforms.Resize((H, W)))
    if in_ch == 1:
        tfs.insert(0, transforms.Grayscale())
    transform = transforms.Compose(tfs)

    ds = load_dataset(dataset_name)

    class TorchDS(torch.utils.data.Dataset):
        def __init__(self, hf_split):
            self.data = hf_split
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            x = transform(item["image"])  # [C,H,W]
            y = int(item["label"]) if "label" in item else int(item["labels"])  # some datasets
            return x, y

    train_ds = TorchDS(ds["train"])
    test_ds = TorchDS(ds.get("test", ds["train"]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = _build_model_from_spec(plan, in_ch, num_classes, H)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
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

    # Save artifacts and plan
    torch.save(model.state_dict(), f"{MODELS_DIR}/{job_id}/model.pt")
    with open(meta_path, "w") as f:
        json.dump({
            "jobId": job_id,
            "slug": slug,
            "dataset": dataset_name,
            "input": {"shape": [in_ch, H, W]},
            "num_classes": num_classes,
            "plan": plan,
            "metrics": {"accuracy": acc},
        }, f)
    write_status("completed", {"slug": slug})
    return {"jobId": job_id, "slug": slug, "metrics": {"accuracy": acc}}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
def predict_internal(job_or_slug: str, pixels: Optional[list[float]] = None) -> Dict[str, Any]:
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
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    model.eval()

    if pixels is None:
        raise ValueError("pixels required for image classification demo")

    x = (torch.tensor(pixels, dtype=torch.float32).view(1, int(in_ch), int(H), int(W)))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy().tolist()[0]
        label = int(logits.argmax(dim=1).item())
    return {"label": str(label), "probs": probs}


@app.function(image=image, volumes={MODELS_DIR: volume}, secrets=[openai_secret])
@modal.web_endpoint(method="POST")
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
    out = predict_internal.remote(slug, pixels)
    return out