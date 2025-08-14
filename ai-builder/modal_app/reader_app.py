import modal

VOLUME_NAME = "ai-builder-models"
MODELS_DIR = "/models"

# A lightweight helper Modal app to read artifacts from the shared volume
app = modal.App("ai-builder-reader")
shared_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(image=modal.Image.debian_slim(), volumes={MODELS_DIR: shared_volume}, timeout=60)
def read_file(job_id: str, name: str) -> str:
    """Return file contents as text, or empty string if not found."""
    import os as _os
    path = f"{MODELS_DIR}/{job_id}/{name}"
    if not _os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()