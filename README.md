## AI Builder

An end-to-end workflow that turns natural language prompts and Hugging Face dataset summaries into runnable machine learning training code, executes it in a secure GPU sandbox on Modal, tracks artifacts/metrics, and serves simple inference.

### What it does
- **Dataset summarization**: Builds a compact JSON summary of any Hugging Face dataset (features, splits, example row, basic card data).
- **Code generation**: Uses OpenAI `gpt-5` to generate training code tailored to the dataset summary and user prompt.
- **Sandboxed training**: Executes the generated code inside a GPU Modal Sandbox and writes artifacts to a shared volume.
- **Artifacts** written under `/models/{job_id}`:
  - `code.py` (generated training script)
  - `losses.csv` (step, train_loss, val_loss, val_accuracy)
  - `metrics.json` (final metrics)
  - `meta.json` (includes `jobId`, `task`, optional `slug`)
  - `hf_model/` (saved model/tokenizer for transformers-based models)
- **APIs**: Start jobs, poll status, run inference via FastAPI endpoints hosted on Modal.
- **Dashboard**: A Streamlit UI to start/attach jobs, watch training curves, view generated code, and run inference.

### Repository layout (relevant parts)
- `hack/ai-builder/modal_app/` (Python Modal apps)
  - `main.py`: Core Modal app (`ai-builder`) with `train`, `predict_internal`, `start`, `status`, `predict` endpoints
  - `reader_app.py`: Helper Modal app (`ai-builder-reader`) to read files from the shared volume
  - `st_app.py`: Streamlit dashboard for starting jobs, monitoring, and inference
  - `dataset_sumarizer.py`: Utilities/CLI to summarize Hugging Face datasets

Note: You may also have a separate frontend in `ai-builder/` (Node/Next.js). This README focuses on the Python/Modal workflow.

### Prerequisites
- Python 3.10+
- A Modal account and CLI (`pip install modal`) and `modal token set ...` authentication
- OpenAI API key with access to `gpt-5`
- Optional: Hugging Face auth if accessing gated datasets

### Configure secrets
Create a Modal secret named `openai-secret` containing your OpenAI key:

```bash
modal secret create openai-secret --from-literal OPENAI_API_KEY=sk-...
```

### Local Python environment (for the Streamlit dashboard & CLI)
```bash
python -m venv /Users/mundvetter/hack/venv
source /Users/mundvetter/hack/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install modal streamlit pandas
```

The Modal training image installs its own runtime deps (torch, transformers, datasets, etc.) as defined inside `main.py` and `requirements.txt`.

### Deploy the Modal apps
Deploy both the core app and the reader app so Streamlit/clients can look them up and call them:

```bash
modal deploy /Users/mundvetter/hack/hack/ai-builder/modal_app/main.py
modal deploy /Users/mundvetter/hack/hack/ai-builder/modal_app/reader_app.py
```

Deployment will print HTTPS URLs for the exposed FastAPI endpoints.

### Run the Streamlit dashboard
```bash
streamlit run /Users/mundvetter/hack/hack/ai-builder/modal_app/st_app.py
```

In the sidebar:
- Enter a prompt (e.g., "create a sentiment classifier for imdb dataset") and dataset ID (e.g., `imdb`).
- Optionally provide a custom Job ID, then click Start Training. Or attach to an existing job.
- The main view shows status, generated code, live loss/accuracy charts, final metrics, and a simple inference panel.

### REST API
After deploy, use the printed base URL (example placeholder: `https://<your-account>--ai-builder.modal.run`).

- Start a job
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"create a sentiment classifier for imdb dataset"}' \
  https://YOUR_BASE_URL/start
```

- Check status
```bash
curl "https://YOUR_BASE_URL/status?jobId=JOB_ID"
```

- Predict (text input or `inputs`, or `pixels`/`image_base64` for image-classification)
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"slug":"JOB_ID_OR_SLUG","text":"I loved this movie!"}' \
  https://YOUR_BASE_URL/predict
```

### Programmatic inference via Modal
```python
import modal

predict_fn = modal.Function.lookup("ai-builder", "predict_internal")
out = predict_fn.remote("JOB_ID_OR_SLUG", {"text": "I loved this movie!"})
print(out)
```

### Dataset summarizer CLI
```bash
python /Users/mundvetter/hack/hack/ai-builder/modal_app/dataset_sumarizer.py --dataset imdb
# with config
python /Users/mundvetter/hack/hack/ai-builder/modal_app/dataset_sumarizer.py --dataset glue --config sst2
```

### How training works (high level)
1. `dataset_sumarizer.py` gathers dataset features/splits/example.
2. `main.py::train` prompts OpenAI `gpt-5` to emit standalone Python training code constrained to fast, small runs.
3. The generated code is written to `/models/{job_id}/code.py` and executed in a Modal Sandbox with GPU `H200`.
4. During/after training, artifacts (losses, metrics, meta, and optional `hf_model/`) are saved to the shared volume.
5. Inference loads the saved model using `transformers` pipelines based on the `task` in `meta.json`.

### Troubleshooting
- If the Streamlit app cannot find functions, ensure both Modal apps are deployed and you are logged in via `modal token set`.
- Ensure the `openai-secret` secret exists and contains `OPENAI_API_KEY`.
- GPU type `H200` must be available on your Modal plan; adjust in `main.py` if needed.
- Large datasets: the generated code intentionally subsamples and limits epochs to finish within ~2 minutes.

### License
See `LICENSE` at the repository root.

