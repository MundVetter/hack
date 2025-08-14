import os
import sys
import time
import json
from io import StringIO
from typing import Optional

import streamlit as st
import pandas as pd
import modal

# Ensure we can import the local package `modal_app`
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# if CURRENT_DIR not in sys.path:
# 	sys.path.append(CURRENT_DIR)

# # Import the remote train function from our Modal app
# try:
from main import train  # type: ignore
# except Exception as import_err:  # Fallback to inform the user
# 	train = None  # type: ignore
# 	st.error(f"Failed to import train function from modal_app.main: {import_err}")

APP_NAME = "ai-builder"
VOLUME_NAME = "ai-builder-models"
MODELS_DIR = "/models"



def start_training(job_id: str, prompt: str, dataset_id: str) -> None:
	"""Spawn the remote training job."""
	if train is None:
		raise RuntimeError("Training function is unavailable. Check imports.")
	train_fn = modal.Function.lookup("ai-builder", "train")
	# Start remote job asynchronously
	train_fn.spawn(job_id, prompt, dataset_id)


def poll_text(job_id: str, filename: str) -> str:
	"""Fetch text artifact via Modal; returns empty string if not found."""
	try:
		read_file_fn = modal.Function.lookup("ai-builder-reader", "read_file")
		result = read_file_fn.remote(job_id, filename)  # type: ignore[attr-defined]
		return str(result) if result is not None else ""
	except Exception:
		return ""


def parse_json(text: str) -> Optional[dict]:
	try:
		return json.loads(text) if text else None
	except Exception:
		return None


st.set_page_config(page_title="AI Builder Trainer", layout="wide")
st.title("AI Builder - Training Dashboard")

with st.sidebar:
	st.header("Start a new job")
	default_prompt = "create a sentiment classifier for imdb dataset"
	prompt = st.text_area("Prompt", value=default_prompt, height=120)
	dataset_id = st.text_input("Dataset ID", value="imdb")
	custom_job_id = st.text_input("Optional Job ID (leave blank to auto-generate)", value="")
	start_btn = st.button("Start Training", type="primary")

	st.divider()
	st.header("Attach to existing job")
	existing_job_id = st.text_input("Existing Job ID", value="")
	attach_btn = st.button("Attach")

# Session state
if "job_id" not in st.session_state:
	st.session_state.job_id = None

if start_btn:
	jid = custom_job_id.strip() or f"job-{int(time.time())}"
	st.session_state.job_id = jid
	with st.status("Spawning training job...", expanded=True) as status_box:
		try:
			start_training(jid, prompt.strip(), dataset_id.strip())
			status_box.update(label=f"Job {jid} started", state="complete")
		except Exception as e:
			status_box.update(label=f"Failed to start job: {e}", state="error")

if attach_btn and existing_job_id.strip():
	st.session_state.job_id = existing_job_id.strip()

job_id = st.session_state.job_id

if not job_id:
	st.info("Start a new training job or attach to an existing one from the sidebar.")
	st.stop()

st.subheader(f"Job: {job_id}")

col1, col2 = st.columns([1, 1])
status_placeholder = st.empty()
code_placeholder = st.empty()
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# Live update loop
max_minutes = 8
poll_interval_sec = 2
end_time = time.time() + max_minutes * 60
shown_balloons = False

while True:
	# Status
	status_text = poll_text(job_id, "status.json")
	status_json = parse_json(status_text) or {}
	status_str = status_json.get("status", "unknown")
	status_placeholder.write(f"**Status**: {status_str}")

	# Code (if available)
	code_text = poll_text(job_id, "code.py")
	if code_text:
		with col1:
			st.markdown("**Generated Code**")
			st.code(code_text, language="python")

	# Loss curve (if available)
	losses_csv = poll_text(job_id, "losses.csv") or poll_text(job_id, "loss.csv")
	if losses_csv:
		try:
			df = pd.read_csv(StringIO(losses_csv))
			if {"step", "train_loss"}.issubset(df.columns):
				with col2:
					st.markdown("**Training/Validation Loss**")
					plot_df = df[["step", "train_loss"]].copy()
					plot_df = plot_df.rename(columns={"train_loss": "Train Loss"})  # type: ignore[attr-defined]
					if "val_loss" in df.columns:
						plot_df["Val Loss"] = df["val_loss"]
					plot_df = plot_df.set_index("step")
					st.line_chart(plot_df)

					# Also show accuracy if available
					if "val_accuracy" in df.columns:
						acc_df = df[["step", "val_accuracy"]].copy()
						acc_df = acc_df.rename(columns={"val_accuracy": "Validation Accuracy"})
						acc_df = acc_df.set_index("step")
						st.markdown("**Validation Accuracy**")
						st.line_chart(acc_df)
		except Exception:
			pass

	# Metrics (if available)
	metrics_text = poll_text(job_id, "metrics.json")
	metrics_json = parse_json(metrics_text)
	if metrics_json:
		metrics_placeholder.markdown("**Final Metrics**")
		metrics_placeholder.json(metrics_json)
		if not shown_balloons:
			st.balloons()
			st.snow()
			shown_balloons = True
			break

	# Exit conditions
	if time.time() > end_time:
		st.warning("Timed out while waiting for training to complete.")
		break

	time.sleep(poll_interval_sec)