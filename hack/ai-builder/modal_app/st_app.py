# pyright: reportMissingTypeStubs=false
import os
import sys
import time
import json
import base64
from io import StringIO, BytesIO
from typing import Optional, Any, cast

import streamlit as st
import pandas as pd  # type: ignore
import modal
import openai

# Ensure we can import the local package `modal_app`
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# if CURRENT_DIR not in sys.path:
# 	sys.path.append(CURRENT_DIR)

# # Import the remote train function from our Modal app
# try:
# except Exception as import_err:  # Fallback to inform the user
# 	train = None  # type: ignore
# 	st.error(f"Failed to import train function from modal_app.main: {import_err}")

APP_NAME = "AI Builder - Training Dashboard"
VOLUME_NAME = "ai-builder-models"
MODELS_DIR = "/models"

def generate_code_with_streaming(prompt: str, dataset_summary: dict, job_id: str) -> str:
    """Generate code using OpenAI with streaming response."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    enhanced_input = f"""
USER PROMPT:
{prompt}

DATASET SUMMARY:
{json.dumps(dataset_summary, indent=2)}

Please use the dataset summary above to understand the data structure, features, and available splits. The summary includes:
- Dataset features and their types
- Available splits (train, test, validation) with example counts
- Sample data examples
- Dataset metadata and task information

Use this information to create appropriate data loaders, preprocessing, and model architecture.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""# You are an expert machine learning engineer and scientist.
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
# DO NOT OUTPUT MARKDOWN. OUTPUT ONLY THE CODE.
# """
            },
            {
                "role": "user",
                "content": enhanced_input
            }
        ],
        temperature=0.1,
        max_tokens=4000
    )
    
    # Get the generated code from the response
    if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
        content = response.choices[0].message.content
        if content:
            return content
        else:
            raise RuntimeError("Generated code is empty")
    else:
        raise RuntimeError("Failed to generate code from OpenAI response")

# Note: OpenAI API key must be set in environment variable OPENAI_API_KEY
# You can set it in your shell or create a .env file


def poll_text(job_id: str, filename: str) -> str:
	"""Fetch text artifact via Modal; returns empty string if not found."""
	try:
		read_file_fn: Any = modal.Function.lookup("ai-builder-reader", "read_file")
		result = read_file_fn.remote(job_id, filename)  # type: ignore[attr-defined]
		return str(result) if result is not None else ""
	except Exception:
		return ""


def parse_json(text: str) -> Optional[dict]:
	try:
		return json.loads(text) if text else None
	except Exception:
		return None


st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("AI Builder - Training Dashboard")

with st.sidebar:
	st.header("Start a new job")
	
	# Check if OpenAI API key is available
	if not os.environ.get("OPENAI_API_KEY"):
		st.error("⚠️ OpenAI API key not found! Set OPENAI_API_KEY environment variable.")
		st.info("You can set it in your shell or create a .env file")
		st.stop()
	
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
	
	# Step 1: Create dataset summary
	with st.status("Creating dataset summary...", expanded=True) as status_box:
		try:
			create_summary_fn: Any = modal.Function.lookup("ai-builder", "create_dataset_summary")
			summary = create_summary_fn.remote(dataset_id.strip())
			status_box.update(label="Dataset summary created", state="complete")
			
			# Step 2: Generate code
			with st.status("Generating code...", expanded=True) as code_status:
				try:
					generated_code = generate_code_with_streaming(prompt.strip(), summary, jid)
					code_status.update(label="Code generated", state="complete")
					
					# Display the generated code
					st.subheader("Generated Code")
					st.code(generated_code, language="python")
					
					# Step 3: Start training
					with st.status("Starting training job...", expanded=True) as train_status:
						try:
							train_fn: Any = modal.Function.lookup("ai-builder", "train")
							train_fn.spawn(jid, generated_code)
							train_status.update(label=f"Job {jid} started", state="complete")
						except Exception as e:
							train_status.update(label=f"Failed to start training: {e}", state="error")
							st.error(f"Training failed: {e}")
							
				except Exception as e:
					code_status.update(label=f"Failed to generate code: {e}", state="error")
					st.error(f"Code generation failed: {e}")
					
		except Exception as e:
			status_box.update(label=f"Failed to create dataset summary: {e}", state="error")
			st.error(f"Dataset summary creation failed: {e}")

if attach_btn and existing_job_id.strip():
	st.session_state.job_id = existing_job_id.strip()

job_id = st.session_state.job_id

if not job_id:
	st.info("Start a new training job or attach to an existing one from the sidebar.")
	st.stop()

st.subheader(f"Job: {job_id}")

# Panel switch
with st.sidebar:
	panel = st.radio("Panel", ["Training", "Inference"], index=0, key="panel_selector")

if panel == "Inference":
	st.markdown("**Run Inference**")
	
	# Input type selector
	input_type = st.selectbox(
		"Input Type", 
		["Text", "Image", "Audio"], 
		key="input_type_selector"
	)
	
	# Show help text based on input type
	if input_type == "Text":
		st.info("💡 Text input is best for sentiment analysis, classification, and text generation tasks.")
	elif input_type == "Image":
		st.info("💡 Image input is best for image classification, object detection, and image-to-text tasks.")
	elif input_type == "Audio":
		st.info("💡 Audio input is best for speech recognition, audio classification, and audio-to-text tasks.")
	
	infer_result = None
	
	if input_type == "Text":
		user_input = st.text_area("Input text", value="I loved this movie!", height=100, key="inference_text")
		infer_btn = st.button("Predict", type="secondary", key="predict_button")
		
		if infer_btn:
			try:
				predict_fn: Any = modal.Function.lookup("ai-builder", "predict_internal")
				infer_result = predict_fn.remote(job_id, {"text": user_input})
			except Exception as e:
				infer_result = {"error": str(e)}
	
	elif input_type == "Image":
		st.markdown("**Upload an image for classification**")
		uploaded_image = st.file_uploader(
			"Choose an image file", 
			type=['png', 'jpg', 'jpeg', 'gif', 'bmp'], 
			key="image_uploader"
		)
		
		if uploaded_image is not None:
			# Check file size (limit to 10MB)
			file_size = len(uploaded_image.read())
			uploaded_image.seek(0)  # Reset file pointer
			
			if file_size > 10 * 1024 * 1024:  # 10MB limit
				st.error("File too large! Please upload an image smaller than 10MB.")
			else:
				# Display the uploaded image
				st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
				st.info(f"Image file: {uploaded_image.name} ({file_size} bytes)")
				
				# Convert to base64
				image_bytes = uploaded_image.read()
				image_base64 = base64.b64encode(image_bytes).decode('utf-8')
				
				infer_btn = st.button("Predict on Image", type="secondary", key="predict_image_button")
				
				if infer_btn:
					try:
						predict_fn_img: Any = modal.Function.lookup("ai-builder", "predict_internal")
						infer_result = predict_fn_img.remote(job_id, {"image_base64": image_base64})
					except Exception as e:
						infer_result = {"error": str(e)}
	
	elif input_type == "Audio":
		st.markdown("**Upload an audio file for classification**")
		uploaded_audio = st.file_uploader(
			"Choose an audio file", 
			type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'], 
			key="audio_uploader"
		)
		
		if uploaded_audio is not None:
			# Check file size (limit to 50MB for audio)
			file_size = len(uploaded_audio.read())
			uploaded_audio.seek(0)  # Reset file pointer
			
			if file_size > 50 * 1024 * 1024:  # 50MB limit
				st.error("File too large! Please upload an audio file smaller than 50MB.")
			else:
				# Display audio player
				st.audio(uploaded_audio, format=f'audio/{uploaded_audio.type.split("/")[-1]}')
				
				# Show file info
				st.info(f"Audio file: {uploaded_audio.name} ({file_size} bytes)")
				
				# Convert to base64
				audio_bytes = uploaded_audio.read()
				audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
				
				infer_btn = st.button("Predict on Audio", type="secondary", key="predict_audio_button")
				
				if infer_btn:
					try:
						predict_fn_audio: Any = modal.Function.lookup("ai-builder", "predict_internal")
						infer_result = predict_fn_audio.remote(job_id, {"audio_base64": audio_base64})
					except Exception as e:
						infer_result = {"error": str(e)}
	
	# Display results
	if infer_result is not None:
		st.markdown("**Inference Results**")
		st.json(infer_result)
else:
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

		# Code (if available) with expand/collapse
		code_text = poll_text(job_id, "code.py")
		if code_text:
			with col1:
				with code_placeholder.container():
					st.markdown("**Generated Code**")
					expanded_key = f"code_expanded_{job_id}"
					expanded = bool(st.session_state.get(expanded_key, False))
					if not expanded:
						if st.button("Show full code", key=f"expand_code_{job_id}"):
							st.session_state[expanded_key] = True
						preview_lines = 30
						lines = code_text.splitlines()
						preview = "\n".join(lines[:preview_lines])
						if len(lines) > preview_lines:
							preview += "\n# ... (truncated)"
						st.code(preview, language="python")
					else:
						if st.button("Show less", key=f"collapse_code_{job_id}"):
							st.session_state[expanded_key] = False
						st.code(code_text, language="python")

		# Loss curve (if available)
		losses_csv = poll_text(job_id, "losses.csv") or poll_text(job_id, "loss.csv")
		print("losses_csv", losses_csv)
		if losses_csv:
			try:
				df = pd.read_csv(StringIO(losses_csv))
				if {"step", "train_loss"}.issubset(df.columns):
					with col2:
						with chart_placeholder.container():
							plot_df = df[["step", "train_loss"]].copy()
							plot_df = cast(pd.DataFrame, plot_df).rename(columns={"train_loss": "Train Loss"})
							if "val_loss" in df.columns:
								plot_df["Val Loss"] = df["val_loss"]
							plot_df = plot_df.set_index("step")

							acc_df = None
							if "val_accuracy" in df.columns:
								acc_df = df.loc[:, ["step", "val_accuracy"]].copy()
								acc_df = cast(pd.DataFrame, acc_df).rename(columns={"val_accuracy": "Validation Accuracy"})
								acc_df = acc_df.set_index("step")

							if acc_df is not None:
								loss_tab, acc_tab = st.tabs(["Loss", "Accuracy"])
								with loss_tab:
									st.markdown("**Training/Validation Loss**")
									st.line_chart(plot_df)
								with acc_tab:
									st.markdown("**Validation Accuracy**")
									st.line_chart(acc_df)
							else:
								st.markdown("**Training/Validation Loss**")
								st.line_chart(plot_df)
			except Exception:
				pass

		# Metrics (if available)
		metrics_text = poll_text(job_id, "metrics.json")
		metrics_json = parse_json(metrics_text)
		if metrics_json:
			metrics_placeholder.markdown("**Final Metrics**")
			metrics_placeholder.json(metrics_json)
			if not shown_balloons:
				# st.balloons()
				# st.snow()
				shown_balloons = True
				break

		# Exit conditions
		if time.time() > end_time:
			st.warning("Timed out while waiting for training to complete.")
			break

		time.sleep(poll_interval_sec)