# AI Builder - Updated Training Dashboard

## Overview

The AI Builder has been updated to separate code generation from code execution. Now the Streamlit app handles:
1. **Dataset Summary Creation** - Creates a summary of the dataset using the `create_dataset_summary` function
2. **Code Generation** - Uses OpenAI GPT-4 to generate ML training code based on the prompt and dataset summary
3. **Code Execution** - Calls the `train` function to execute the generated code in a sandboxed environment

## Key Changes

### Before (Old Workflow)
- The `train` function in `main.py` handled both code generation and execution
- Code generation happened remotely on Modal

### After (New Workflow)
- **Streamlit App** (`st_app.py`): Handles dataset summary creation and code generation
- **Modal Functions** (`main.py`): 
  - `create_dataset_summary`: Creates dataset summaries
  - `train`: Only executes pre-generated code
  - `predict_internal`: Handles inference

## Setup Requirements

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Dependencies
Install the Streamlit app dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

### 1. Start the Streamlit App
```bash
cd hack/ai-builder/modal_app
streamlit run st_app.py
```

### 2. Workflow
1. **Enter Prompt**: Describe what you want to build (e.g., "create a sentiment classifier for imdb dataset")
2. **Enter Dataset ID**: Specify the dataset (e.g., "imdb")
3. **Click "Start Training"**: The app will:
   - Create a dataset summary
   - Generate ML code using OpenAI
   - Display the generated code
   - Start the training job

### 3. Monitor Progress
- The app shows real-time status updates
- Generated code is displayed for review
- Training progress is monitored via the existing polling mechanism

## Benefits of New Architecture

1. **Better User Experience**: Users can see the generated code before training starts
2. **Streaming Code Generation**: Code generation happens in the Streamlit app with real-time feedback
3. **Separation of Concerns**: Code generation and execution are now separate
4. **Easier Debugging**: Users can review and modify generated code if needed
5. **Better Error Handling**: Clear separation between generation and execution errors

## File Structure

```
modal_app/
├── st_app.py              # Main Streamlit app (handles code generation)
├── main.py                # Modal functions (execution only)
├── dataset_sumarizer.py   # Dataset summary functionality
├── requirements_streamlit.txt  # Streamlit dependencies
└── README_UPDATED.md      # This file
```

## Troubleshooting

### OpenAI API Key Issues
- Ensure `OPENAI_API_KEY` is set in your environment
- The app will show an error if the key is missing

### Code Generation Failures
- Check your OpenAI API quota and billing
- Verify the prompt is clear and specific
- Ensure the dataset ID is valid

### Training Failures
- Check the generated code for syntax errors
- Verify all required packages are available in the Modal environment
- Check Modal logs for detailed error information
