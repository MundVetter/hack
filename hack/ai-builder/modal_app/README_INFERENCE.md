# AI Builder - Multi-Modal Inference Support

This document describes the enhanced inference capabilities of the AI Builder application, which now supports multiple input types beyond just text.

## Supported Input Types

### 1. Text Input
- **Best for**: Sentiment analysis, text classification, text generation, summarization
- **Input**: Direct text input via text area
- **Example**: "I loved this movie!" for sentiment analysis

### 2. Image Input
- **Best for**: Image classification, object detection, image-to-text, image segmentation
- **Supported formats**: PNG, JPG, JPEG, GIF, BMP
- **File size limit**: 10MB
- **Processing**: Images are automatically converted to base64 format for model inference

### 3. Audio Input
- **Best for**: Speech recognition, audio classification, audio-to-text
- **Supported formats**: WAV, MP3, FLAC, OGG, M4A, AAC
- **File size limit**: 50MB
- **Processing**: Audio files are automatically converted to base64 format for model inference

## How to Use

1. **Start the Streamlit app**: Run `streamlit run st_app.py`
2. **Select the Inference panel** from the sidebar
3. **Choose your input type** from the dropdown (Text, Image, or Audio)
4. **Upload your file** (for Image or Audio) or enter text
5. **Click the Predict button** to run inference
6. **View results** in the JSON format below

## Technical Details

### Backend Changes
- `predict_internal` function now supports `audio_base64` parameter
- Automatic task detection based on input type:
  - Text inputs → `text-classification` (default)
  - Image inputs → `image-classification`
  - Audio inputs → `automatic-speech-recognition`
- Pipeline creation optimized for each task type

### Frontend Changes
- Input type selector with helpful guidance
- File upload with size validation
- Base64 encoding for binary files
- Error handling and user feedback
- File information display

## Model Compatibility

The inference system automatically detects the appropriate task type and creates the corresponding Hugging Face pipeline:
- **Text models**: Use tokenizer + model
- **Image models**: Use model only (no tokenizer needed)
- **Audio models**: Use model only (no tokenizer needed)

## Error Handling

- File size validation prevents oversized uploads
- Graceful error handling for inference failures
- User-friendly error messages
- File format validation

## Future Enhancements

Potential improvements could include:
- Support for video files
- Batch processing capabilities
- Real-time streaming for audio
- Model performance metrics
- Custom preprocessing options
