# Phase 2: Multilingual OCR Integration

This document describes the OCR integration for Stage 2 of the PS-05 document understanding system.

## Overview

Phase 2 adds OCR (Optical Character Recognition) capabilities to extract text from document layout elements detected in Stage 1. It uses Google Cloud Vision API for multilingual text recognition.

## Components

### 1. Image Cropper (`src/stage2/image_cropper.py`)
Extracts text regions from images based on Stage 1 bounding boxes.

**Features:**
- Validates and clips bounding boxes to image boundaries
- Crops regions for specified categories (Text, Title, List)
- Batch processing support
- Optional saving of cropped images for debugging

### 2. Google Cloud Vision OCR Client (`src/stage2/ocr_google.py`)
Handles OCR using Google Cloud Vision API.

**Features:**
- Single image and batch OCR
- Multilingual support (auto-detect or specify languages)
- Document text detection (optimized for dense text)
- Confidence scores and language detection
- Error handling and retry logic

### 3. OCR Pipeline (`src/stage2/ocr_pipeline.py`)
Orchestrates the complete OCR workflow.

**Features:**
- Integrates cropping and OCR
- Processes Stage 1 outputs
- Combines layout detection and OCR results
- Handles errors gracefully

### 4. Combined Inference Script (`scripts/run_stage1_and_2.py`)
End-to-end pipeline: Layout Detection → OCR.

**Features:**
- Runs Stage 1 and Stage 2 together
- Single command for complete processing
- Batch processing support
- Configurable options

## Setup

### 1. Install Dependencies
```bash
pip install google-cloud-vision pyyaml
```

### 2. Configure Google Cloud Vision API

1. **Create Google Cloud Project** (if not done)
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable Cloud Vision API

2. **Create Service Account**
   - Go to "APIs & Services" → "Credentials"
   - Create service account
   - Download JSON key file

3. **Configure Credentials**
   - Save JSON key to: `config/google_cloud_credentials.json`
   - Or set environment variable: `GOOGLE_APPLICATION_CREDENTIALS`

4. **Update Config File**
   - Copy `config/ocr_config.yaml.template` to `config/ocr_config.yaml`
   - Update `credentials_path` in config file

### 3. Verify Setup
```bash
python scripts/test_google_vision_setup.py --credentials config/your-credentials.json
```

## Usage

### Option 1: Combined Pipeline (Recommended)

Run both Stage 1 and Stage 2 together:

```bash
# Single image
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path test_image.png \
    --output_path result.json \
    --config config/ocr_config.yaml

# Directory of images
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_dir test_images/ \
    --output_dir outputs/results/ \
    --config config/ocr_config.yaml
```

### Option 2: Separate Stages

**Step 1: Run Stage 1**
```bash
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path test_image.png \
    --output_path stage1_result.json
```

**Step 2: Run Stage 2 OCR**
```python
from src.stage2.ocr_pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline(config_path="config/ocr_config.yaml")

# Process Stage 1 output
result = pipeline.process_stage1_output(
    "stage1_result.json",
    image_dir="test_images/"
)

# Save results
import json
with open("stage2_result.json", 'w') as f:
    json.dump(result, f, indent=2)
```

### Configuration

Edit `config/ocr_config.yaml` to customize:

```yaml
google_cloud:
  credentials_path: "config/your-credentials.json"
  project_id: ""

ocr_settings:
  confidence_threshold: 0.5
  languages: []  # Empty = auto-detect
  auto_detect_language: true
  use_document_text_detection: true

processing:
  batch_size: 10
  max_retries: 3
  timeout_seconds: 30

categories:
  ocr_categories: [1, 2, 3]  # Text, Title, List
```

## Output Format

The combined output includes both layout detection and OCR results:

```json
{
  "image": "document.png",
  "stage1": {
    "original_shape": [1080, 1920],
    "processed_shape": [1080, 1920],
    "skew_angle": 0.5,
    "num_predictions": 5
  },
  "layout_elements": [
    {
      "class": 1,
      "bbox": [100, 200, 50, 100],
      "ocr": {
        "text": "This is extracted text content",
        "confidence": 0.95,
        "language": "en"
      }
    },
    {
      "class": 2,
      "bbox": [150, 250, 60, 80],
      "ocr": {
        "text": "This is a title",
        "confidence": 0.92,
        "language": "en"
      }
    },
    {
      "class": 4,
      "bbox": [200, 300, 100, 150],
      "ocr": null
    }
  ],
  "summary": {
    "total_elements": 5,
    "elements_with_ocr": 2,
    "elements_without_ocr": 3
  }
}
```

## Command-Line Arguments

### `run_stage1_and_2.py`

**Required:**
- `--model_dir`: Path to Stage 1 inference model directory
- `--image_path` OR `--image_dir`: Input image(s)

**Optional:**
- `--config`: Path to OCR config file (default: `config/ocr_config.yaml`)
- `--output_path`: Output JSON file (single image mode)
- `--output_dir`: Output directory (directory mode)
- `--score_threshold`: Layout detection confidence threshold (default: 0.5)
- `--no_deskew`: Disable de-skewing
- `--max_skew_angle`: Max skew angle to correct (default: 10.0)
- `--cpu`: Use CPU instead of GPU
- `--save_cropped`: Save cropped text regions for debugging

## Supported Languages

Google Cloud Vision API supports 100+ languages. Common ones for multilingual documents:

- **English**: `en`
- **Hindi**: `hi`
- **Bengali**: `bn`
- **Telugu**: `te`
- **Tamil**: `ta`
- **Marathi**: `mr`
- **Gujarati**: `gu`
- **Kannada**: `kn`
- **Malayalam**: `ml`
- **Punjabi**: `pa`
- **Urdu**: `ur`

Auto-detection is recommended for multilingual documents (set `auto_detect_language: true` in config).

## Cost Considerations

### Google Cloud Vision API Pricing

- **Free Tier**: First 1,000 units/month
- **1,001-5,000,000 units**: $1.50 per 1,000 units
- **5,000,001+ units**: $0.60 per 1,000 units

*1 image = 1 unit (regardless of text density)*

### Cost Optimization Tips

1. **Batch Processing**: Process multiple images in one run
2. **Cache Results**: Save OCR results to avoid re-processing
3. **Selective OCR**: Only run OCR on text regions (not tables/figures)
4. **Monitor Usage**: Check Google Cloud Console for usage stats

## Troubleshooting

### Error: "Could not automatically determine credentials"
**Solution**: 
- Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Or specify `credentials_path` in config file

### Error: "API not enabled"
**Solution**: 
- Enable Cloud Vision API in Google Cloud Console
- Verify service account has proper permissions

### Error: "Permission denied"
**Solution**: 
- Check service account permissions
- Verify credentials file is valid
- Ensure Cloud Vision API is enabled

### Low OCR Accuracy
**Solutions**:
- Ensure images are properly preprocessed (de-skewing helps)
- Use `document_text_detection` for dense text
- Specify language hints for better accuracy
- Check image quality (resolution, contrast)

### Rate Limiting
**Solution**: 
- Reduce batch size in config
- Add delays between API calls
- Use retry logic (already implemented)

## Examples

### Example 1: Single Image with Custom Config
```bash
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path document.png \
    --output_path result.json \
    --config config/ocr_config.yaml \
    --score_threshold 0.6 \
    --save_cropped
```

### Example 2: Batch Processing
```bash
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_dir documents/ \
    --output_dir outputs/ocr_results/ \
    --config config/ocr_config.yaml
```

### Example 3: CPU Mode (No GPU)
```bash
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path test.png \
    --output_path result.json \
    --cpu
```

## Integration with Phase 1

Phase 2 seamlessly integrates with Phase 1:

1. **Stage 1 Output**: Uses layout detection results from Phase 1
2. **Automatic Processing**: Combined script handles both stages
3. **Unified Output**: Single JSON file with both layout and OCR data

## Next Steps

After Phase 2 is complete, proceed to:
- **Phase 3**: Table and Figure processing (multimodal content generation)
- **Phase 4**: Final API integration (FastAPI)

## Files Created

- `src/stage2/image_cropper.py` - Image cropping utilities
- `src/stage2/ocr_google.py` - Google Cloud Vision API client
- `src/stage2/ocr_pipeline.py` - OCR pipeline orchestrator
- `scripts/run_stage1_and_2.py` - Combined inference script
- `config/ocr_config.yaml` - OCR configuration
- `scripts/test_google_vision_setup.py` - Setup verification script

## Support

- [Google Cloud Vision API Documentation](https://cloud.google.com/vision/docs)
- [Python Client Library](https://googleapis.dev/python/vision/latest/)
- [Pricing Information](https://cloud.google.com/vision/pricing)


