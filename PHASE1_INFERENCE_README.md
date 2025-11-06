# Phase 1: Document Layout Analysis (DLA) - Inference Pipeline

This document describes the complete inference pipeline for Stage 1 of the PS-05 document understanding system.

## Overview

The Phase 1 pipeline performs:
1. **Image De-skewing**: Corrects rotation/skew in document images
2. **Layout Detection**: Detects document elements (Text, Title, List, Table, Figure)
3. **Output Formatting**: Converts predictions to required JSON format: `{class: int, bbox: [x, y, h, w]}`

## Components

### 1. Model Export Script (`scripts/export_model.py`)
Exports trained PaddleDetection model to inference format.

**Usage:**
```bash
# Export with config and weights
python scripts/export_model.py \
    -c PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml \
    -w PaddleDetection/output/ppyoloe_plus_crn_s_80e_ps05/model_final.pdparams \
    -o models/inference/ppyoloe_ps05

# Or auto-detect weights from config
python scripts/export_model.py \
    -c PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml \
    -o models/inference/ppyoloe_ps05
```

**Output:**
- `model.pdmodel`: Model structure
- `model.pdiparams`: Model weights
- `model.pdiparams.info`: Weights metadata
- `infer_cfg.yml`: Inference configuration

### 2. De-skewing Module (`src/preprocessing/deskew.py`)
Corrects image rotation/skew before layout detection.

**Features:**
- Hough line transform for skew detection
- Projection profile analysis (alternative method)
- Automatic rotation correction

**Usage:**
```python
from src.preprocessing.deskew import deskew_image_file

# Deskew an image
deskewed_image, angle = deskew_image_file(
    "path/to/image.png",
    output_path="path/to/deskewed.png"
)
print(f"Corrected {angle:.2f} degrees")
```

### 3. Output Formatter (`src/utils/output_formatter.py`)
Converts model predictions to required JSON format.

**Features:**
- Bounding box format conversion (x1y1x2y2 ↔ xyhw)
- Score thresholding
- PaddleDetection output parsing

**Usage:**
```python
from src.utils.output_formatter import format_paddle_detection_output

# Format predictions
predictions = format_paddle_detection_output(
    paddle_output,
    score_threshold=0.5
)
# Returns: [{"class": 1, "bbox": [x, y, h, w]}, ...]
```

### 4. Main Inference Script (`scripts/run_stage1.py`)
Complete pipeline for layout detection inference.

**Usage:**

```bash
# Single image
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path path/to/image.png \
    --output_path output.json

# Directory of images
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_dir path/to/images/ \
    --output_dir outputs/stage1/

# With options
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path image.png \
    --score_threshold 0.6 \
    --no_deskew \
    --cpu
```

**Arguments:**
- `--model_dir`: Path to exported inference model directory
- `--image_path`: Single image file path
- `--image_dir`: Directory containing images (alternative to --image_path)
- `--output_path`: Output JSON file path (for single image)
- `--output_dir`: Output directory for JSON files (for directory mode)
- `--score_threshold`: Minimum confidence score (default: 0.5)
- `--no_deskew`: Disable de-skewing preprocessing
- `--max_skew_angle`: Maximum angle to correct (default: 10.0)
- `--cpu`: Use CPU instead of GPU

**Output Format:**
```json
{
  "image": "image.png",
  "original_shape": [1080, 1920],
  "processed_shape": [1080, 1920],
  "skew_angle": 0.5,
  "predictions": [
    {
      "class": 1,
      "bbox": [100, 200, 50, 100]
    },
    {
      "class": 2,
      "bbox": [150, 250, 60, 80]
    }
  ],
  "num_predictions": 2
}
```

## Workflow

### Step 1: Train Model (on Workstation)
```bash
# Train the model
python scripts/train_ps05.py
```

### Step 2: Export Model (on Workstation or Local)
```bash
# After training completes, export the model
python scripts/export_model.py \
    -c PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml \
    -w PaddleDetection/output/ppyoloe_plus_crn_s_80e_ps05/model_final.pdparams \
    -o models/inference/ppyoloe_ps05
```

### Step 3: Run Inference (Local)
```bash
# Single image
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path test_image.png \
    --output_path result.json

# Batch processing
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_dir test_images/ \
    --output_dir results/
```

## Category Mapping

The model detects 6 categories:
- **0**: Background
- **1**: Text
- **2**: Title
- **3**: List
- **4**: Table
- **5**: Figure

## Notes

1. **Model Format**: The exported model uses PaddleDetection's inference format, which is optimized for deployment.

2. **Preprocessing**: De-skewing is optional but recommended for rotated documents. The script automatically detects and corrects skew up to 10 degrees by default.

3. **Performance**: 
   - GPU inference is significantly faster than CPU
   - Batch processing is more efficient for multiple images
   - Consider adjusting `score_threshold` based on your accuracy needs

4. **Output Format**: The bbox format is `[x, y, h, w]` where:
   - `x, y`: Top-left corner coordinates
   - `h, w`: Height and width

5. **Compatibility**: Works with models exported from PaddleDetection 2.x. The script supports both Inference API and Trainer API (fallback).

## Troubleshooting

### Model not found
- Ensure the model was exported successfully
- Check that `model.pdmodel`, `model.pdiparams`, and `infer_cfg.yml` exist

### Import errors
- Make sure PaddleDetection is installed: `pip install -e PaddleDetection/`
- Verify PaddlePaddle is installed: `pip install paddlepaddle-gpu` (for GPU)

### GPU not available
- Use `--cpu` flag for CPU inference
- Check GPU with: `nvidia-smi`

### Low accuracy
- Adjust `--score_threshold` (lower = more detections, higher = fewer but more confident)
- Ensure images are properly preprocessed (de-skewing helps)
- Verify model was trained on similar data distribution

## Next Steps

After Phase 1 is complete, proceed to:
- **Phase 2**: OCR integration for text extraction
- **Phase 3**: Table and Figure processing
- **Phase 4**: Final API integration

