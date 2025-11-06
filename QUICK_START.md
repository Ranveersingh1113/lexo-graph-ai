# Quick Start Guide

Get started with the document understanding pipeline in minutes.

## Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended)
- Google Cloud account (for OCR)
- Conda (recommended)

## Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd lexo-graph-ai
```

## Step 2: Set Up Environment

```bash
# Create conda environment
conda create -n doc-comp python=3.10 -y
conda activate doc-comp

# Install PaddlePaddle
pip install paddlepaddle-gpu  # For GPU
# OR
pip install paddlepaddle  # For CPU

# Install PaddleDetection
cd PaddleDetection
pip install -r requirements.txt
pip install -e .
cd ..

# Install other dependencies
pip install google-cloud-vision pyyaml
pip install transformers torch torchvision
pip install opencv-python tqdm pandas numpy
pip install augraphy
```

## Step 3: Set Up Credentials

### Google Cloud Vision API

1. **Get Credentials:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create project and enable Cloud Vision API
   - Create service account and download JSON key

2. **Save Credentials:**
   ```bash
   # Copy your downloaded JSON file to:
   config/google_cloud_credentials.json
   ```

3. **Test Setup:**
   ```bash
   python scripts/test_google_vision_setup.py --credentials config/google_cloud_credentials.json
   ```

See `SETUP_CREDENTIALS.md` for detailed instructions.

## Step 4: Prepare Data

### Option A: Use Existing Data

If you have data in Google Drive:
```bash
# Download from Google Drive
python scripts/download_images_from_gdrive.py <FOLDER_ID>
```

### Option B: Convert PS05 Dataset

```bash
# Convert PS05 to COCO format
python scripts/convert_ps05_to_coco.py

# (Optional) Augment data
python scripts/augment_data.py
```

## Step 5: Train Model (On Workstation)

```bash
# Train model
python scripts/train_ps05.py

# Export model after training
python scripts/export_model.py \
    -c PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml \
    -w PaddleDetection/output/ppyoloe_plus_crn_s_80e_ps05/model_final.pdparams \
    -o models/inference/ppyoloe_ps05
```

## Step 6: Run Inference

### Complete Pipeline (All Stages)

```bash
python scripts/run_complete_pipeline.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path document.png \
    --output_path result.json
```

### Stage 1 Only (Layout Detection)

```bash
python scripts/run_stage1.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path document.png \
    --output_path stage1_result.json
```

### Stage 1 + Stage 2 (Layout + OCR)

```bash
python scripts/run_stage1_and_2.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path document.png \
    --output_path result.json \
    --config config/ocr_config.yaml
```

## Step 7: Evaluate Models (Optional)

For Phase 3 model evaluation:

```bash
# Create test dataset
python scripts/create_test_dataset.py \
    --images_dir test_images/ \
    --output test_data.json

# Edit test_data.json and add ground truth

# Run evaluation
python scripts/evaluate_phase3_models.py \
    --test_data test_data.json \
    --output_dir evaluation_results/
```

## Troubleshooting

### Credentials Not Found
- Check file exists: `ls config/google_cloud_credentials.json`
- Verify path in `config/ocr_config.yaml`
- See `SETUP_CREDENTIALS.md`

### Model Not Found
- Export model first: `python scripts/export_model.py`
- Check model directory exists
- Verify config file paths

### Import Errors
- Install missing packages: `pip install <package>`
- Activate conda environment: `conda activate doc-comp`
- Check Python version: `python --version` (should be 3.10)

### GPU Issues
- Use `--cpu` flag for CPU inference
- Check GPU: `nvidia-smi`
- Verify CUDA installation

## Next Steps

- Read phase-specific documentation
- Test with sample images
- Evaluate models for Phase 3
- Prepare for Phase 4 API integration

## Support

- **Setup Issues**: See `SETUP_CREDENTIALS.md`
- **Phase 1**: See `PHASE1_INFERENCE_README.md`
- **Phase 2**: See `PHASE2_OCR_README.md`
- **Phase 3**: See `PHASE3_EVALUATION_README.md`
- **Phase 4**: See `PHASE4_PLAN.md`

