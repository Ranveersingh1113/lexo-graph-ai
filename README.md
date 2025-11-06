# Intelligent Multilingual Document Understanding (PS-05)

A comprehensive solution for complete document understanding including layout detection, OCR, table extraction, and figure captioning. Designed for the PS-05 competition challenge.

## 📋 Project Overview

This project implements a complete document understanding pipeline with four stages:

### Stage 1: Document Layout Analysis (DLA)
- Detects and classifies document elements into 6 categories:
  - **0: Background**
  - **1: Text**
  - **2: Title**
  - **3: List**
  - **4: Table**
  - **5: Figure**
- Uses PaddleDetection's PP-YOLOE model
- Includes image de-skewing preprocessing

### Stage 2: Multilingual OCR
- Extracts text from detected text regions
- Supports 100+ languages via Google Cloud Vision API
- Auto-detects language per region
- Handles multilingual documents

### Stage 3: Table & Figure Processing
- **Tables**: Structure recognition + natural language summarization
- **Figures**: Automatic caption generation using visual language models
- Multiple model options for evaluation and comparison

### Stage 4: API Integration (In Progress)
- FastAPI REST API for complete pipeline
- Natural language descriptions + structured JSON output
- Batch processing support

## 🏗️ Project Structure

```
lexo-graph-ai/
├── src/                      # Source code modules
│   ├── preprocessing/        # Image preprocessing (de-skewing)
│   ├── stage2/              # OCR pipeline (Google Cloud Vision)
│   ├── stage3/              # Table & Figure processing
│   └── utils/               # Utility functions
├── scripts/                  # Executable scripts
│   ├── Phase 1: export_model.py, run_stage1.py
│   ├── Phase 2: run_stage1_and_2.py
│   ├── Phase 3: evaluate_phase3_models.py, create_test_dataset.py
│   ├── Phase 4: run_complete_pipeline.py
│   ├── Data: convert_ps05_to_coco.py, augment_data.py, etc.
│   └── Training: train_ps05.py
├── config/                   # Configuration files
│   ├── ocr_config.yaml      # OCR configuration
│   ├── stage3_config.yaml   # Stage 3 configuration
│   └── *.template           # Credential templates
├── frontend/                 # React frontend application
├── PaddleDetection/          # PaddleDetection framework
├── data/                     # Dataset directory (not in git)
└── docs/                     # Documentation files
```

See `PROJECT_STRUCTURE.md` for detailed structure.

## ✅ Completed Work

### Phase 0: Setup & Data Unification ✓

1. **Environment Setup** ✓
   - Created conda environment `doc-comp` with Python 3.10
   - Installed PaddlePaddle, PaddleDetection, PyTorch (CUDA 11.8)
   - Installed Augraphy, HuggingFace Hub, OpenCV, and other dependencies
   - Fixed compatibility issues (NumPy 1.26.4, SciPy 1.13.1)

2. **Data Conversion** ✓
   - Converted PS05 dataset from paired PNG+JSON format to COCO format
   - **Result**: 4,000 images, 40,667 annotations
   - **Output**: `data/ps05_coco/annotations/train.json` and `data/ps05_coco/images/`

3. **Data Augmentation** ✓
   - Fixed Augraphy imports and pipeline configuration
   - Script ready: `scripts/augment_data.py`
   - Will double dataset to ~8,000 images when executed

### Dataset Statistics

```
Category Distribution:
  Text (id=1):     29,131 annotations
  Title (id=2):     8,501 annotations
  Table (id=4):     1,756 annotations
  Figure (id=5):    1,132 annotations
  List (id=3):        147 annotations
  Background (id=0):    0 annotations
```

### Phase 1: Layout Detection & Inference ✓
- Model export script
- De-skewing preprocessing
- Inference pipeline with output formatting
- See `PHASE1_INFERENCE_README.md`

### Phase 2: Multilingual OCR ✓
- Google Cloud Vision API integration
- Image cropping from detected regions
- OCR pipeline with language detection
- See `PHASE2_OCR_README.md`

### Phase 3: Table & Figure Processing ✓
- Multiple TSR models (Table-Transformer, PaddleOCR)
- Multiple figure captioning models (BLIP-2, LLaVA, BLIP)
- Evaluation framework for model comparison
- See `PHASE3_MODELS_EXPLANATION.md` and `PHASE3_EVALUATION_README.md`

### Phase 4: API Integration (In Progress)
- FastAPI REST API
- Natural language descriptions
- Complete pipeline integration
- See `PHASE4_PLAN.md`

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd lexo-graph-ai

# Create conda environment
conda create -n doc-comp python=3.10 -y
conda activate doc-comp

# Install dependencies
pip install paddlepaddle-gpu
pip install google-cloud-vision pyyaml
pip install transformers torch torchvision
pip install opencv-python tqdm pandas
```

### 2. Set Up Credentials

**Google Cloud Vision API:**
1. Follow `SETUP_GOOGLE_CLOUD_VISION.md`
2. Download credentials JSON file
3. Save to `config/google_cloud_credentials.json`
4. See `SETUP_CREDENTIALS.md` for details

### 3. Configure

Update configuration files:
- `config/ocr_config.yaml` - OCR settings
- `config/stage3_config.yaml` - Table/Figure models

### 4. Test Setup

```bash
# Test Google Cloud Vision setup
python scripts/test_google_vision_setup.py --credentials config/google_cloud_credentials.json
```

### 5. Run Pipeline

**After model training:**
```bash
# Export model
python scripts/export_model.py -c <config> -w <weights>

# Run complete pipeline
python scripts/run_complete_pipeline.py \
    --model_dir models/inference/ppyoloe_ps05 \
    --image_path document.png \
    --output_path result.json
```

## 📚 Documentation

- **Setup**: `SETUP_CREDENTIALS.md`, `SETUP_GOOGLE_CLOUD_VISION.md`
- **Phase 1**: `PHASE1_INFERENCE_README.md`
- **Phase 2**: `PHASE2_OCR_README.md`
- **Phase 3**: `PHASE3_MODELS_EXPLANATION.md`, `PHASE3_EVALUATION_README.md`
- **Phase 4**: `PHASE4_PLAN.md`
- **Structure**: `PROJECT_STRUCTURE.md`

## 🔐 Credentials Setup

**Important**: Credentials are excluded from git for security.

**After cloning:**
1. Copy your Google Cloud credentials to `config/google_cloud_credentials.json`
2. See `SETUP_CREDENTIALS.md` for detailed instructions
3. Template files are provided in `config/*.template`

## 🚀 Next Steps

1. **Configure PaddleDetection**
   - Create config file for PP-DocLayout (RT-DETR) model
   - Point to `data/ps05_coco/annotations/train.json`
   - Set up training parameters (batch size, learning rate, epochs)

2. **Run Training**
   ```bash
   conda activate doc-comp
   cd PaddleDetection
   python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
       --dataset_dir ../data/ps05_coco
   ```

3. **Model Export**
   - Export best checkpoint to static inference model
   - Prepare for deployment

4. **Create Inference Script**
   - Build `run_stage1.py` script
   - Implement de-skewing
   - Format output as JSON: `{class: int, bbox: [x, y, h, w]}`

### Phase 2+: Future Work

- Multilingual OCR integration
- Multimodal content generation (Table/Figure processing)
- Final API integration

## 🛠️ Setup Instructions

### Prerequisites

- Windows/Linux with CUDA 11.8 compatible GPU
- Conda (Miniconda or Anaconda)
- Git

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd lexo-graph-ai

# 2. Create conda environment
conda create -n doc-comp python=3.10 -y
conda activate doc-comp

# 3. Install PaddlePaddle (Windows)
python -m pip install paddlepaddle

# 4. Install PaddleDetection dependencies
cd PaddleDetection
python -m pip install -r requirements.txt
python -m pip install -e .
cd ..

# 5. Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 6. Install additional packages
pip install augraphy huggingface_hub datasets opencv-python pandas tqdm scikit-learn

# 7. Fix compatibility
pip install "numpy==1.26.4" --force-reinstall --no-deps
pip install "scipy==1.13.1" --force-reinstall
```

### Data Preparation

```bash
# 1. Convert PS05 to COCO format
python scripts/convert_ps05_to_coco.py

# 2. (Optional) Augment data
python scripts/augment_data.py

# 3. Check dataset statistics
python scripts/train_with_ps05_only.py
```

## 📝 Scripts Description

| Script | Purpose | Status |
|--------|---------|--------|
| `convert_ps05_to_coco.py` | Convert PS05 dataset to COCO format | ✓ Ready |
| `augment_data.py` | Apply Augraphy augmentation to PS05 dataset | ✓ Ready |
| `download_public_data.py` | Download DocLayNet/PubLayNet (optional) | ✓ Ready |
| `merge_coco_datasets.py` | Merge multiple COCO datasets | ✓ Ready |
| `train_ps05.py` | Training script wrapper with validation | ✓ Ready |
| `train_with_ps05_only.py` | Display PS05 dataset statistics | ✓ Ready |

## 🔧 Configuration Notes

### Current Setup
- **Dataset**: PS05 only (~4,000 images, can be augmented to ~8,000)
- **Format**: COCO format
- **Location**: `data/ps05_coco/`
- **Model**: PaddleDetection PP-DocLayout (RT-DETR)

### Optional: Adding Public Datasets
If you have sufficient disk space (~200GB), you can:
1. Run `python scripts/download_public_data.py` to download DocLayNet/PubLayNet
2. Run `python scripts/merge_coco_datasets.py` to merge all datasets
3. This will significantly increase dataset size (~100,000+ images)

## 📊 Performance Considerations

- **Training Time**: Estimated 4-8 hours on GPU (depends on GPU specs)
- **Disk Space**: 
  - PS05 only: ~10 GB
  - With public datasets: ~200-250 GB
- **Recommended**: Use GPU lab with high-end GPUs for faster training

## 🐛 Known Issues & Solutions

1. **SciPy/NumPy Compatibility**: Fixed by using SciPy 1.13.1 with NumPy 1.26.4
2. **Augraphy Imports**: Fixed by using correct class names (not `*Pipeline` classes)
3. **Windows Conda Activation**: Use `conda run -n doc-comp` if activation doesn't work

## 📚 References

- [PaddleDetection Documentation](https://github.com/PaddlePaddle/PaddleDetection)
- [Augraphy Documentation](https://github.com/sparkfish/augraphy)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

## 🤝 Contributing

This is a competition project. For questions or issues, please refer to the project documentation.

## 📄 License

See LICENSE file for details.

---

**Status**: Phase 0 Complete ✓ | Phase 1 Ready for GPU Lab Training

**Last Updated**: November 2025