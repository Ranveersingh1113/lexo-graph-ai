# Project Structure

This document describes the organization of the lexo-graph-ai project.

## Directory Structure

```
lexo-graph-ai/
├── api/                          # Phase 4: FastAPI application (to be created)
├── config/                       # Configuration files
│   ├── ocr_config.yaml          # OCR/Stage 2 configuration
│   ├── ocr_config.yaml.template # Template for OCR config
│   ├── stage3_config.yaml       # Stage 3 (Tables/Figures) configuration
│   ├── google_cloud_credentials.json.template  # Template (credentials not in git)
│   └── credentials.json.template # Template (credentials not in git)
├── data/                         # Dataset directory (not in git)
│   └── ps05_coco/               # PS-05 dataset in COCO format
│       ├── annotations/          # Annotation files
│       └── images/              # Image files (not in git)
├── docs/                         # Documentation (if needed)
├── frontend/                     # React frontend application
├── notebooks/                    # Jupyter notebooks for exploration
├── PaddleDetection/              # PaddleDetection framework (cloned)
├── scripts/                      # Data processing and pipeline scripts
│   ├── Phase 1: Layout Detection
│   │   ├── export_model.py      # Export trained model to inference format
│   │   └── run_stage1.py        # Stage 1 inference script
│   ├── Phase 2: OCR
│   │   └── run_stage1_and_2.py  # Combined Stage 1 + Stage 2
│   ├── Phase 3: Tables & Figures
│   │   ├── evaluate_phase3_models.py  # Model evaluation script
│   │   └── create_test_dataset.py     # Test dataset creation
│   ├── Phase 4: Complete Pipeline
│   │   └── run_complete_pipeline.py   # Complete pipeline (all stages)
│   ├── Data Processing
│   │   ├── convert_ps05_to_coco.py    # Convert PS05 to COCO format
│   │   ├── augment_data.py            # Data augmentation
│   │   ├── merge_coco_datasets.py     # Merge multiple COCO datasets
│   │   └── download_public_data.py    # Download public datasets
│   ├── Training
│   │   ├── train_ps05.py              # Training script wrapper
│   │   └── train_with_ps05_only.py    # Dataset statistics checker
│   ├── Google Drive Integration
│   │   ├── upload_images_to_gdrive.py
│   │   ├── download_images_from_gdrive.py
│   │   └── push_images_in_batches.py
│   └── Setup & Testing
│       └── test_google_vision_setup.py # Test Google Cloud Vision setup
├── src/                          # Source code modules
│   ├── preprocessing/            # Image preprocessing
│   │   └── deskew.py            # Image de-skewing
│   ├── stage2/                   # Stage 2: OCR
│   │   ├── image_cropper.py     # Crop text regions
│   │   ├── ocr_google.py        # Google Cloud Vision OCR client
│   │   └── ocr_pipeline.py      # OCR pipeline orchestrator
│   ├── stage3/                   # Stage 3: Tables & Figures
│   │   ├── table_processing.py  # Table structure recognition
│   │   ├── figure_processing.py # Figure captioning
│   │   ├── pipeline.py          # Stage 3 pipeline
│   │   └── evaluation.py        # Evaluation framework
│   └── utils/                    # Utility functions
│       └── output_formatter.py  # Output formatting utilities
├── .gitignore                    # Git ignore rules
├── README.md                     # Main project README
├── SETUP_CREDENTIALS.md          # Credentials setup guide
├── SETUP_GOOGLE_CLOUD_VISION.md  # Google Cloud Vision setup
├── PHASE1_INFERENCE_README.md    # Phase 1 documentation
├── PHASE2_OCR_README.md          # Phase 2 documentation
├── PHASE3_MODELS_EXPLANATION.md  # Phase 3 model explanations
├── PHASE3_EVALUATION_README.md   # Phase 3 evaluation guide
├── PHASE3_IMPLEMENTATION_SUMMARY.md  # Phase 3 summary
└── PHASE4_PLAN.md                # Phase 4 plan
```

## Key Directories

### `/scripts`
All executable scripts for data processing, training, and inference.

### `/src`
Reusable Python modules organized by stage.

### `/config`
Configuration files for different stages. **Credentials are excluded from git.**

### `/data`
Dataset storage (excluded from git due to size).

## Important Files

### Configuration Files
- `config/ocr_config.yaml` - OCR configuration (Stage 2)
- `config/stage3_config.yaml` - Table/Figure processing config (Stage 3)
- `config/*.template` - Template files (safe to commit)

### Credentials
- `config/google_cloud_credentials.json` - **NOT in git** (use template)
- `config/credentials.json` - **NOT in git** (use template)
- See `SETUP_CREDENTIALS.md` for setup instructions

### Documentation
- Phase-specific READMEs in root directory
- Setup guides for each component
- Inline code documentation

## File Naming Conventions

- **Scripts**: `lowercase_with_underscores.py`
- **Modules**: `lowercase_with_underscores.py`
- **Configs**: `lowercase_with_underscores.yaml`
- **Documentation**: `UPPERCASE_WITH_UNDERSCORES.md` or `PascalCase.md`

## Git Organization

### Committed to Git
- All source code (`src/`, `scripts/`)
- Configuration templates (`config/*.template`, `config/*.yaml`)
- Documentation (`.md` files)
- Project structure files

### Excluded from Git
- Credentials (`config/*.json` except templates)
- Data files (`data/ps05_coco/images/`)
- Model outputs (`output/`, `*.pdparams`)
- Training logs (`training_logs/`)
- Large datasets

## Setup After Clone

1. Set up credentials (see `SETUP_CREDENTIALS.md`)
2. Install dependencies
3. Configure paths in config files
4. Download/place data files
5. Test setup with provided test scripts

