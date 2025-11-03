# Intelligent Multilingual Document Understanding (PS-05)

A comprehensive solution for document layout analysis using PaddleDetection, designed for the PS-05 competition challenge.

## 📋 Project Overview

This project implements **Stage 1: Document Layout Analysis (DLA)** for multilingual document understanding. The solution uses PaddleDetection's PP-DocLayout (RT-DETR) model to detect and classify document elements into 6 categories:

- **0: Background**
- **1: Text**
- **2: Title**
- **3: List**
- **4: Table**
- **5: Figure**

## 🏗️ Project Structure

```
lexo-graph-ai/
├── frontend/                 # React frontend application
├── PaddleDetection/          # PaddleDetection framework (cloned)
├── scripts/                  # Data processing and training scripts
│   ├── convert_ps05_to_coco.py      # Convert PS05 to COCO format
│   ├── augment_data.py              # Data augmentation with Augraphy
│   ├── download_public_data.py      # Download public datasets (optional)
│   ├── merge_coco_datasets.py       # Merge multiple COCO datasets
│   ├── train_ps05.py                # Training script wrapper
│   └── train_with_ps05_only.py      # Dataset statistics checker
├── data/                     # Dataset directory (not in git)
│   └── ps05_coco/           # Converted PS05 dataset in COCO format
├── train_PS05/              # Original PS05 dataset (not in git)
└── README.md                # This file
```

## ✅ Completed Work

### Phase 0: Setup & Data Unification

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

## 🚀 Next Steps

### Phase 1: Model Training (To be done on GPU lab)

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