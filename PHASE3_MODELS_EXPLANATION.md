# Phase 3: Table & Figure Processing - Models Explanation

This document explains the models used in Phase 3 and how they work in the document understanding pipeline.

## Overview

Phase 3 processes two types of document elements detected in Stage 1:
1. **Tables** (Category ID: 4) - Extract structured data and generate summaries
2. **Figures** (Category ID: 5) - Generate descriptive captions

---

## Part 1: Table Processing Pipeline

### Architecture Overview

```
[Stage 1 Output] → [Crop Table Region] → [TSR Model] → [Structured Data] → [LLM] → [Summary]
```

### Models Used

#### 1. Table Structure Recognition (TSR) Model

**What it does:**
- Analyzes table images and recognizes the structure
- Identifies rows, columns, headers, and cell boundaries
- Extracts text content from each cell
- Outputs structured data (CSV/JSON format)

**Models Available:**

**A. Table-Transformer (Microsoft)**
- **Architecture**: Transformer-based, similar to DETR (Detection Transformer)
- **How it works**:
  1. Takes table image as input
  2. Uses vision transformer to understand table layout
  3. Detects table structure (rows, columns, cells)
  4. Extracts text from each cell using OCR
  5. Outputs structured table data
- **Output**: JSON with cells, rows, columns structure
- **Advantages**: High accuracy, handles complex layouts
- **Size**: ~500 MB

**B. PaddleOCR Table Recognition**
- **Architecture**: PaddleOCR's table recognition module
- **How it works**:
  1. Detects table region
  2. Recognizes table structure
  3. Extracts text from cells
  4. Outputs structured format
- **Output**: CSV or structured JSON
- **Advantages**: Fast, already integrated with PaddleOCR
- **Size**: ~200 MB (part of PaddleOCR)

**C. Alternative: TableNet (if needed)**
- Segmentation-based approach
- Less common but available

**How it works in Phase 3:**
```
Input: Cropped table image (from Stage 1 bbox)
↓
TSR Model processes image
↓
Output: Structured table data
{
  "headers": ["Column 1", "Column 2", ...],
  "rows": [
    ["Cell 1", "Cell 2", ...],
    ["Cell 3", "Cell 4", ...]
  ]
}
```

#### 2. Language Model (LLM) for Table Summarization

**What it does:**
- Takes structured table data as input
- Generates natural language summary
- Explains key insights, trends, or important information

**Models Available:**

**A. FLAN-T5 (Google)**
- **Architecture**: T5 (Text-to-Text Transfer Transformer) fine-tuned on instruction following
- **How it works**:
  1. Takes structured data as text input
  2. Uses encoder-decoder architecture
  3. Generates summary based on instructions
  4. Outputs natural language description
- **Sizes Available**:
  - `flan-t5-small`: ~80M parameters, ~300 MB
  - `flan-t5-base`: ~250M parameters, ~900 MB
  - `flan-t5-large`: ~780M parameters, ~3 GB
- **Advantages**: Good instruction following, fast inference
- **Use Case**: Generate summaries like "This table shows sales data for Q1 2024..."

**B. GPT-2 (OpenAI) - Alternative**
- Smaller, faster
- Less instruction-following capability
- Good for simple summaries

**C. BART (Facebook) - Alternative**
- Good for summarization tasks
- Medium size

**How it works in Phase 3:**
```
Input: Structured table data (JSON/CSV)
↓
LLM generates summary
↓
Output: "This table shows sales data for January 2024. 
         Total revenue was $50,000, with Product A 
         generating $20,000 (40%)..."
```

---

## Part 2: Figure Processing Pipeline

### Architecture Overview

```
[Stage 1 Output] → [Crop Figure Region] → [VLM Model] → [Caption/Description]
```

### Models Used

#### Visual Language Model (VLM) for Figure Captioning

**What it does:**
- Analyzes figure/chart/image
- Understands visual content
- Generates descriptive text caption
- Explains what the figure shows

**Models Available:**

**A. LLaVA (Large Language and Vision Assistant)**
- **Architecture**: Vision Transformer + Large Language Model
- **How it works**:
  1. Takes image as input
  2. Vision encoder extracts visual features
  3. Projects visual features to language space
  4. Language model generates caption based on visual understanding
  5. Outputs detailed description
- **Sizes Available**:
  - `llava-1.5-7b`: ~7B parameters, ~13 GB (requires GPU)
  - `llava-1.5-13b`: ~13B parameters, ~26 GB (requires GPU)
  - Smaller variants available (quantized)
- **Advantages**: 
  - Excellent understanding of visual content
  - Can describe charts, graphs, diagrams
  - Handles complex visual reasoning
- **Example Output**: "This bar chart shows the comparison of quarterly sales figures. Q1 has the highest sales at 45 units, followed by Q3 at 38 units..."

**B. BLIP-2 (Bootstrapping Language-Image Pre-training)**
- **Architecture**: Vision Transformer + Query Transformer + LLM
- **How it works**:
  1. Vision encoder processes image
  2. Query Transformer learns to extract relevant visual features
  3. LLM generates caption from visual queries
  4. Outputs natural language description
- **Sizes Available**:
  - `blip2-opt-2.7b`: ~2.7B parameters, ~5 GB
  - `blip2-opt-6.7b`: ~6.7B parameters, ~13 GB
- **Advantages**: 
  - Efficient architecture
  - Good for captioning
  - Faster than LLaVA
- **Use Case**: Good for charts, diagrams, photographs

**C. BLIP (Original)**
- **Architecture**: Similar to BLIP-2 but simpler
- **How it works**: Vision + Language fusion
- **Size**: ~1.5 GB
- **Advantages**: Smaller, faster, good for basic captioning

**D. GPT-4V / Claude 3 (API-based)**
- **Alternative**: Use cloud APIs (requires API keys)
- **Advantages**: Best quality, no local GPU needed
- **Disadvantages**: Cost per image, requires internet

**How it works in Phase 3:**
```
Input: Cropped figure image (chart, diagram, photo)
↓
VLM Model processes image
↓
Output: "This line graph shows temperature trends over 12 months. 
         The temperature peaks in July at 35°C and reaches its 
         lowest point in January at 5°C. There's a clear seasonal 
         pattern with gradual increases in spring and decreases in autumn."
```

---

## Complete Phase 3 Pipeline Flow

### For Tables (Category ID: 4)

```
1. Stage 1 detects table bbox: [x, y, h, w]
   ↓
2. Crop table region from original image
   ↓
3. TSR Model:
   - Processes table image
   - Detects structure (rows, columns)
   - Extracts cell text
   - Outputs structured data (JSON/CSV)
   ↓
4. LLM (FLAN-T5):
   - Takes structured data
   - Generates natural language summary
   - Outputs: "This table shows..."
   ↓
5. Final Output:
   {
     "class": 4,
     "bbox": [x, y, h, w],
     "table": {
       "structured_data": {...},
       "summary": "This table shows sales data..."
     }
   }
```

### For Figures (Category ID: 5)

```
1. Stage 1 detects figure bbox: [x, y, h, w]
   ↓
2. Crop figure region from original image
   ↓
3. VLM Model (LLaVA/BLIP-2):
   - Processes figure image
   - Understands visual content
   - Generates descriptive caption
   - Outputs: "This chart shows..."
   ↓
4. Final Output:
   {
     "class": 5,
     "bbox": [x, y, h, w],
     "figure": {
       "caption": "This bar chart shows quarterly sales...",
       "confidence": 0.95
     }
   }
```

---

## Model Selection Recommendations

### For Your RTX 4050 (16GB VRAM)

**Table Processing:**
- **TSR**: Table-Transformer (HuggingFace) - ~500 MB
- **LLM**: FLAN-T5-small - ~300 MB
- **Total**: ~800 MB ✅ Fits easily

**Figure Processing:**
- **Option 1**: BLIP-2-opt-2.7b - ~5 GB ✅ Good fit
- **Option 2**: LLaVA-1.5-7b (quantized) - ~7 GB ✅ Fits
- **Option 3**: BLIP (original) - ~1.5 GB ✅ Fits easily, faster

**Recommendation**: Start with BLIP-2 for good quality/speed balance

### For Workstation (More VRAM)

- **Figure Processing**: LLaVA-1.5-13b or unquantized versions
- Better quality but requires more memory

---

## Technical Details

### How Models Download Automatically

When you first use a model:

```python
from transformers import AutoModel, AutoProcessor

# This automatically downloads from HuggingFace
model = AutoModel.from_pretrained("microsoft/table-transformer-structure-recognition")
# Downloads to: ~/.cache/huggingface/hub/
```

**First run**: Downloads model weights (~1-5 minutes depending on size)
**Subsequent runs**: Uses cached weights (instant)

### Model Loading Strategy

**Option 1: Full Model Loading**
- Load entire model into memory
- Faster inference
- Requires more VRAM

**Option 2: Quantized Models**
- Reduced precision (8-bit, 4-bit)
- Smaller memory footprint
- Slightly lower quality
- Good for limited VRAM

**Option 3: Model Offloading**
- Load parts of model on-demand
- Slower but uses less memory
- For very large models

### Inference Speed

**Table Processing:**
- TSR: ~1-2 seconds per table
- LLM Summary: ~0.5-1 second per table
- **Total**: ~2-3 seconds per table

**Figure Processing:**
- BLIP-2: ~2-4 seconds per figure
- LLaVA: ~3-6 seconds per figure
- **Total**: ~2-6 seconds per figure

---

## Integration with Existing Pipeline

### Current Flow (Stage 1 + Stage 2)

```
Image → Stage 1 (Layout Detection) → Stage 2 (OCR) → Output
```

### With Phase 3 Added

```
Image → Stage 1 (Layout Detection) → 
    ├─→ Stage 2 (OCR) for Text/Title/List
    ├─→ Stage 3 (Table Processing) for Tables
    └─→ Stage 3 (Figure Processing) for Figures
→ Combined Output
```

### Final Output Structure

```json
{
  "image": "document.png",
  "layout_elements": [
    {
      "class": 1,  // Text
      "bbox": [x, y, h, w],
      "ocr": {"text": "...", "confidence": 0.95}
    },
    {
      "class": 4,  // Table
      "bbox": [x, y, h, w],
      "table": {
        "structured_data": {
          "headers": [...],
          "rows": [...]
        },
        "summary": "This table shows..."
      }
    },
    {
      "class": 5,  // Figure
      "bbox": [x, y, h, w],
      "figure": {
        "caption": "This chart shows...",
        "confidence": 0.92
      }
    }
  ]
}
```

---

## Dependencies

### Required Packages

```bash
# For Table Processing
pip install transformers torch torchvision
pip install pandas  # For table data handling

# For Figure Processing
pip install transformers torch torchvision
pip install accelerate  # For model optimization
pip install bitsandbytes  # For quantization (optional)

# Optional: For better performance
pip install accelerate bitsandbytes
```

### Model Downloads (Automatic)

- **First run**: Downloads models (~2-5 GB total)
- **Cached**: Subsequent runs use cached models
- **Location**: `~/.cache/huggingface/hub/`

---

## Cost Considerations

### Local Models (Recommended)
- ✅ **One-time download**: Models cached locally
- ✅ **No per-use cost**: Free after download
- ✅ **Privacy**: Data stays local
- ⚠️ **GPU required**: For reasonable speed

### Cloud APIs (Alternative)
- **GPT-4V**: ~$0.01-0.03 per image
- **Claude 3**: Similar pricing
- **Pros**: No GPU needed, best quality
- **Cons**: Ongoing cost, requires internet

---

## Summary

### What You Need to Know

1. **No datasets needed** - Models are pre-trained
2. **Automatic downloads** - Models download on first use
3. **Works locally** - No API keys needed (unless using cloud APIs)
4. **GPU recommended** - For reasonable inference speed
5. **Memory usage** - ~2-7 GB VRAM depending on model choice

### Model Choices for Your Setup

**Recommended Configuration:**
- **Table TSR**: Table-Transformer (HuggingFace)
- **Table Summary**: FLAN-T5-small
- **Figure Caption**: BLIP-2-opt-2.7b

**Total Memory**: ~6 GB VRAM ✅ Fits on RTX 4050

---

## Ready for Implementation

When you're ready, I'll implement:
1. Table processing pipeline with Table-Transformer + FLAN-T5
2. Figure processing pipeline with BLIP-2
3. Integration with Stage 1 + Stage 2
4. Complete end-to-end pipeline

All models will download automatically on first use!

