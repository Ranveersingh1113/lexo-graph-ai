# Phase 3: Model Evaluation Framework

This document explains how to evaluate and compare different models for Phase 3 (Table and Figure processing).

## Overview

The evaluation framework allows you to:
1. Test multiple TSR models on table images
2. Test multiple figure captioning models on figure images
3. Compare performance using standard metrics
4. Generate comparative analysis reports
5. Select the best models based on evaluation results

## Evaluation Metrics

### Table Structure Recognition (TSR) Metrics

1. **Structure Accuracy**: Accuracy of detecting table structure (headers, rows)
2. **Cell Accuracy**: Accuracy of extracting text from individual cells
3. **Edit Distance**: Similarity between predicted and ground truth structure

### Figure Captioning Metrics

1. **BLEU Score**: Precision-based metric for caption quality
2. **ROUGE-L**: Longest common subsequence-based metric
3. **METEOR**: Metric considering synonyms and paraphrases
4. **CIDEr**: Consensus-based image description evaluation
5. **Semantic Similarity**: Embedding-based semantic similarity

## Setup

### 1. Install Required Packages

```bash
pip install transformers torch torchvision
pip install sentence-transformers  # For semantic similarity
pip install nltk  # For BLEU, METEOR
pip install rouge-score  # For ROUGE
pip install pandas  # For results analysis
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('wordnet')  # For METEOR
nltk.download('punkt')    # For tokenization
```

## Creating Test Dataset

### Step 1: Prepare Test Images

Organize your test images:
```
test_data/
├── tables/
│   ├── table1.png
│   ├── table2.png
│   └── ...
└── figures/
    ├── figure1.png
    ├── figure2.png
    └── ...
```

### Step 2: Create Test Dataset Template

```bash
python scripts/create_test_dataset.py \
    --images_dir test_data/tables/ \
    --output test_data.json \
    --type tables
```

### Step 3: Add Ground Truth Annotations

Edit `test_data.json` and add ground truth:

```json
{
  "tables": [
    {
      "image_path": "test_data/tables/table1.png",
      "ground_truth": {
        "headers": ["Name", "Age", "City"],
        "rows": [
          ["John", "25", "New York"],
          ["Jane", "30", "London"]
        ]
      }
    }
  ],
  "figures": [
    {
      "image_path": "test_data/figures/figure1.png",
      "ground_truth": "This bar chart shows quarterly sales data for 2024."
    }
  ]
}
```

## Running Evaluation

### Evaluate All Models

```bash
python scripts/evaluate_phase3_models.py \
    --test_data test_data.json \
    --output_dir evaluation_results/
```

### Evaluate Only TSR Models

```bash
python scripts/evaluate_phase3_models.py \
    --test_data test_data.json \
    --output_dir evaluation_results/ \
    --tsr_only
```

### Evaluate Only Caption Models

```bash
python scripts/evaluate_phase3_models.py \
    --test_data test_data.json \
    --output_dir evaluation_results/ \
    --caption_only
```

## Output Files

After evaluation, you'll get:

1. **`tsr_results.json`**: Detailed TSR model results
2. **`caption_results.json`**: Detailed caption model results
3. **`comparative_report.json`**: Machine-readable comparison
4. **`comparative_report.md`**: Human-readable markdown report

## Understanding Results

### TSR Results

```json
{
  "model": "table-transformer",
  "structure_accuracy": 0.95,
  "cell_accuracy": 0.92,
  "edit_distance": 0.88,
  "avg_inference_time": 2.3,
  "total_samples": 10,
  "successful_samples": 10
}
```

**Interpretation:**
- Higher `structure_accuracy` = Better table structure detection
- Higher `cell_accuracy` = Better text extraction from cells
- Higher `edit_distance` = More similar to ground truth
- Lower `avg_inference_time` = Faster processing

### Caption Results

```json
{
  "model": "blip2-opt-2.7b",
  "bleu": 0.45,
  "rouge_l": 0.52,
  "meteor": 0.38,
  "cider": 0.65,
  "semantic_similarity": 0.78,
  "avg_inference_time": 3.2
}
```

**Interpretation:**
- Higher scores = Better caption quality
- `semantic_similarity` = Most important for semantic correctness
- `bleu` = Precision of word matching
- `rouge_l` = Recall of important phrases
- `meteor` = Consideration of synonyms

## Model Selection Recommendations

### Based on Evaluation Results

**For TSR:**
- Choose model with highest `structure_accuracy` + `cell_accuracy`
- Consider `avg_inference_time` if speed is important

**For Figure Captioning:**
- Choose model with highest `semantic_similarity` or `overall_score`
- Consider `avg_inference_time` for production use

### Trade-offs

1. **Accuracy vs Speed**: Larger models = better accuracy but slower
2. **GPU Memory**: Larger models need more VRAM
3. **Cost**: API-based models (GPT-4V) have ongoing costs

## Custom Metrics

You can add custom metrics from your PDF by modifying:

1. `src/stage3/evaluation.py` - Add metric calculation functions
2. Update `ModelEvaluator` class to include new metrics
3. Update report generation to include new metrics

## Example Workflow

```bash
# 1. Create test dataset
python scripts/create_test_dataset.py \
    --images_dir test_images/ \
    --output test_data.json

# 2. Edit test_data.json and add ground truth

# 3. Run evaluation
python scripts/evaluate_phase3_models.py \
    --test_data test_data.json \
    --output_dir results/

# 4. Review results
cat results/comparative_report.md

# 5. Update config with best models
# Edit config/stage3_config.yaml with best models from report
```

## Troubleshooting

### Model Download Issues
- Models download automatically on first use
- Check internet connection
- Verify HuggingFace access

### GPU Memory Errors
- Use smaller models
- Enable model quantization
- Process fewer samples at once

### Evaluation Errors
- Check ground truth format matches expected structure
- Verify image paths are correct
- Ensure all required packages are installed

## Next Steps

After evaluation:
1. Select best models based on results
2. Update `config/stage3_config.yaml` with selected models
3. Use in production pipeline

