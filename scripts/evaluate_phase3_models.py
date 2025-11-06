#!/usr/bin/env python3
"""
Comparative Evaluation Script for Phase 3 Models

Tests multiple TSR and Figure Captioning models and generates a comparative analysis report.

Usage:
    python scripts/evaluate_phase3_models.py \
        --test_data test_data.json \
        --output_dir evaluation_results/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stage3.table_processing import create_table_processor, TSR_MODELS, create_table_summarizer
from src.stage3.figure_processing import create_figure_processor, FIGURE_MODELS
from src.stage3.evaluation import ModelEvaluator, TSRMetrics, CaptionMetrics


def evaluate_all_tsr_models(evaluator: ModelEvaluator, output_dir: Path) -> pd.DataFrame:
    """Evaluate all TSR models."""
    print("\n" + "=" * 60)
    print("Evaluating TSR Models")
    print("=" * 60)
    
    results = []
    
    for model_name in TSR_MODELS.keys():
        print(f"\n[TSR] Testing: {model_name}")
        try:
            processor = create_table_processor(model_name)
            result = evaluator.evaluate_tsr_model(processor, model_name)
            results.append(result)
            print(f"  ✓ Structure Accuracy: {result.get('structure_accuracy', 0):.3f}")
            print(f"  ✓ Cell Accuracy: {result.get('cell_accuracy', 0):.3f}")
            print(f"  ✓ Avg Inference Time: {result.get('avg_inference_time', 0):.3f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"model": model_name, "error": str(e)})
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / "tsr_results.json"
    df.to_json(output_file, indent=2, orient='records')
    print(f"\n✓ TSR results saved to: {output_file}")
    
    return df


def evaluate_all_caption_models(evaluator: ModelEvaluator, output_dir: Path) -> pd.DataFrame:
    """Evaluate all figure captioning models."""
    print("\n" + "=" * 60)
    print("Evaluating Figure Captioning Models")
    print("=" * 60)
    
    results = []
    
    for model_name in FIGURE_MODELS.keys():
        print(f"\n[Caption] Testing: {model_name}")
        try:
            processor = create_figure_processor(model_name)
            result = evaluator.evaluate_caption_model(processor, model_name)
            results.append(result)
            print(f"  ✓ BLEU: {result.get('bleu', 0):.3f}")
            print(f"  ✓ ROUGE-L: {result.get('rouge_l', 0):.3f}")
            print(f"  ✓ METEOR: {result.get('meteor', 0):.3f}")
            print(f"  ✓ Avg Inference Time: {result.get('avg_inference_time', 0):.3f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"model": model_name, "error": str(e)})
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / "caption_results.json"
    df.to_json(output_file, indent=2, orient='records')
    print(f"\n✓ Caption results saved to: {output_file}")
    
    return df


def generate_comparative_report(
    tsr_results: pd.DataFrame,
    caption_results: pd.DataFrame,
    output_dir: Path
):
    """Generate comparative analysis report."""
    print("\n" + "=" * 60)
    print("Generating Comparative Analysis Report")
    print("=" * 60)
    
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "summary": {},
        "tsr_comparison": {},
        "caption_comparison": {},
        "recommendations": {}
    }
    
    # TSR Comparison
    if not tsr_results.empty and "error" not in tsr_results.columns:
        valid_tsr = tsr_results[~tsr_results.get("structure_accuracy", pd.Series()).isna()]
        if not valid_tsr.empty:
            best_tsr = valid_tsr.loc[valid_tsr["structure_accuracy"].idxmax()]
            report["tsr_comparison"] = {
                "best_model": best_tsr["model"],
                "best_structure_accuracy": float(best_tsr["structure_accuracy"]),
                "best_cell_accuracy": float(best_tsr["cell_accuracy"]),
                "fastest_model": valid_tsr.loc[valid_tsr["avg_inference_time"].idxmin()]["model"],
                "all_results": tsr_results.to_dict('records')
            }
    
    # Caption Comparison
    if not caption_results.empty and "error" not in caption_results.columns:
        valid_caption = caption_results[~caption_results.get("bleu", pd.Series()).isna()]
        if not valid_caption.empty:
            # Find best by different metrics
            best_bleu = valid_caption.loc[valid_caption["bleu"].idxmax()]
            best_rouge = valid_caption.loc[valid_caption["rouge_l"].idxmax()]
            best_meteor = valid_caption.loc[valid_caption["meteor"].idxmax()]
            
            # Overall score (weighted average)
            valid_caption["overall_score"] = (
                valid_caption["bleu"] * 0.25 +
                valid_caption["rouge_l"] * 0.25 +
                valid_caption["meteor"] * 0.25 +
                valid_caption.get("semantic_similarity", 0) * 0.25
            )
            best_overall = valid_caption.loc[valid_caption["overall_score"].idxmax()]
            
            report["caption_comparison"] = {
                "best_by_bleu": best_bleu["model"],
                "best_by_rouge": best_rouge["model"],
                "best_by_meteor": best_meteor["model"],
                "best_overall": best_overall["model"],
                "best_overall_score": float(best_overall["overall_score"]),
                "fastest_model": valid_caption.loc[valid_caption["avg_inference_time"].idxmin()]["model"],
                "all_results": caption_results.to_dict('records')
            }
    
    # Recommendations
    recommendations = []
    
    if report.get("tsr_comparison"):
        best_tsr = report["tsr_comparison"]["best_model"]
        recommendations.append(f"TSR Model: Use {best_tsr} for best accuracy")
    
    if report.get("caption_comparison"):
        best_caption = report["caption_comparison"]["best_overall"]
        recommendations.append(f"Caption Model: Use {best_caption} for best overall performance")
    
    report["recommendations"] = recommendations
    
    # Save report
    report_file = output_dir / "comparative_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report, tsr_results, caption_results)
    markdown_file = output_dir / "comparative_report.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    print(f"\n✓ Comparative report saved to: {report_file}")
    print(f"✓ Markdown report saved to: {markdown_file}")
    
    return report


def generate_markdown_report(report: dict, tsr_df: pd.DataFrame, caption_df: pd.DataFrame) -> str:
    """Generate markdown formatted report."""
    md = f"""# Phase 3 Models - Comparative Analysis Report

**Evaluation Date**: {report['evaluation_date']}

## Executive Summary

This report compares multiple models for Table Structure Recognition (TSR) and Figure Captioning tasks.

## Table Structure Recognition (TSR) Models

"""
    
    if not tsr_df.empty:
        md += "### Results Summary\n\n"
        md += "| Model | Structure Accuracy | Cell Accuracy | Edit Distance | Avg Inference Time (s) |\n"
        md += "|-------|-------------------|---------------|---------------|------------------------|\n"
        
        for _, row in tsr_df.iterrows():
            if "error" not in row:
                md += f"| {row['model']} | {row.get('structure_accuracy', 0):.3f} | "
                md += f"{row.get('cell_accuracy', 0):.3f} | {row.get('edit_distance', 0):.3f} | "
                md += f"{row.get('avg_inference_time', 0):.3f} |\n"
        
        if report.get("tsr_comparison"):
            comp = report["tsr_comparison"]
            md += f"\n### Best Model\n\n"
            md += f"- **Best Accuracy**: {comp['best_model']} (Structure Accuracy: {comp['best_structure_accuracy']:.3f})\n"
            md += f"- **Fastest**: {comp['fastest_model']}\n"
    
    md += "\n## Figure Captioning Models\n\n"
    
    if not caption_df.empty:
        md += "### Results Summary\n\n"
        md += "| Model | BLEU | ROUGE-L | METEOR | CIDEr | Semantic Similarity | Avg Inference Time (s) |\n"
        md += "|-------|------|---------|--------|-------|---------------------|------------------------|\n"
        
        for _, row in caption_df.iterrows():
            if "error" not in row:
                md += f"| {row['model']} | {row.get('bleu', 0):.3f} | "
                md += f"{row.get('rouge_l', 0):.3f} | {row.get('meteor', 0):.3f} | "
                md += f"{row.get('cider', 0):.3f} | {row.get('semantic_similarity', 0):.3f} | "
                md += f"{row.get('avg_inference_time', 0):.3f} |\n"
        
        if report.get("caption_comparison"):
            comp = report["caption_comparison"]
            md += f"\n### Best Models\n\n"
            md += f"- **Best Overall**: {comp['best_overall']} (Score: {comp['best_overall_score']:.3f})\n"
            md += f"- **Best BLEU**: {comp['best_by_bleu']}\n"
            md += f"- **Best ROUGE-L**: {comp['best_by_rouge']}\n"
            md += f"- **Best METEOR**: {comp['best_by_meteor']}\n"
            md += f"- **Fastest**: {comp['fastest_model']}\n"
    
    md += "\n## Recommendations\n\n"
    for rec in report.get("recommendations", []):
        md += f"- {rec}\n"
    
    md += "\n## Detailed Results\n\n"
    md += "See `tsr_results.json` and `caption_results.json` for detailed per-sample results.\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of Phase 3 models"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--tsr_only",
        action="store_true",
        help="Only evaluate TSR models"
    )
    parser.add_argument(
        "--caption_only",
        action="store_true",
        help="Only evaluate caption models"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.test_data)
    
    # Evaluate models
    tsr_results = pd.DataFrame()
    caption_results = pd.DataFrame()
    
    if not args.caption_only:
        tsr_results = evaluate_all_tsr_models(evaluator, output_dir)
    
    if not args.tsr_only:
        caption_results = evaluate_all_caption_models(evaluator, output_dir)
    
    # Generate report
    report = generate_comparative_report(tsr_results, caption_results, output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nRecommendations:")
    for rec in report.get("recommendations", []):
        print(f"  • {rec}")


if __name__ == "__main__":
    main()


