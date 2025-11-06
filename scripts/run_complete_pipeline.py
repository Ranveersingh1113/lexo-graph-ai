#!/usr/bin/env python3
"""
Complete Pipeline: Stage 1 + Stage 2 + Stage 3

Runs the complete document understanding pipeline:
- Stage 1: Layout Detection
- Stage 2: OCR on text regions
- Stage 3: Table and Figure processing
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Stage 1+2 components
import importlib.util
scripts_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("run_stage1", scripts_dir / "run_stage1.py")
run_stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_stage1)
LayoutDetectionModel = run_stage1.LayoutDetectionModel
process_stage1 = run_stage1.process_image

from src.stage2.ocr_pipeline import OCRPipeline
from src.stage3.pipeline import Stage3Pipeline


def run_complete_pipeline(
    model_dir: str,
    image_path: str | Path,
    stage2_config: str = "config/ocr_config.yaml",
    stage3_config: str = "config/stage3_config.yaml",
    output_path: Optional[str | Path] = None,
    score_threshold: float = 0.5,
    deskew: bool = True,
    use_gpu: bool = True
) -> Dict:
    """Run complete pipeline: Stage 1 + 2 + 3."""
    import logging
    logger = logging.getLogger('complete_pipeline')
    
    logger.info("=" * 60)
    logger.info("Complete Document Understanding Pipeline")
    logger.info("=" * 60)
    
    # Stage 1: Layout Detection
    logger.info("\n[Stage 1] Layout Detection...")
    layout_model = LayoutDetectionModel(model_dir=model_dir, use_gpu=use_gpu, score_threshold=score_threshold)
    stage1_result = process_stage1(image_path, layout_model, deskew=deskew)
    logger.info(f"  ✓ Found {stage1_result['num_predictions']} layout elements")
    
    # Stage 2: OCR
    logger.info("\n[Stage 2] OCR on text regions...")
    ocr_pipeline = OCRPipeline(config_path=stage2_config)
    stage2_result = ocr_pipeline.process_image(image_path, stage1_result['predictions'])
    logger.info(f"  ✓ OCR completed on {stage2_result['num_ocr_results']} regions")
    
    # Stage 3: Tables and Figures
    logger.info("\n[Stage 3] Table and Figure processing...")
    stage3_pipeline = Stage3Pipeline(config_path=stage3_config)
    stage3_result = stage3_pipeline.process_image(image_path, stage1_result['predictions'])
    logger.info(f"  ✓ Processed {len(stage3_result['tables'])} tables and {len(stage3_result['figures'])} figures")
    
    # Combine all results
    final_result = {
        "image": str(Path(image_path).name),
        "stage1": {
            "original_shape": stage1_result.get('original_shape'),
            "processed_shape": stage1_result.get('processed_shape'),
            "skew_angle": stage1_result.get('skew_angle'),
            "num_predictions": stage1_result['num_predictions']
        },
        "layout_elements": []
    }
    
    # Combine all elements
    # Stage 2 OCR results
    ocr_by_bbox = {tuple(e['bbox']): e for e in stage2_result['layout_elements'] if e.get('ocr')}
    
    # Stage 3 table results
    tables_by_bbox = {tuple(t['bbox']): t for t in stage3_result['tables']}
    
    # Stage 3 figure results
    figures_by_bbox = {tuple(f['bbox']): f for f in stage3_result['figures']}
    
    # Combine all predictions
    for pred in stage1_result['predictions']:
        bbox = tuple(pred['bbox'])
        class_id = pred['class']
        
        element = {
            "class": class_id,
            "bbox": pred['bbox']
        }
        
        # Add OCR if available
        if bbox in ocr_by_bbox:
            element['ocr'] = ocr_by_bbox[bbox]['ocr']
        
        # Add table data if available
        if class_id == 4 and bbox in tables_by_bbox:
            element['table'] = tables_by_bbox[bbox]
        
        # Add figure data if available
        if class_id == 5 and bbox in figures_by_bbox:
            element['figure'] = figures_by_bbox[bbox]
        
        final_result['layout_elements'].append(element)
    
    # Add summary
    final_result['summary'] = {
        "total_elements": len(final_result['layout_elements']),
        "elements_with_ocr": len([e for e in final_result['layout_elements'] if e.get('ocr')]),
        "tables": len([e for e in final_result['layout_elements'] if e.get('table')]),
        "figures": len([e for e in final_result['layout_elements'] if e.get('figure')])
    }
    
    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✓ Complete results saved to: {output_path}")
    
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Complete Document Understanding Pipeline")
    parser.add_argument("--model_dir", type=str, required=True, help="Stage 1 model directory")
    parser.add_argument("--image_path", type=str, help="Input image path")
    parser.add_argument("--image_dir", type=str, help="Input image directory")
    parser.add_argument("--output_path", type=str, help="Output JSON path (single image)")
    parser.add_argument("--output_dir", type=str, default="outputs/complete", help="Output directory")
    parser.add_argument("--stage2_config", type=str, default="config/ocr_config.yaml")
    parser.add_argument("--stage3_config", type=str, default="config/stage3_config.yaml")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--no_deskew", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.image_path:
        result = run_complete_pipeline(
            model_dir=args.model_dir,
            image_path=args.image_path,
            stage2_config=args.stage2_config,
            stage3_config=args.stage3_config,
            output_path=args.output_path,
            score_threshold=args.score_threshold,
            deskew=not args.no_deskew,
            use_gpu=not args.cpu
        )
        print("\nSummary:")
        print(f"  Total elements: {result['summary']['total_elements']}")
        print(f"  With OCR: {result['summary']['elements_with_ocr']}")
        print(f"  Tables: {result['summary']['tables']}")
        print(f"  Figures: {result['summary']['figures']}")
    else:
        print("Directory mode not yet implemented. Use --image_path for single image.")


if __name__ == "__main__":
    main()


