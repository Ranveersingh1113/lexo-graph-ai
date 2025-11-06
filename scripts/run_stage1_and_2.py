#!/usr/bin/env python3
"""
Combined Stage 1 + Stage 2 Inference Pipeline

This script runs both:
- Stage 1: Document Layout Detection
- Stage 2: OCR on detected text regions

Usage:
    python scripts/run_stage1_and_2.py \
        --model_dir models/inference/ppyoloe_ps05 \
        --image_path path/to/image.png \
        --output_path result.json \
        --config config/ocr_config.yaml
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

# Import Stage 1 components
# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import Stage 1 module components
import importlib.util
spec = importlib.util.spec_from_file_location("run_stage1", scripts_dir / "run_stage1.py")
run_stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_stage1)
LayoutDetectionModel = run_stage1.LayoutDetectionModel
process_stage1 = run_stage1.process_image

# Import Stage 2 components
from src.stage2.ocr_pipeline import OCRPipeline


def run_combined_pipeline(
    model_dir: str,
    image_path: str | Path,
    config_path: str | Path = "config/ocr_config.yaml",
    output_path: Optional[str | Path] = None,
    score_threshold: float = 0.5,
    deskew: bool = True,
    max_skew_angle: float = 10.0,
    use_gpu: bool = True,
    save_cropped: bool = False,
    output_dir: Optional[str | Path] = None
) -> Dict:
    """
    Run complete pipeline: Stage 1 (Layout Detection) + Stage 2 (OCR).
    
    Args:
        model_dir: Path to Stage 1 inference model directory
        image_path: Path to input image
        config_path: Path to OCR config file
        output_path: Path to save combined results JSON
        score_threshold: Minimum confidence for layout detection
        deskew: Whether to apply de-skewing
        max_skew_angle: Maximum skew angle to correct
        use_gpu: Whether to use GPU for Stage 1
        save_cropped: Whether to save cropped text regions
        output_dir: Directory for outputs (cropped images, etc.)
    
    Returns:
        Combined results dictionary
    """
    import logging
    logger = logging.getLogger('run_stage1_and_2')
    
    logger.info("=" * 60)
    logger.info("Starting Combined Stage 1 + Stage 2 Pipeline")
    logger.info("=" * 60)
    
    # Stage 1: Layout Detection
    logger.info("\n[Stage 1] Running Layout Detection...")
    logger.info(f"  Model: {model_dir}")
    logger.info(f"  Image: {image_path}")
    
    # Initialize Stage 1 model
    layout_model = LayoutDetectionModel(
        model_dir=model_dir,
        use_gpu=use_gpu,
        score_threshold=score_threshold
    )
    
    # Run Stage 1 inference
    stage1_result = process_stage1(
        image_path,
        layout_model,
        deskew=deskew,
        max_skew_angle=max_skew_angle
    )
    
    logger.info(f"  ✓ Found {stage1_result['num_predictions']} layout elements")
    
    # Stage 2: OCR
    logger.info("\n[Stage 2] Running OCR on text regions...")
    logger.info(f"  Config: {config_path}")
    
    # Initialize OCR pipeline
    ocr_pipeline = OCRPipeline(config_path=str(config_path))
    
    # Run OCR
    combined_result = ocr_pipeline.process_image(
        image_path,
        stage1_result['predictions'],
        save_cropped=save_cropped,
        output_dir=output_dir
    )
    
    logger.info(f"  ✓ OCR completed on {combined_result['num_ocr_results']} regions")
    
    # Combine results
    final_result = {
        'image': combined_result['image'],
        'stage1': {
            'original_shape': stage1_result.get('original_shape'),
            'processed_shape': stage1_result.get('processed_shape'),
            'skew_angle': stage1_result.get('skew_angle'),
            'num_predictions': stage1_result['num_predictions']
        },
        'layout_elements': combined_result['layout_elements'],
        'summary': {
            'total_elements': len(combined_result['layout_elements']),
            'elements_with_ocr': combined_result['num_ocr_results'],
            'elements_without_ocr': len(combined_result['layout_elements']) - combined_result['num_ocr_results']
        }
    }
    
    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✓ Results saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    
    return final_result


def process_directory(
    model_dir: str,
    image_dir: str | Path,
    config_path: str | Path = "config/ocr_config.yaml",
    output_dir: str | Path = "outputs/stage1_and_2",
    score_threshold: float = 0.5,
    deskew: bool = True,
    max_skew_angle: float = 10.0,
    use_gpu: bool = True,
    save_cropped: bool = False
):
    """Process directory of images."""
    import logging
    logger = logging.getLogger('run_stage1_and_2')
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {image_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize models once
    logger.info("Initializing models...")
    layout_model = LayoutDetectionModel(
        model_dir=model_dir,
        use_gpu=use_gpu,
        score_threshold=score_threshold
    )
    ocr_pipeline = OCRPipeline(config_path=str(config_path))
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            # Stage 1
            stage1_result = process_stage1(
                image_path,
                layout_model,
                deskew=deskew,
                max_skew_angle=max_skew_angle
            )
            
            # Stage 2
            combined_result = ocr_pipeline.process_image(
                image_path,
                stage1_result['predictions'],
                save_cropped=save_cropped,
                output_dir=output_dir / "cropped" if save_cropped else None
            )
            
            # Combine
            final_result = {
                'image': combined_result['image'],
                'stage1': {
                    'original_shape': stage1_result.get('original_shape'),
                    'processed_shape': stage1_result.get('processed_shape'),
                    'skew_angle': stage1_result.get('skew_angle'),
                    'num_predictions': stage1_result['num_predictions']
                },
                'layout_elements': combined_result['layout_elements'],
                'summary': {
                    'total_elements': len(combined_result['layout_elements']),
                    'elements_with_ocr': combined_result['num_ocr_results'],
                    'elements_without_ocr': len(combined_result['layout_elements']) - combined_result['num_ocr_results']
                }
            }
            
            # Save
            output_file = output_dir / f"{image_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  ✓ Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {image_path.name}: {e}")
            continue
    
    logger.info(f"\n✓ Processing complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined Stage 1 + Stage 2 Inference Pipeline"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to Stage 1 inference model directory"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to single image file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images (alternative to --image_path)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ocr_config.yaml",
        help="Path to OCR config file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save output JSON (for single image)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage1_and_2",
        help="Directory to save outputs (for directory mode)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for layout detection"
    )
    parser.add_argument(
        "--no_deskew",
        action="store_true",
        help="Disable de-skewing"
    )
    parser.add_argument(
        "--max_skew_angle",
        type=float,
        default=10.0,
        help="Maximum skew angle to correct"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "--save_cropped",
        action="store_true",
        help="Save cropped text regions for debugging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        parser.error("Must provide either --image_path or --image_dir")
    
    if args.image_path and args.image_dir:
        parser.error("Cannot use both --image_path and --image_dir")
    
    # Process
    if args.image_path:
        # Single image
        result = run_combined_pipeline(
            model_dir=args.model_dir,
            image_path=args.image_path,
            config_path=args.config,
            output_path=args.output_path,
            score_threshold=args.score_threshold,
            deskew=not args.no_deskew,
            max_skew_angle=args.max_skew_angle,
            use_gpu=not args.cpu,
            save_cropped=args.save_cropped,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total layout elements: {result['summary']['total_elements']}")
        print(f"  Elements with OCR: {result['summary']['elements_with_ocr']}")
        print(f"  Elements without OCR: {result['summary']['elements_without_ocr']}")
        print("=" * 60)
    else:
        # Directory mode
        process_directory(
            model_dir=args.model_dir,
            image_dir=args.image_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            deskew=not args.no_deskew,
            max_skew_angle=args.max_skew_angle,
            use_gpu=not args.cpu,
            save_cropped=args.save_cropped
        )


if __name__ == "__main__":
    main()

