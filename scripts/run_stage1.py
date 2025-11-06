#!/usr/bin/env python3
"""
Stage 1: Document Layout Analysis (DLA) Inference Script

This script performs inference on document images:
1. De-skews the image (rotation correction)
2. Runs layout detection model
3. Formats output as JSON: {class: int, bbox: [x, y, h, w]}

Usage:
    python scripts/run_stage1.py \
        --model_dir models/inference/ppyoloe_ps05 \
        --image_path path/to/image.png \
        --output_path output.json
    
    # Or for directory of images:
    python scripts/run_stage1.py \
        --model_dir models/inference/ppyoloe_ps05 \
        --image_dir path/to/images/ \
        --output_dir outputs/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import preprocessing and utilities
from src.preprocessing.deskew import deskew_image
from src.utils.output_formatter import format_paddle_detection_output, save_predictions_json

# Add PaddleDetection to path
paddle_dir = project_root / "PaddleDetection"
if paddle_dir.exists():
    sys.path.insert(0, str(paddle_dir))

# Try to import PaddleDetection inference API
try:
    import paddle
    from paddle.inference import Config, create_predictor
    import yaml
    PADDLE_INFERENCE_AVAILABLE = True
except ImportError:
    PADDLE_INFERENCE_AVAILABLE = False

# Fallback to Trainer API if inference API not available
if not PADDLE_INFERENCE_AVAILABLE:
    try:
        from ppdet.engine import Trainer
        from ppdet.core.workspace import load_config
        from ppdet.utils.check import check_gpu
        from ppdet.utils.logger import setup_logger
        logger = setup_logger('run_stage1')
        USE_TRAINER_API = True
    except ImportError as e:
        print(f"ERROR: Could not import PaddleDetection. Make sure it's installed.")
        print(f"Error: {e}")
        sys.exit(1)
else:
    USE_TRAINER_API = False
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('run_stage1')


class LayoutDetectionModel:
    """Wrapper for PaddleDetection layout detection model."""
    
    def __init__(self, model_dir: str, use_gpu: bool = True, score_threshold: float = 0.5):
        """
        Initialize the layout detection model.
        
        Args:
            model_dir: Directory containing exported inference model
            use_gpu: Whether to use GPU
            score_threshold: Minimum confidence score for predictions
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.score_threshold = score_threshold
        
        # Check if model directory exists
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Check for required files
        model_file = self.model_dir / "model.pdmodel"
        params_file = self.model_dir / "model.pdiparams"
        config_file = self.model_dir / "infer_cfg.yml"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not params_file.exists():
            raise FileNotFoundError(f"Params file not found: {params_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Use inference API if available, otherwise fallback to Trainer
        if PADDLE_INFERENCE_AVAILABLE and not USE_TRAINER_API:
            self._init_inference_api(model_file, params_file, config_file)
        else:
            self._init_trainer_api(config_file, params_file)
        
        logger.info(f"Model loaded from: {model_dir}")
        logger.info(f"Using device: {'GPU' if use_gpu else 'CPU'}")
    
    def _init_inference_api(self, model_file, params_file, config_file):
        """Initialize using Paddle Inference API."""
        # Load config
        with open(config_file, 'r', encoding='utf-8') as f:
            self.infer_cfg = yaml.safe_load(f)
        
        # Create predictor config
        config = Config(str(model_file), str(params_file))
        
        if self.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(1)
        
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)
        
        # Create predictor
        self.predictor = create_predictor(config)
        
        # Get input/output handles
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        
        logger.info("Using Paddle Inference API")
    
    def _init_trainer_api(self, config_file, params_file):
        """Initialize using Trainer API (fallback)."""
        # Setup device
        if self.use_gpu:
            check_gpu()
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')
        
        # Load inference config
        self.cfg = load_config(str(config_file))
        
        # Create trainer
        self.trainer = Trainer(self.cfg, mode='test')
        
        # Load model weights
        self.trainer.load_weights(str(params_file))
        
        logger.info("Using Trainer API (fallback mode)")
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (BGR format, numpy array)
        
        Returns:
            List of formatted predictions: [{class: int, bbox: [x, y, h, w]}, ...]
        """
        if hasattr(self, 'predictor'):
            # Use inference API
            return self._predict_inference_api(image)
        else:
            # Use Trainer API
            return self._predict_trainer_api(image)
    
    def _predict_inference_api(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Predict using Paddle Inference API."""
        # Preprocess image (simplified - may need to match config)
        # This is a basic implementation; you may need to adjust based on your model's preprocessing
        h, w = image.shape[:2]
        
        # Prepare input
        # Note: This is simplified - actual preprocessing should match your model config
        input_tensor = self.predictor.get_input_handle(self.input_names[0])
        input_tensor.copy_from_cpu(np.array([image]).astype('float32'))
        
        # Run inference
        self.predictor.run()
        
        # Get output
        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_tensor.copy_to_cpu()
        
        # Format output (simplified - adjust based on actual output format)
        # PaddleDetection typically outputs bbox as [N, 6] where each row is [class_id, score, x1, y1, x2, y2]
        result = {
            'bbox': output_data,
            'bbox_num': [len(output_data)]  # Assuming single image
        }
        
        predictions = format_paddle_detection_output(
            result,
            score_threshold=self.score_threshold
        )
        
        return predictions
    
    def _predict_trainer_api(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Predict using Trainer API."""
        # Run inference using trainer
        results = self.trainer.predict([image])
        
        if not results or len(results) == 0:
            return []
        
        # Extract predictions from first image
        result = results[0]
        
        # Format output
        predictions = format_paddle_detection_output(
            result,
            score_threshold=self.score_threshold
        )
        
        return predictions
    
    def predict_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images (BGR format)
        
        Returns:
            List of prediction lists, one per image
        """
        all_predictions = []
        
        if hasattr(self, 'predictor'):
            # Inference API - process one by one (batch support can be added)
            for image in images:
                predictions = self._predict_inference_api(image)
                all_predictions.append(predictions)
        else:
            # Trainer API
            results = self.trainer.predict(images)
            for result in results:
                predictions = format_paddle_detection_output(
                    result,
                    score_threshold=self.score_threshold
                )
                all_predictions.append(predictions)
        
        return all_predictions


def process_image(image_path: str | Path,
                 model: LayoutDetectionModel,
                 deskew: bool = True,
                 max_skew_angle: float = 10.0) -> Dict[str, Any]:
    """
    Process a single image through the pipeline.
    
    Args:
        image_path: Path to input image
        model: Layout detection model
        deskew: Whether to apply de-skewing
        max_skew_angle: Maximum angle to correct
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]
    skew_angle = 0.0
    
    # De-skew if requested
    if deskew:
        try:
            image, skew_angle = deskew_image(image, max_angle=max_skew_angle)
            logger.info(f"De-skewed image: {skew_angle:.2f} degrees")
        except Exception as e:
            logger.warning(f"De-skewing failed: {e}. Continuing without de-skewing.")
    
    # Run inference
    predictions = model.predict(image)
    
    # Prepare output
    output = {
        "image": str(Path(image_path).name),
        "original_shape": list(original_shape),
        "processed_shape": list(image.shape[:2]),
        "skew_angle": float(skew_angle),
        "predictions": predictions,
        "num_predictions": len(predictions)
    }
    
    return output


def process_directory(image_dir: str | Path,
                     model: LayoutDetectionModel,
                     output_dir: str | Path,
                     deskew: bool = True,
                     max_skew_angle: float = 10.0,
                     extensions: List[str] = None) -> None:
    """
    Process all images in a directory.
    
    Args:
        image_dir: Directory containing images
        model: Layout detection model
        output_dir: Directory to save JSON outputs
        deskew: Whether to apply de-skewing
        max_skew_angle: Maximum angle to correct
        extensions: Image file extensions to process (default: ['.png', '.jpg', '.jpeg'])
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {image_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"Processing [{i}/{len(image_files)}]: {image_path.name}")
            
            output = process_image(
                image_path,
                model,
                deskew=deskew,
                max_skew_angle=max_skew_angle
            )
            
            # Save output
            output_file = output_dir / f"{image_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  → Saved: {output_file}")
            logger.info(f"  → Found {output['num_predictions']} layout elements")
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            continue
    
    logger.info(f"Processing complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Document Layout Analysis Inference"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing exported inference model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to single image file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save JSON output (for single image)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stage1",
        help="Directory to save JSON outputs (for directory mode)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for predictions (default: 0.5)"
    )
    parser.add_argument(
        "--no_deskew",
        action="store_true",
        help="Disable de-skewing preprocessing"
    )
    parser.add_argument(
        "--max_skew_angle",
        type=float,
        default=10.0,
        help="Maximum skew angle to correct in degrees (default: 10.0)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        parser.error("Must provide either --image_path or --image_dir")
    
    if args.image_path and args.image_dir:
        parser.error("Cannot use both --image_path and --image_dir")
    
    # Initialize model
    logger.info("=" * 60)
    logger.info("Initializing Layout Detection Model")
    logger.info("=" * 60)
    
    try:
        model = LayoutDetectionModel(
            model_dir=args.model_dir,
            use_gpu=not args.cpu,
            score_threshold=args.score_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Process image(s)
    if args.image_path:
        # Single image mode
        logger.info("=" * 60)
        logger.info("Processing Single Image")
        logger.info("=" * 60)
        
        try:
            output = process_image(
                args.image_path,
                model,
                deskew=not args.no_deskew,
                max_skew_angle=args.max_skew_angle
            )
            
            # Save or print output
            if args.output_path:
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to: {args.output_path}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
            
            logger.info(f"Found {output['num_predictions']} layout elements")
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            sys.exit(1)
    
    else:
        # Directory mode
        logger.info("=" * 60)
        logger.info("Processing Image Directory")
        logger.info("=" * 60)
        
        process_directory(
            args.image_dir,
            model,
            args.output_dir,
            deskew=not args.no_deskew,
            max_skew_angle=args.max_skew_angle
        )


if __name__ == "__main__":
    main()

