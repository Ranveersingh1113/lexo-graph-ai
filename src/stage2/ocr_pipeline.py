"""
OCR Pipeline for Stage 2.

This module orchestrates the complete OCR pipeline:
1. Takes Stage 1 layout detection results
2. Crops text regions from images
3. Performs OCR on cropped regions
4. Combines and formats results
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np

from src.stage2.image_cropper import crop_regions_from_predictions, crop_from_image_file
from src.stage2.ocr_google import GoogleVisionOCR, create_ocr_client


class OCRPipeline:
    """Complete OCR pipeline integrating cropping and OCR."""
    
    def __init__(
        self,
        ocr_client: Optional[GoogleVisionOCR] = None,
        config_path: Optional[str] = None,
        ocr_categories: Optional[List[int]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize OCR pipeline.
        
        Args:
            ocr_client: GoogleVisionOCR client instance (if None, creates from config)
            config_path: Path to config file (used if ocr_client is None)
            ocr_categories: List of category IDs to perform OCR on (default: [1, 2, 3])
            confidence_threshold: Minimum confidence score for OCR results
        """
        # Initialize OCR client
        if ocr_client is None:
            if config_path:
                self.ocr_client = create_ocr_client(config_path)
            else:
                self.ocr_client = create_ocr_client()
        else:
            self.ocr_client = ocr_client
        
        # Categories to perform OCR on (Text, Title, List)
        self.ocr_categories = ocr_categories or [1, 2, 3]
        self.confidence_threshold = confidence_threshold
    
    def process_image(
        self,
        image_path: str | Path,
        stage1_predictions: List[Dict[str, Any]],
        save_cropped: bool = False,
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Process a single image: crop regions and perform OCR.
        
        Args:
            image_path: Path to input image
            stage1_predictions: List of predictions from Stage 1
            save_cropped: Whether to save cropped images
            output_dir: Directory to save cropped images (if save_cropped=True)
        
        Returns:
            Dictionary with OCR results:
            {
                'image': str,
                'layout_elements': [
                    {
                        'class': int,
                        'bbox': [x, y, h, w],
                        'ocr': {
                            'text': str,
                            'confidence': float,
                            'language': str
                        }
                    },
                    ...
                ]
            }
        """
        image_path = Path(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Crop regions for OCR
        cropped_regions = crop_regions_from_predictions(
            image,
            stage1_predictions,
            category_filter=self.ocr_categories
        )
        
        # Save cropped images if requested
        if save_cropped and output_dir:
            from src.stage2.image_cropper import save_cropped_images
            save_cropped_images(
                cropped_regions,
                output_dir,
                base_name=image_path.stem
            )
        
        # Perform OCR on each cropped region
        ocr_results = []
        for region in cropped_regions:
            try:
                # Perform OCR
                ocr_result = self.ocr_client.detect_text(
                    region['cropped_image'],
                    confidence_threshold=self.confidence_threshold
                )
                
                # Combine with layout detection info
                ocr_results.append({
                    'class': region['class'],
                    'bbox': region['bbox'],
                    'ocr': {
                        'text': ocr_result['text'],
                        'confidence': ocr_result['confidence'],
                        'language': ocr_result['language']
                    }
                })
            except Exception as e:
                # Add error entry
                ocr_results.append({
                    'class': region['class'],
                    'bbox': region['bbox'],
                    'ocr': {
                        'text': '',
                        'confidence': 0.0,
                        'language': None,
                        'error': str(e)
                    }
                })
        
        # Combine all layout elements (with and without OCR)
        all_elements = []
        
        # Add elements with OCR
        ocr_by_bbox = {tuple(r['bbox']): r for r in ocr_results}
        
        for pred in stage1_predictions:
            bbox = tuple(pred['bbox'])
            if bbox in ocr_by_bbox:
                # Has OCR result
                all_elements.append(ocr_by_bbox[bbox])
            else:
                # No OCR (category not in ocr_categories or failed)
                all_elements.append({
                    'class': pred['class'],
                    'bbox': pred['bbox'],
                    'ocr': None
                })
        
        return {
            'image': image_path.name,
            'layout_elements': all_elements,
            'num_ocr_results': len(ocr_results)
        }
    
    def process_stage1_output(
        self,
        stage1_json_path: str | Path,
        image_dir: Optional[str | Path] = None,
        save_cropped: bool = False,
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Process Stage 1 output JSON file and perform OCR.
        
        Args:
            stage1_json_path: Path to Stage 1 output JSON
            image_dir: Directory containing images (if paths in JSON are relative)
            save_cropped: Whether to save cropped images
            output_dir: Directory for outputs
        
        Returns:
            Combined results with OCR
        """
        stage1_json_path = Path(stage1_json_path)
        
        # Load Stage 1 results
        with open(stage1_json_path, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        
        # Get image path
        image_name = stage1_data.get('image', '')
        if image_dir:
            image_path = Path(image_dir) / image_name
        else:
            image_path = Path(image_name)
        
        # Get predictions
        predictions = stage1_data.get('predictions', [])
        
        # Process image
        result = self.process_image(
            image_path,
            predictions,
            save_cropped=save_cropped,
            output_dir=output_dir
        )
        
        # Add metadata from Stage 1
        result['stage1_metadata'] = {
            'original_shape': stage1_data.get('original_shape'),
            'processed_shape': stage1_data.get('processed_shape'),
            'skew_angle': stage1_data.get('skew_angle'),
            'num_predictions': stage1_data.get('num_predictions')
        }
        
        return result
    
    def process_batch(
        self,
        stage1_results: List[Dict[str, Any]],
        image_dir: Optional[str | Path] = None,
        save_cropped: bool = False,
        output_dir: Optional[str | Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple Stage 1 results.
        
        Args:
            stage1_results: List of Stage 1 result dictionaries
            image_dir: Directory containing images
            save_cropped: Whether to save cropped images
            output_dir: Directory for outputs
        
        Returns:
            List of combined results with OCR
        """
        all_results = []
        
        for stage1_result in stage1_results:
            try:
                # Get image path
                image_name = stage1_result.get('image', '')
                if image_dir:
                    image_path = Path(image_dir) / image_name
                else:
                    image_path = Path(image_name)
                
                # Get predictions
                predictions = stage1_result.get('predictions', [])
                
                # Process
                result = self.process_image(
                    image_path,
                    predictions,
                    save_cropped=save_cropped,
                    output_dir=output_dir
                )
                
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {stage1_result.get('image', 'unknown')}: {e}")
                continue
        
        return all_results


if __name__ == "__main__":
    # Test pipeline
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ocr_pipeline.py <stage1_json> <image_path> [--config config.yaml]")
        sys.exit(1)
    
    stage1_json = sys.argv[1]
    image_path = sys.argv[2]
    config_path = None
    
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    
    # Create pipeline
    pipeline = OCRPipeline(config_path=config_path)
    
    # Load Stage 1 results
    with open(stage1_json, 'r') as f:
        stage1_data = json.load(f)
    
    predictions = stage1_data.get('predictions', [])
    
    # Process
    print(f"Processing image: {image_path}")
    result = pipeline.process_image(image_path, predictions)
    
    # Print results
    print(f"\nOCR Results:")
    print(f"Found {result['num_ocr_results']} text regions with OCR")
    print("\n" + "=" * 60)
    for element in result['layout_elements']:
        if element.get('ocr'):
            ocr = element['ocr']
            print(f"\nClass {element['class']}:")
            print(f"  Text: {ocr['text'][:100]}...")
            print(f"  Confidence: {ocr['confidence']:.2f}")
            print(f"  Language: {ocr.get('language', 'Unknown')}")


