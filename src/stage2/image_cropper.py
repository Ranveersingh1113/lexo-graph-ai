"""
Image cropping utilities for extracting text regions from document images.

This module handles cropping of image regions based on bounding box coordinates
from Stage 1 layout detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


def validate_bbox(bbox: List[float], image_shape: Tuple[int, int]) -> Tuple[bool, List[float]]:
    """
    Validate and clip bounding box to image boundaries.
    
    Args:
        bbox: Bounding box in format [x, y, h, w]
        image_shape: Image shape as (height, width)
    
    Returns:
        Tuple of (is_valid, clipped_bbox)
    """
    x, y, h, w = bbox
    img_h, img_w = image_shape
    
    # Check if bbox is valid
    if w <= 0 or h <= 0:
        return False, bbox
    
    if x < 0 or y < 0 or x >= img_w or y >= img_h:
        return False, bbox
    
    # Clip to image boundaries
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    
    # Ensure minimum size
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False, bbox
    
    return True, [x1, y1, y2 - y1, x2 - x1]  # Return as [x, y, h, w]


def crop_region(image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """
    Crop a region from an image based on bounding box.
    
    Args:
        image: Input image (BGR format, numpy array)
        bbox: Bounding box in format [x, y, h, w] (top-left corner, height, width)
    
    Returns:
        Cropped image region, or None if invalid
    """
    if image is None or len(image.shape) < 2:
        return None
    
    # Validate bbox
    is_valid, clipped_bbox = validate_bbox(bbox, image.shape[:2])
    if not is_valid:
        return None
    
    x, y, h, w = clipped_bbox
    
    # Crop image
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def crop_regions_from_predictions(
    image: np.ndarray,
    predictions: List[Dict[str, Any]],
    category_filter: Optional[List[int]] = None,
    min_size: Tuple[int, int] = (10, 10)
) -> List[Dict[str, Any]]:
    """
    Crop multiple regions from image based on predictions.
    
    Args:
        image: Input image (BGR format, numpy array)
        predictions: List of predictions from Stage 1, each with 'class' and 'bbox'
        category_filter: List of category IDs to include (None = all categories)
        min_size: Minimum size (height, width) for valid crops
    
    Returns:
        List of dictionaries with cropped images and metadata:
        {
            'cropped_image': np.ndarray,
            'class': int,
            'bbox': [x, y, h, w],
            'original_bbox': [x, y, h, w],
            'index': int
        }
    """
    cropped_regions = []
    
    for idx, pred in enumerate(predictions):
        class_id = pred.get('class', pred.get('class_id', None))
        bbox = pred.get('bbox', [])
        
        if class_id is None or len(bbox) != 4:
            continue
        
        # Filter by category if specified
        if category_filter is not None and class_id not in category_filter:
            continue
        
        # Crop region
        cropped = crop_region(image, bbox)
        
        if cropped is None:
            continue
        
        # Check minimum size
        if cropped.shape[0] < min_size[0] or cropped.shape[1] < min_size[1]:
            continue
        
        cropped_regions.append({
            'cropped_image': cropped,
            'class': class_id,
            'bbox': bbox,
            'original_bbox': bbox.copy(),
            'index': idx
        })
    
    return cropped_regions


def save_cropped_images(
    cropped_regions: List[Dict[str, Any]],
    output_dir: str | Path,
    base_name: str = "image",
    image_format: str = "png"
) -> List[str]:
    """
    Save cropped images to disk.
    
    Args:
        cropped_regions: List of cropped regions from crop_regions_from_predictions
        output_dir: Directory to save cropped images
        base_name: Base name for saved files
        image_format: Image format ('png', 'jpg', etc.)
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for region in cropped_regions:
        idx = region['index']
        class_id = region['class']
        bbox = region['bbox']
        
        # Create filename
        filename = f"{base_name}_class{class_id}_idx{idx}.{image_format}"
        filepath = output_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), region['cropped_image'])
        saved_paths.append(str(filepath))
    
    return saved_paths


def crop_from_image_file(
    image_path: str | Path,
    predictions: List[Dict[str, Any]],
    category_filter: Optional[List[int]] = None,
    save_crops: bool = False,
    output_dir: Optional[str | Path] = None
) -> List[Dict[str, Any]]:
    """
    Crop regions from an image file based on predictions.
    
    Args:
        image_path: Path to input image
        predictions: List of predictions from Stage 1
        category_filter: List of category IDs to include
        save_crops: Whether to save cropped images to disk
        output_dir: Directory to save crops (if save_crops=True)
    
    Returns:
        List of cropped regions with metadata
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Crop regions
    cropped_regions = crop_regions_from_predictions(
        image,
        predictions,
        category_filter=category_filter
    )
    
    # Save if requested
    if save_crops and output_dir:
        base_name = Path(image_path).stem
        save_cropped_images(cropped_regions, output_dir, base_name)
    
    return cropped_regions


if __name__ == "__main__":
    # Test with a sample
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python image_cropper.py <image_path> <predictions_json>")
        sys.exit(1)
    
    import json
    
    image_path = sys.argv[1]
    predictions_path = sys.argv[2]
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        data = json.load(f)
        predictions = data.get('predictions', [])
    
    # Crop regions
    cropped_regions = crop_from_image_file(
        image_path,
        predictions,
        category_filter=[1, 2, 3],  # Text, Title, List
        save_crops=True,
        output_dir="outputs/test_crops"
    )
    
    print(f"Cropped {len(cropped_regions)} regions from image")

