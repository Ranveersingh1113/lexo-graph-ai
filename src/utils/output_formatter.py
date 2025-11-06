"""
Output formatting utilities for PS-05 inference results.

Converts model predictions to required JSON format:
{class: int, bbox: [x, y, h, w]}
"""

import json
from typing import List, Dict, Any, Tuple
import numpy as np


# Category mapping (PS-05 categories)
CATEGORY_NAMES = {
    0: "Background",
    1: "Text",
    2: "Title",
    3: "List",
    4: "Table",
    5: "Figure"
}


def convert_bbox_format(bbox: List[float], 
                       from_format: str = 'x1y1x2y2',
                       to_format: str = 'xyhw') -> List[float]:
    """
    Convert bounding box between formats.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2] or [x, y, w, h]
        from_format: Source format ('x1y1x2y2' or 'xyhw')
        to_format: Target format ('x1y1x2y2' or 'xyhw')
    
    Returns:
        Converted bounding box
    """
    if from_format == to_format:
        return bbox
    
    if from_format == 'x1y1x2y2' and to_format == 'xyhw':
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return [x, y, h, w]
    
    elif from_format == 'xyhw' and to_format == 'x1y1x2y2':
        x, y, h, w = bbox
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return [x1, y1, x2, y2]
    
    else:
        raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")


def format_prediction(class_id: int,
                     bbox: List[float],
                     score: float = 1.0,
                     bbox_format: str = 'x1y1x2y2',
                     output_format: str = 'xyhw') -> Dict[str, Any]:
    """
    Format a single prediction to required JSON format.
    
    Args:
        class_id: Category ID (0-5)
        bbox: Bounding box [x1, y1, x2, y2] or [x, y, h, w]
        score: Confidence score (optional, not in output)
        bbox_format: Input bbox format
        output_format: Output bbox format ('xyhw' required for PS-05)
    
    Returns:
        Formatted prediction: {class: int, bbox: [x, y, h, w]}
    """
    # Convert bbox to required format
    if bbox_format != output_format:
        bbox = convert_bbox_format(bbox, from_format=bbox_format, to_format=output_format)
    
    # Ensure bbox values are integers (pixel coordinates)
    bbox = [int(round(coord)) for coord in bbox]
    
    # Ensure class_id is valid
    if class_id not in CATEGORY_NAMES:
        raise ValueError(f"Invalid class_id: {class_id}. Must be 0-5.")
    
    return {
        "class": int(class_id),
        "bbox": bbox
    }


def format_predictions(predictions: List[Dict[str, Any]],
                       score_threshold: float = 0.5,
                       bbox_format: str = 'x1y1x2y2',
                       output_format: str = 'xyhw') -> List[Dict[str, Any]]:
    """
    Format multiple predictions to required JSON format.
    
    Args:
        predictions: List of predictions, each with 'class_id', 'bbox', 'score'
        score_threshold: Minimum confidence score to include
        bbox_format: Input bbox format
        output_format: Output bbox format
    
    Returns:
        List of formatted predictions
    """
    formatted = []
    
    for pred in predictions:
        # Extract components
        class_id = pred.get('class_id', pred.get('category_id', pred.get('class', None)))
        bbox = pred.get('bbox', pred.get('box', None))
        score = pred.get('score', pred.get('confidence', 1.0))
        
        if class_id is None or bbox is None:
            continue
        
        # Filter by score
        if score < score_threshold:
            continue
        
        # Format prediction
        try:
            formatted_pred = format_prediction(
                class_id=class_id,
                bbox=bbox,
                score=score,
                bbox_format=bbox_format,
                output_format=output_format
            )
            formatted.append(formatted_pred)
        except Exception as e:
            print(f"Warning: Skipping invalid prediction: {e}")
            continue
    
    return formatted


def format_paddle_detection_output(paddle_output: Dict[str, Any],
                                   score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Format PaddleDetection inference output to required format.
    
    PaddleDetection output format:
    - bbox: [N, 6] where each row is [class_id, score, x1, y1, x2, y2]
    - bbox_num: [batch_size] number of boxes per image
    
    Args:
        paddle_output: Output from PaddleDetection inference
        score_threshold: Minimum confidence score
    
    Returns:
        List of formatted predictions
    """
    bbox = paddle_output.get('bbox', None)
    bbox_num = paddle_output.get('bbox_num', None)
    
    if bbox is None:
        return []
    
    # Convert to list of predictions
    predictions = []
    
    # Handle batch dimension
    if bbox_num is not None:
        # Multiple images in batch
        start_idx = 0
        for num_boxes in bbox_num:
            for i in range(start_idx, start_idx + num_boxes):
                if i >= len(bbox):
                    break
                row = bbox[i]
                if len(row) >= 6:
                    class_id, score, x1, y1, x2, y2 = row[:6]
                    predictions.append({
                        'class_id': int(class_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'score': float(score)
                    })
            start_idx += num_boxes
    else:
        # Single image or all boxes for one image
        for row in bbox:
            if len(row) >= 6:
                class_id, score, x1, y1, x2, y2 = row[:6]
                predictions.append({
                    'class_id': int(class_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score)
                })
    
    # Format predictions
    return format_predictions(
        predictions,
        score_threshold=score_threshold,
        bbox_format='x1y1x2y2',
        output_format='xyhw'
    )


def save_predictions_json(predictions: List[Dict[str, Any]],
                         output_path: str,
                         image_name: Optional[str] = None) -> None:
    """
    Save predictions to JSON file.
    
    Args:
        predictions: List of formatted predictions
        output_path: Path to save JSON file
        image_name: Optional image name to include in output
    """
    output = {
        "predictions": predictions
    }
    
    if image_name:
        output["image"] = image_name
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test formatting
    test_predictions = [
        {
            'class_id': 1,
            'bbox': [100, 200, 300, 400],  # x1, y1, x2, y2
            'score': 0.95
        },
        {
            'class_id': 2,
            'bbox': [150, 250, 350, 450],
            'score': 0.87
        }
    ]
    
    formatted = format_predictions(test_predictions, score_threshold=0.5)
    print("Formatted predictions:")
    print(json.dumps(formatted, indent=2))
    
    # Test PaddleDetection output format
    paddle_output = {
        'bbox': np.array([
            [1, 0.95, 100, 200, 300, 400],
            [2, 0.87, 150, 250, 350, 450],
            [4, 0.45, 200, 300, 400, 500]  # Below threshold
        ]),
        'bbox_num': [3]
    }
    
    formatted_paddle = format_paddle_detection_output(paddle_output, score_threshold=0.5)
    print("\nFormatted PaddleDetection output:")
    print(json.dumps(formatted_paddle, indent=2))


