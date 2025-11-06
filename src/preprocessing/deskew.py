"""
Image de-skewing (rotation correction) for document images.

This module implements skew detection and correction to straighten
rotated document images before layout analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def detect_skew_angle(image: np.ndarray, 
                      method: str = 'hough') -> float:
    """
    Detect the skew angle of a document image.
    
    Args:
        image: Input image (BGR or grayscale)
        method: Detection method ('hough' or 'projection')
    
    Returns:
        Skew angle in degrees (positive = counter-clockwise)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if method == 'hough':
        return _detect_skew_hough(binary)
    elif method == 'projection':
        return _detect_skew_projection(binary)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hough' or 'projection'.")


def _detect_skew_hough(binary: np.ndarray) -> float:
    """Detect skew using Hough line transform."""
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min(binary.shape) // 4,
        maxLineGap=20
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Filter out near-vertical lines (likely not text lines)
        if abs(angle) < 85:  # Not vertical
            angles.append(angle)
    
    if len(angles) == 0:
        return 0.0
    
    # Return median angle (more robust than mean)
    return float(np.median(angles))


def _detect_skew_projection(binary: np.ndarray) -> float:
    """Detect skew using projection profile analysis."""
    h, w = binary.shape
    
    # Test angles from -45 to 45 degrees
    angles = np.arange(-45, 46, 0.5)
    best_angle = 0.0
    max_variance = 0
    
    for angle in angles:
        # Rotate image
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        # Horizontal projection
        projection = np.sum(rotated, axis=1)
        
        # Calculate variance (higher variance = better alignment)
        variance = np.var(projection)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    return best_angle


def deskew_image(image: np.ndarray, 
                 angle: Optional[float] = None,
                 method: str = 'hough',
                 max_angle: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Correct skew in a document image.
    
    Args:
        image: Input image (BGR)
        angle: Skew angle in degrees (if None, will be detected)
        method: Detection method if angle is None
        max_angle: Maximum angle to correct (ignore if larger)
    
    Returns:
        Tuple of (deskewed_image, corrected_angle)
    """
    # Detect angle if not provided
    if angle is None:
        angle = detect_skew_angle(image, method=method)
    
    # Don't correct if angle is too large (likely not skew)
    if abs(angle) > max_angle:
        return image.copy(), 0.0
    
    # Don't correct if angle is very small
    if abs(angle) < 0.1:
        return image.copy(), 0.0
    
    # Correct rotation
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Calculate new image dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    deskewed = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return deskewed, -angle


def deskew_image_file(image_path: str | Path,
                      output_path: Optional[str | Path] = None,
                      method: str = 'hough',
                      max_angle: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Load image, deskew it, and optionally save.
    
    Args:
        image_path: Path to input image
        output_path: Path to save deskewed image (optional)
        method: Detection method
        max_angle: Maximum angle to correct
    
    Returns:
        Tuple of (deskewed_image, corrected_angle)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Deskew
    deskewed, angle = deskew_image(image, method=method, max_angle=max_angle)
    
    # Save if output path provided
    if output_path is not None:
        cv2.imwrite(str(output_path), deskewed)
    
    return deskewed, angle


if __name__ == "__main__":
    # Test with a sample image
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deskew.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        deskewed, angle = deskew_image_file(image_path, output_path)
        print(f"Skew angle detected: {angle:.2f} degrees")
        if output_path:
            print(f"Deskewed image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


