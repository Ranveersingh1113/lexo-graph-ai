"""
Google Cloud Vision API client for OCR.

This module provides OCR functionality using Google Cloud Vision API
for extracting text from document images.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from google.api_core import retry
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("Warning: google-cloud-vision not installed. Install with: pip install google-cloud-vision")


class GoogleVisionOCR:
    """Google Cloud Vision API OCR client."""
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        languages: Optional[List[str]] = None,
        auto_detect_language: bool = True,
        use_document_text_detection: bool = True,
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Initialize Google Cloud Vision OCR client.
        
        Args:
            credentials_path: Path to service account JSON credentials file
            project_id: Google Cloud project ID (optional, auto-detected)
            languages: List of language codes (e.g., ['en', 'hi', 'bn'])
            auto_detect_language: Whether to auto-detect language
            use_document_text_detection: Use document text detection (better for dense text)
            max_retries: Maximum number of retries for API calls
            timeout_seconds: Timeout for API calls
        """
        if not GOOGLE_VISION_AVAILABLE:
            raise ImportError(
                "google-cloud-vision not installed. "
                "Install with: pip install google-cloud-vision"
            )
        
        # Initialize client with credentials
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Use default credentials from environment
            self.client = vision.ImageAnnotatorClient()
        
        self.project_id = project_id
        self.languages = languages or []
        self.auto_detect_language = auto_detect_language
        self.use_document_text_detection = use_document_text_detection
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
    
    def _prepare_image(self, image: np.ndarray | bytes | str | Path) -> vision.Image:
        """
        Prepare image for API call.
        
        Args:
            image: Image as numpy array, bytes, file path, or Path object
        
        Returns:
            vision.Image object
        """
        if isinstance(image, (str, Path)):
            # Load from file
            with open(image, 'rb') as image_file:
                content = image_file.read()
        elif isinstance(image, bytes):
            # Already bytes
            content = image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to bytes
            _, encoded_image = cv2.imencode('.png', image)
            content = encoded_image.tobytes()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return vision.Image(content=content)
    
    def _extract_text_from_response(
        self,
        response: Any,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from API response.
        
        Args:
            response: Response from Google Cloud Vision API
            confidence_threshold: Minimum confidence score
        
        Returns:
            Dictionary with text, confidence, language, etc.
        """
        if not response.text_annotations:
            return {
                'text': '',
                'confidence': 0.0,
                'language': None,
                'full_text_annotation': None
            }
        
        # Get full text (first annotation contains full text)
        full_text_annotation = response.text_annotations[0]
        detected_text = full_text_annotation.description
        
        # Calculate average confidence from individual words
        confidences = []
        if hasattr(response, 'full_text_annotation') and response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            if word.property and word.property.detected_languages:
                                confidences.append(word.confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
        
        # Detect language
        detected_language = None
        if self.auto_detect_language:
            if hasattr(response, 'full_text_annotation') and response.full_text_annotation:
                # Try to get language from first detected language
                for page in response.full_text_annotation.pages:
                    if page.property and page.property.detected_languages:
                        detected_language = page.property.detected_languages[0].language_code
                        break
        
        return {
            'text': detected_text.strip(),
            'confidence': float(avg_confidence),
            'language': detected_language,
            'full_text_annotation': response.full_text_annotation if hasattr(response, 'full_text_annotation') else None
        }
    
    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def detect_text(
        self,
        image: np.ndarray | bytes | str | Path,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform OCR on a single image.
        
        Args:
            image: Input image (numpy array, bytes, or file path)
            confidence_threshold: Minimum confidence score
        
        Returns:
            Dictionary with OCR results:
            {
                'text': str,
                'confidence': float,
                'language': str,
                'full_text_annotation': Any
            }
        """
        # Prepare image
        vision_image = self._prepare_image(image)
        
        # Configure image context with language hints
        image_context = vision.ImageContext()
        if self.languages:
            image_context.language_hints = self.languages
        
        # Perform OCR
        try:
            if self.use_document_text_detection:
                response = self.client.document_text_detection(
                    image=vision_image,
                    image_context=image_context
                )
            else:
                response = self.client.text_detection(
                    image=vision_image,
                    image_context=image_context
                )
            
            # Check for errors
            if response.error.message:
                raise Exception(f"API Error: {response.error.message}")
            
            # Extract text
            result = self._extract_text_from_response(response, confidence_threshold)
            return result
            
        except Exception as e:
            # Retry logic is handled by decorator
            raise Exception(f"OCR failed: {str(e)}")
    
    def detect_text_batch(
        self,
        images: List[np.ndarray | bytes | str | Path],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform OCR on multiple images.
        
        Args:
            images: List of input images
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of OCR results (one per image)
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.detect_text(image, confidence_threshold)
                results.append(result)
            except Exception as e:
                # Add error result
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'language': None,
                    'error': str(e)
                })
            
            # Small delay to avoid rate limiting
            if i < len(images) - 1:
                time.sleep(0.1)
        
        return results
    
    def detect_text_with_bbox(
        self,
        image: np.ndarray | bytes | str | Path,
        bbox: List[float],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform OCR on a specific region of an image.
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, h, w] to crop region
            confidence_threshold: Minimum confidence score
        
        Returns:
            OCR results for the cropped region
        """
        # Crop region first
        if isinstance(image, np.ndarray):
            cropped = image[int(bbox[1]):int(bbox[1]+bbox[2]), 
                           int(bbox[0]):int(bbox[0]+bbox[3])]
        else:
            # Load and crop
            img = cv2.imread(str(image))
            cropped = img[int(bbox[1]):int(bbox[1]+bbox[2]), 
                         int(bbox[0]):int(bbox[0]+bbox[3])]
        
        # Perform OCR on cropped region
        return self.detect_text(cropped, confidence_threshold)


def create_ocr_client(config_path: Optional[str] = None) -> GoogleVisionOCR:
    """
    Create OCR client from configuration file.
    
    Args:
        config_path: Path to YAML config file (default: config/ocr_config.yaml)
    
    Returns:
        GoogleVisionOCR client instance
    """
    import yaml
    
    if config_path is None:
        config_path = "config/ocr_config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    google_config = config.get('google_cloud', {})
    ocr_config = config.get('ocr_settings', {})
    processing_config = config.get('processing', {})
    
    # Create client
    client = GoogleVisionOCR(
        credentials_path=google_config.get('credentials_path'),
        project_id=google_config.get('project_id'),
        languages=ocr_config.get('languages', []),
        auto_detect_language=ocr_config.get('auto_detect_language', True),
        use_document_text_detection=ocr_config.get('use_document_text_detection', True),
        max_retries=processing_config.get('max_retries', 3),
        timeout_seconds=processing_config.get('timeout_seconds', 30)
    )
    
    return client


if __name__ == "__main__":
    # Test OCR
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_google.py <image_path> [--config config.yaml]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    config_path = None
    
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]
    
    # Create client
    if config_path:
        ocr_client = create_ocr_client(config_path)
    else:
        ocr_client = create_ocr_client()
    
    # Perform OCR
    print(f"Performing OCR on: {image_path}")
    result = ocr_client.detect_text(image_path)
    
    print(f"\nDetected Text:")
    print(result['text'])
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"Language: {result['language'] or 'Unknown'}")


