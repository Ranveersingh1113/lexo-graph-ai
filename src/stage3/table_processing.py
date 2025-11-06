"""
Table Processing Module

Supports multiple TSR models and LLM summarization models.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from pathlib import Path


class TableProcessor:
    """Base class for table processors."""
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Process table image and return structured data."""
        raise NotImplementedError


class TableTransformerProcessor(TableProcessor):
    """Table-Transformer model for table structure recognition."""
    
    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition"):
        """
        Initialize Table-Transformer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            from PIL import Image
            
            self.model_name = model_name
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model.eval()
            self.PIL = Image
            self.loaded = True
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch pillow")
        except Exception as e:
            raise Exception(f"Failed to load Table-Transformer: {e}")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Process table image using Table-Transformer."""
        # Convert to PIL Image
        pil_image = self.PIL.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Post-process (simplified - actual implementation would parse outputs)
        # This is a placeholder - actual TSR parsing is more complex
        return {
            "model": "table-transformer",
            "structured_data": self._parse_outputs(outputs, pil_image),
            "raw_output": outputs
        }
    
    def _parse_outputs(self, outputs, image):
        """Parse model outputs to structured format."""
        # Placeholder - actual implementation would:
        # 1. Extract bounding boxes for cells
        # 2. Extract text from cells (requires OCR)
        # 3. Organize into rows/columns
        # 4. Identify headers
        return {
            "headers": [],
            "rows": [],
            "format": "json"
        }


class PaddleOCRTableProcessor(TableProcessor):
    """PaddleOCR table recognition."""
    
    def __init__(self):
        """Initialize PaddleOCR table recognition."""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.loaded = True
        except ImportError:
            raise ImportError("Install PaddleOCR: pip install paddlepaddle paddleocr")
        except Exception as e:
            raise Exception(f"Failed to load PaddleOCR: {e}")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Process table using PaddleOCR."""
        # PaddleOCR table recognition
        result = self.ocr.ocr(image, cls=True)
        
        # Parse results to structured format
        structured_data = self._parse_paddleocr(result)
        
        return {
            "model": "paddleocr",
            "structured_data": structured_data,
            "raw_output": result
        }
    
    def _parse_paddleocr(self, result):
        """Parse PaddleOCR results to structured format."""
        # Placeholder - parse OCR results into table structure
        return {
            "headers": [],
            "rows": [],
            "format": "json"
        }


class TableSummarizer:
    """Summarizes structured table data using LLMs."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize LLM for table summarization.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.loaded = True
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
        except Exception as e:
            raise Exception(f"Failed to load summarization model: {e}")
    
    def summarize(self, structured_data: Dict[str, Any], prompt_template: Optional[str] = None) -> str:
        """
        Generate summary of table data.
        
        Args:
            structured_data: Structured table data
            prompt_template: Custom prompt template
        
        Returns:
            Summary text
        """
        # Convert structured data to text
        table_text = self._format_table_text(structured_data)
        
        # Create prompt
        if prompt_template is None:
            prompt = f"Summarize this table data in natural language:\n{table_text}"
        else:
            prompt = prompt_template.format(table_data=table_text)
        
        # Generate summary
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=200, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def _format_table_text(self, structured_data: Dict[str, Any]) -> str:
        """Format structured data as text for LLM."""
        headers = structured_data.get("headers", [])
        rows = structured_data.get("rows", [])
        
        text = "Headers: " + ", ".join(str(h) for h in headers) + "\n"
        text += "Rows:\n"
        for i, row in enumerate(rows[:10]):  # Limit to first 10 rows
            text += f"Row {i+1}: " + ", ".join(str(cell) for cell in row) + "\n"
        
        return text


# Model registry for easy switching
TSR_MODELS = {
    "table-transformer": TableTransformerProcessor,
    "paddleocr": PaddleOCRTableProcessor,
}

SUMMARIZATION_MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "gpt2": "gpt2",
    "bart-large": "facebook/bart-large-cnn",
}


def create_table_processor(model_name: str = "table-transformer") -> TableProcessor:
    """Factory function to create table processor."""
    if model_name in TSR_MODELS:
        return TSR_MODELS[model_name]()
    else:
        raise ValueError(f"Unknown TSR model: {model_name}. Available: {list(TSR_MODELS.keys())}")


def create_table_summarizer(model_name: str = "flan-t5-small") -> TableSummarizer:
    """Factory function to create table summarizer."""
    if model_name in SUMMARIZATION_MODELS:
        model_id = SUMMARIZATION_MODELS[model_name]
        return TableSummarizer(model_id)
    else:
        raise ValueError(f"Unknown summarization model: {model_name}. Available: {list(SUMMARIZATION_MODELS.keys())}")


