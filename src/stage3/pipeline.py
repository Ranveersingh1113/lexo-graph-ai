"""
Phase 3 Pipeline: Table and Figure Processing

Integrates table and figure processing with Stage 1+2 outputs.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.stage2.image_cropper import crop_regions_from_predictions
from src.stage3.table_processing import create_table_processor, create_table_summarizer
from src.stage3.figure_processing import create_figure_processor


class Stage3Pipeline:
    """Complete Stage 3 pipeline for table and figure processing."""
    
    def __init__(
        self,
        tsr_model: str = "table-transformer",
        summarization_model: str = "flan-t5-small",
        figure_model: str = "blip2-opt-2.7b",
        config_path: Optional[str] = None
    ):
        """
        Initialize Stage 3 pipeline.
        
        Args:
            tsr_model: TSR model name
            summarization_model: Summarization model name
            figure_model: Figure captioning model name
            config_path: Path to config file (optional)
        """
        # Load config if provided
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            stage3_config = config.get('stage3', {})
            tsr_model = stage3_config.get('tsr_model', tsr_model)
            summarization_model = stage3_config.get('summarization_model', summarization_model)
            figure_model = stage3_config.get('figure_model', figure_model)
        
        # Initialize models
        print(f"Loading TSR model: {tsr_model}")
        self.tsr_processor = create_table_processor(tsr_model)
        
        print(f"Loading summarization model: {summarization_model}")
        self.summarizer = create_table_summarizer(summarization_model)
        
        print(f"Loading figure model: {figure_model}")
        self.figure_processor = create_figure_processor(figure_model)
        
        print("✓ Stage 3 models loaded")
    
    def process_image(
        self,
        image_path: str | Path,
        stage1_predictions: List[Dict[str, Any]],
        save_intermediate: bool = False,
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Process image for tables and figures.
        
        Args:
            image_path: Path to input image
            stage1_predictions: Predictions from Stage 1
            save_intermediate: Whether to save intermediate results
            output_dir: Directory for intermediate outputs
        
        Returns:
            Dictionary with processed tables and figures
        """
        image_path = Path(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Separate tables and figures
        tables = [p for p in stage1_predictions if p.get('class') == 4]
        figures = [p for p in stage1_predictions if p.get('class') == 5]
        
        results = {
            "image": image_path.name,
            "tables": [],
            "figures": []
        }
        
        # Process tables
        if tables:
            print(f"Processing {len(tables)} table(s)...")
            table_regions = crop_regions_from_predictions(image, tables, category_filter=[4])
            
            for region in table_regions:
                try:
                    # TSR
                    table_result = self.tsr_processor.process(region['cropped_image'])
                    structured_data = table_result.get("structured_data", {})
                    
                    # Summarization
                    summary = self.summarizer.summarize(structured_data)
                    
                    results["tables"].append({
                        "bbox": region['bbox'],
                        "structured_data": structured_data,
                        "summary": summary
                    })
                except Exception as e:
                    print(f"Error processing table: {e}")
                    results["tables"].append({
                        "bbox": region['bbox'],
                        "error": str(e)
                    })
        
        # Process figures
        if figures:
            print(f"Processing {len(figures)} figure(s)...")
            figure_regions = crop_regions_from_predictions(image, figures, category_filter=[5])
            
            for region in figure_regions:
                try:
                    # Caption generation
                    caption_result = self.figure_processor.process(region['cropped_image'])
                    
                    results["figures"].append({
                        "bbox": region['bbox'],
                        "caption": caption_result.get("caption", ""),
                        "confidence": caption_result.get("confidence", 0.0)
                    })
                except Exception as e:
                    print(f"Error processing figure: {e}")
                    results["figures"].append({
                        "bbox": region['bbox'],
                        "error": str(e)
                    })
        
        return results
    
    def process_stage1_output(
        self,
        stage1_json_path: str | Path,
        image_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Process Stage 1 output JSON file.
        
        Args:
            stage1_json_path: Path to Stage 1 output JSON
            image_dir: Directory containing images
        
        Returns:
            Processed results
        """
        import json
        
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
        
        # Process
        return self.process_image(image_path, predictions)


