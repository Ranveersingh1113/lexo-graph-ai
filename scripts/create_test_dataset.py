#!/usr/bin/env python3
"""
Create test dataset for Phase 3 model evaluation.

This script helps create a test dataset JSON file from your images with ground truth annotations.

Usage:
    python scripts/create_test_dataset.py \
        --images_dir images/ \
        --output test_data.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_test_dataset_template(
    images_dir: str,
    output_path: str,
    image_type: str = "both"  # "tables", "figures", or "both"
):
    """
    Create a template test dataset JSON file.
    
    Args:
        images_dir: Directory containing test images
        output_path: Path to save test dataset JSON
        image_type: Type of images ("tables", "figures", or "both")
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    
    # Find images
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    # Create template
    dataset = {
        "tables": [],
        "figures": []
    }
    
    for image_file in image_files:
        if image_type in ["tables", "both"]:
            dataset["tables"].append({
                "image_path": str(image_file),
                "ground_truth": {
                    "headers": [],  # Fill in manually
                    "rows": []      # Fill in manually
                }
            })
        
        if image_type in ["figures", "both"]:
            dataset["figures"].append({
                "image_path": str(image_file),
                "ground_truth": ""  # Fill in manually
            })
    
    # Save template
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created test dataset template with {len(dataset['tables'])} tables and {len(dataset['figures'])} figures")
    print(f"Saved to: {output_path}")
    print("\nNext steps:")
    print("1. Fill in ground truth annotations for each image")
    print("2. For tables: Add headers and rows data")
    print("3. For figures: Add caption text")
    print("4. Run evaluation: python scripts/evaluate_phase3_models.py --test_data test_data.json")


def main():
    parser = argparse.ArgumentParser(
        description="Create test dataset template for Phase 3 evaluation"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["tables", "figures", "both"],
        default="both",
        help="Type of images to include"
    )
    
    args = parser.parse_args()
    
    create_test_dataset_template(args.images_dir, args.output, args.type)


if __name__ == "__main__":
    main()


