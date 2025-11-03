"""
Step 2: Convert train_PS05 dataset to COCO format

This script converts the train_PS05/train/ dataset (paired .png and .json files)
into a single COCO-formatted JSON file and organizes images in the required structure.
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Competition class definitions
PS05_CATEGORIES = [
    {"id": 0, "name": "Background", "supercategory": "none"},
    {"id": 1, "name": "Text", "supercategory": "text"},
    {"id": 2, "name": "Title", "supercategory": "text"},
    {"id": 3, "name": "List", "supercategory": "text"},
    {"id": 4, "name": "Table", "supercategory": "table"},
    {"id": 5, "name": "Figure", "supercategory": "figure"},
]

# Category mapping from PS05 category_id to COCO category_id
# The PS05 dataset uses category_id directly matching the competition classes
CATEGORY_ID_MAP = {
    0: 0,  # Background
    1: 1,  # Text
    2: 2,  # Title
    3: 3,  # List
    4: 4,  # Table
    5: 5,  # Figure
}


def convert_ps05_to_coco(input_dir, output_dir):
    """
    Convert PS05 dataset to COCO format.
    
    Args:
        input_dir: Path to train_PS05/train/ directory
        output_dir: Path to output directory (data/ps05_coco/)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    annotations_dir = output_path / "annotations"
    images_dir = output_path / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "PS-05 Training Dataset in COCO format",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": PS05_CATEGORIES
    }
    
    # Find all PNG files
    png_files = sorted(input_path.glob("*.png"))
    
    image_id = 1
    annotation_id = 1
    
    print(f"Found {len(png_files)} PNG files. Converting to COCO format...")
    
    for png_file in tqdm(png_files, desc="Converting images"):
        json_file = png_file.with_suffix(".json")
        
        if not json_file.exists():
            print(f"Warning: No JSON file found for {png_file.name}, skipping...")
            continue
        
        # Read JSON annotation file
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}, skipping...")
            continue
        
        # Get image dimensions (we'll read from the actual image)
        try:
            import cv2
            img = cv2.imread(str(png_file))
            if img is None:
                print(f"Warning: Could not read image {png_file.name}, skipping...")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {png_file.name}: {e}, skipping...")
            continue
        
        # Copy image to output directory
        output_image_path = images_dir / png_file.name
        shutil.copy2(png_file, output_image_path)
        
        # Add image entry
        image_entry = {
            "id": image_id,
            "file_name": png_file.name,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_entry)
        
        # Process annotations
        if "annotations" in data and isinstance(data["annotations"], list):
            for ann in data["annotations"]:
                if "bbox" not in ann or "category_id" not in ann:
                    continue
                
                bbox = ann["bbox"]
                if len(bbox) != 4:
                    continue
                
                # PS05 format: [x, y, w, h]
                x, y, w, h = bbox
                category_id = ann["category_id"]
                
                # Map category_id (should already match, but validate)
                if category_id not in CATEGORY_ID_MAP:
                    print(f"Warning: Unknown category_id {category_id} in {png_file.name}, skipping annotation...")
                    continue
                
                coco_category_id = CATEGORY_ID_MAP[category_id]
                
                # Ensure bbox is within image bounds
                x = max(0, min(x, width))
                y = max(0, min(y, height))
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                # COCO format: [x, y, width, height] and area
                area = w * h
                
                # Add annotation entry
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": coco_category_id,
                    "bbox": [x, y, w, h],  # COCO format: [x, y, width, height]
                    "area": area,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation_entry)
                annotation_id += 1
        
        image_id += 1
    
    # Save COCO JSON
    output_json = annotations_dir / "train.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Output JSON: {output_json}")
    print(f"Output images: {images_dir}")
    
    # Print category statistics
    print("\nCategory statistics:")
    from collections import Counter
    category_counts = Counter([ann["category_id"] for ann in coco_data["annotations"]])
    for cat in PS05_CATEGORIES:
        count = category_counts.get(cat["id"], 0)
        print(f"  {cat['name']} (id={cat['id']}): {count} annotations")


if __name__ == "__main__":
    # Paths
    input_dir = "train_PS05/train"
    output_dir = "data/ps05_coco"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        print("Please ensure train_PS05/train/ directory is in the project root.")
        exit(1)
    
    convert_ps05_to_coco(input_dir, output_dir)
