"""
Step 4: Merge multiple COCO datasets into one unified dataset

This script merges:
- data/ps05_coco/annotations/train.json
- data/doclaynet/annotations/train.json (or similar)
- data/publaynet/annotations/train.json (or similar)

Into a single COCO dataset with unified category labels.
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Master category list for PS-05 competition
MASTER_CATEGORIES = [
    {"id": 0, "name": "Background", "supercategory": "none"},
    {"id": 1, "name": "Text", "supercategory": "text"},
    {"id": 2, "name": "Title", "supercategory": "text"},
    {"id": 3, "name": "List", "supercategory": "text"},
    {"id": 4, "name": "Table", "supercategory": "table"},
    {"id": 5, "name": "Figure", "supercategory": "figure"},
]

# Category name to master ID mapping
CATEGORY_NAME_TO_ID = {cat["name"].lower(): cat["id"] for cat in MASTER_CATEGORIES}

# Category mapping for different datasets
# DocLayNet category mappings
DOCLAYNET_MAPPING = {
    "text": 1,      # Text
    "title": 2,     # Title
    "list": 3,      # List
    "table": 4,     # Table
    "figure": 5,    # Figure
}

# PubLayNet category mappings
PUBLAYNET_MAPPING = {
    "text": 1,      # Text
    "title": 2,     # Title
    "list": 3,      # List
    "table": 4,     # Table
    "figure": 5,    # Figure
}


def load_coco_dataset(json_path):
    """Load a COCO JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def map_category(category_name, category_id, dataset_name):
    """
    Map a category from source dataset to master category ID.
    
    Args:
        category_name: Name of the category
        category_id: Original category ID
        dataset_name: Name of the source dataset ('ps05', 'doclaynet', 'publaynet')
    
    Returns:
        Master category ID or None if category should be skipped
    """
    category_name_lower = category_name.lower() if category_name else ""
    
    if dataset_name == "ps05":
        # PS05 already uses the same category IDs
        if category_id in [0, 1, 2, 3, 4, 5]:
            return category_id
        return None
    
    elif dataset_name == "doclaynet":
        # Try mapping by name
        if category_name_lower in DOCLAYNET_MAPPING:
            return DOCLAYNET_MAPPING[category_name_lower]
        # Try mapping by ID (if category_id corresponds to a known category)
        # DocLayNet typically has categories: Text, Title, List, Table, Figure
        # Map common IDs
        if category_id < len(DOCLAYNET_MAPPING):
            mapped_names = list(DOCLAYNET_MAPPING.keys())
            if category_id < len(mapped_names):
                return DOCLAYNET_MAPPING[mapped_names[category_id]]
        return None
    
    elif dataset_name == "publaynet":
        # Try mapping by name
        if category_name_lower in PUBLAYNET_MAPPING:
            return PUBLAYNET_MAPPING[category_name_lower]
        # PubLayNet has: Text, Title, List, Table, Figure
        # Try common mappings
        if category_id < len(PUBLAYNET_MAPPING):
            mapped_names = list(PUBLAYNET_MAPPING.keys())
            if category_id < len(mapped_names):
                return PUBLAYNET_MAPPING[mapped_names[category_id]]
        return None
    
    return None


def merge_coco_datasets(dataset_paths, output_dir):
    """
    Merge multiple COCO datasets into one.
    
    Args:
        dataset_paths: List of tuples (dataset_name, json_path, images_dir)
        output_dir: Path to output directory
    """
    output_path = Path(output_dir)
    annotations_dir = output_path / "annotations"
    images_dir = output_path / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize merged COCO structure
    merged_coco = {
        "info": {
            "description": "Merged COCO dataset for PS-05 (PS05 + DocLayNet + PubLayNet)",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": MASTER_CATEGORIES
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    stats = defaultdict(lambda: {"images": 0, "annotations": 0, "skipped": 0})
    
    print("=" * 60)
    print("Merging COCO datasets...")
    print("=" * 60)
    
    for dataset_name, json_path, images_source_dir in dataset_paths:
        print(f"\nProcessing {dataset_name}...")
        
        if not os.path.exists(json_path):
            print(f"  ⚠ Warning: {json_path} not found, skipping...")
            continue
        
        if not os.path.exists(images_source_dir):
            print(f"  ⚠ Warning: {images_source_dir} not found, skipping...")
            continue
        
        try:
            coco_data = load_coco_dataset(json_path)
        except Exception as e:
            print(f"  ✗ Error loading {json_path}: {e}")
            continue
        
        # Build category mapping from source dataset
        source_categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
        
        # Process images
        print(f"  Processing {len(coco_data.get('images', []))} images...")
        for img_entry in tqdm(coco_data.get("images", []), desc=f"  {dataset_name} images"):
            source_filename = img_entry["file_name"]
            source_image_path = Path(images_source_dir) / source_filename
            
            if not source_image_path.exists():
                print(f"    ⚠ Image not found: {source_image_path}")
                continue
            
            # Create new image entry with offset ID
            new_image_id = image_id_offset + img_entry["id"]
            new_filename = f"{dataset_name}_{source_filename}"
            
            # Copy image to merged directory
            dest_image_path = images_dir / new_filename
            try:
                shutil.copy2(source_image_path, dest_image_path)
            except Exception as e:
                print(f"    ✗ Error copying {source_filename}: {e}")
                continue
            
            new_image_entry = {
                "id": new_image_id,
                "file_name": new_filename,
                "width": img_entry["width"],
                "height": img_entry["height"]
            }
            merged_coco["images"].append(new_image_entry)
            stats[dataset_name]["images"] += 1
        
        # Process annotations
        print(f"  Processing {len(coco_data.get('annotations', []))} annotations...")
        for ann_entry in tqdm(coco_data.get("annotations", []), desc=f"  {dataset_name} annotations"):
            source_category_id = ann_entry["category_id"]
            source_category_name = source_categories.get(source_category_id, "")
            
            # Map to master category
            master_category_id = map_category(
                source_category_name,
                source_category_id,
                dataset_name
            )
            
            if master_category_id is None:
                stats[dataset_name]["skipped"] += 1
                continue
            
            # Get corresponding image ID (with offset)
            source_image_id = ann_entry["image_id"]
            new_image_id = image_id_offset + source_image_id
            
            # Check if image exists in merged dataset
            image_ids = {img["id"] for img in merged_coco["images"]}
            if new_image_id not in image_ids:
                continue
            
            # Create new annotation entry
            new_annotation_id = annotation_id_offset + ann_entry["id"]
            new_ann_entry = {
                "id": new_annotation_id,
                "image_id": new_image_id,
                "category_id": master_category_id,
                "bbox": ann_entry["bbox"],  # Already in [x, y, w, h] format
                "area": ann_entry.get("area", ann_entry["bbox"][2] * ann_entry["bbox"][3]),
                "iscrowd": ann_entry.get("iscrowd", 0)
            }
            merged_coco["annotations"].append(new_ann_entry)
            stats[dataset_name]["annotations"] += 1
        
        # Update offsets for next dataset
        if coco_data.get("images"):
            image_id_offset += max(img["id"] for img in coco_data["images"]) + 1
        if coco_data.get("annotations"):
            annotation_id_offset += max(ann["id"] for ann in coco_data["annotations"]) + 1
    
    # Save merged COCO JSON
    output_json = annotations_dir / "train.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_coco, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nTotal merged images: {len(merged_coco['images'])}")
    print(f"Total merged annotations: {len(merged_coco['annotations'])}")
    print(f"\nOutput JSON: {output_json}")
    print(f"Output images: {images_dir}")
    
    print("\nPer-dataset statistics:")
    for dataset_name, stat in stats.items():
        print(f"  {dataset_name}:")
        print(f"    Images: {stat['images']}")
        print(f"    Annotations: {stat['annotations']}")
        print(f"    Skipped annotations: {stat['skipped']}")
    
    # Category statistics
    print("\nCategory statistics (merged):")
    from collections import Counter
    category_counts = Counter([ann["category_id"] for ann in merged_coco["annotations"]])
    for cat in MASTER_CATEGORIES:
        count = category_counts.get(cat["id"], 0)
        print(f"  {cat['name']} (id={cat['id']}): {count} annotations")


if __name__ == "__main__":
    # Define dataset paths
    # Format: (dataset_name, json_path, images_dir)
    dataset_paths = [
        ("ps05", "data/ps05_coco/annotations/train.json", "data/ps05_coco/images"),
        ("doclaynet", "data/doclaynet/annotations/train.json", "data/doclaynet/images"),
        ("publaynet", "data/publaynet/annotations/train.json", "data/publaynet/images"),
    ]
    
    # Filter out non-existent datasets
    existing_paths = []
    for name, json_path, img_dir in dataset_paths:
        if os.path.exists(json_path) and os.path.exists(img_dir):
            existing_paths.append((name, json_path, img_dir))
        else:
            print(f"⚠ Skipping {name}: paths not found")
            print(f"  JSON: {json_path}")
            print(f"  Images: {img_dir}")
    
    if not existing_paths:
        print("Error: No valid datasets found to merge!")
        print("\nPlease ensure:")
        print("  1. Run convert_ps05_to_coco.py first")
        print("  2. Download public datasets using download_public_data.py")
        exit(1)
    
    output_dir = "data/merged_coco"
    merge_coco_datasets(existing_paths, output_dir)
