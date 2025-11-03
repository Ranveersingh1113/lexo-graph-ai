"""
Step 5: Data Augmentation using Augraphy

This script applies realistic document augmentation to the PS05 COCO dataset,
creating "dirty" versions of images to improve model robustness.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from augraphy import (
    AugraphyPipeline,
    InkBleed,
    DirtyDrum,
    DepthSimulatedBlur,
    Faxify,
    Folding,
    LightingGradient,
    LowInkRandomLines,
    NoiseTexturize,
)

# Augmentation pipeline configuration
# Using post_phase for most augmentations (applied after ink/paper phases)
AUG_PIPELINE = AugraphyPipeline(
    post_phase=[
        InkBleed(intensity_range=(0.1, 0.3), p=0.3),
        DirtyDrum(line_width_range=(1, 2), p=0.3),
        DepthSimulatedBlur(p=0.3),
        Faxify(scale_range=(0.5, 1.0), monochrome=0, p=0.3),
        Folding(fold_count=2, fold_noise=0.1, p=0.2),
        LightingGradient(p=0.3),
        LowInkRandomLines(count_range=(1, 3), p=0.2),
        NoiseTexturize(p=0.2),
    ]
)


def augment_image(image_path, output_path):
    """
    Apply augmentation pipeline to an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save augmented image
    
    Returns:
        Success status
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # Apply augmentation
        augmented = AUG_PIPELINE(image)
        
        # Save augmented image
        cv2.imwrite(str(output_path), augmented)
        return True
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False


def augment_coco_dataset(coco_json_path, images_dir, output_json_path=None):
    """
    Augment all images in a COCO dataset and update annotations.
    
    Args:
        coco_json_path: Path to COCO train.json
        images_dir: Directory containing images
        output_json_path: Path to save updated COCO JSON (default: same as input)
    """
    images_dir = Path(images_dir)
    
    # Load COCO data
    print(f"Loading COCO dataset from {coco_json_path}...")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    if output_json_path is None:
        output_json_path = coco_json_path
    
    # Create new lists for augmented data
    new_images = []
    new_annotations = []
    
    # Copy existing data
    new_images.extend(coco_data["images"])
    new_annotations.extend(coco_data["annotations"])
    
    # Get ID offsets
    max_image_id = max([img["id"] for img in coco_data["images"]]) if coco_data["images"] else 0
    max_annotation_id = max([ann["id"] for ann in coco_data["annotations"]]) if coco_data["annotations"] else 0
    
    new_image_id = max_image_id + 1
    new_annotation_id = max_annotation_id + 1
    
    print(f"\nFound {len(coco_data['images'])} images to augment")
    print(f"Applying augmentation pipeline...")
    
    # Create mapping from image_id to annotations
    image_to_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_annotations:
            image_to_annotations[img_id] = []
        image_to_annotations[img_id].append(ann)
    
    # Process each image
    successful_augs = 0
    failed_augs = 0
    
    for img_entry in tqdm(coco_data["images"], desc="Augmenting images"):
        source_filename = img_entry["file_name"]
        source_image_path = images_dir / source_filename
        
        if not source_image_path.exists():
            print(f"  ⚠ Image not found: {source_image_path}")
            failed_augs += 1
            continue
        
        # Create augmented filename
        stem = Path(source_filename).stem
        suffix = Path(source_filename).suffix
        augmented_filename = f"{stem}_aug{suffix}"
        augmented_image_path = images_dir / augmented_filename
        
        # Apply augmentation
        success = augment_image(source_image_path, augmented_image_path)
        
        if not success:
            failed_augs += 1
            continue
        
        successful_augs += 1
        
        # Add new image entry
        new_image_entry = {
            "id": new_image_id,
            "file_name": augmented_filename,
            "width": img_entry["width"],
            "height": img_entry["height"]
        }
        new_images.append(new_image_entry)
        
        # Copy annotations for this image (same bboxes, new image_id)
        if img_entry["id"] in image_to_annotations:
            for ann in image_to_annotations[img_entry["id"]]:
                new_ann_entry = {
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"].copy(),  # Same bbox
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0)
                }
                new_annotations.append(new_ann_entry)
                new_annotation_id += 1
        
        new_image_id += 1
    
    # Update COCO data
    coco_data["images"] = new_images
    coco_data["annotations"] = new_annotations
    
    # Save updated COCO JSON
    print(f"\nSaving updated COCO JSON to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Augmentation complete!")
    print("=" * 60)
    print(f"\nOriginal images: {len(coco_data['images']) - successful_augs}")
    print(f"Augmented images: {successful_augs}")
    print(f"Failed augmentations: {failed_augs}")
    print(f"\nTotal images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"\nUpdated JSON: {output_json_path}")


if __name__ == "__main__":
    # Paths - Using PS05 dataset (not merged datasets)
    coco_json_path = "data/ps05_coco/annotations/train.json"
    images_dir = "data/ps05_coco/images"
    
    # Check if files exist
    if not os.path.exists(coco_json_path):
        print(f"Error: COCO JSON not found at {coco_json_path}")
        print("Please run convert_ps05_to_coco.py first!")
        exit(1)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        exit(1)
    
    print("=" * 60)
    print("Augmenting PS05 Dataset")
    print("=" * 60)
    print(f"Dataset: {coco_json_path}")
    print(f"Images: {images_dir}")
    print("\nThis will double your dataset size (~4,000 -> ~8,000 images)\n")
    
    # Run augmentation
    augment_coco_dataset(coco_json_path, images_dir)
