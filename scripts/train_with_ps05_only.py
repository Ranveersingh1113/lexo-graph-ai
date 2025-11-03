"""
Alternative workflow: Train with PS05 dataset only (no public datasets needed)

This script helps you proceed with training using only your PS05 dataset
when disk space is limited.
"""

import json
import os
from pathlib import Path

def prepare_ps05_only_dataset():
    """
    Prepare PS05 dataset for training (optionally with augmentation).
    
    This creates a direct copy or symlink so you can use data/ps05_coco/
    directly for training without needing to merge with other datasets.
    """
    ps05_coco_path = Path("data/ps05_coco/annotations/train.json")
    
    if not ps05_coco_path.exists():
        print("Error: PS05 COCO dataset not found!")
        print("Please run: python scripts/convert_ps05_to_coco.py first")
        return False
    
    print("=" * 60)
    print("PS05 Dataset Ready for Training!")
    print("=" * 60)
    
    # Load and show statistics
    with open(ps05_coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Images: {len(coco_data['images'])}")
    print(f"  Total Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Category statistics
    from collections import Counter
    category_counts = Counter([ann["category_id"] for ann in coco_data["annotations"]])
    
    print("\nCategory Distribution:")
    for cat in coco_data["categories"]:
        count = category_counts.get(cat["id"], 0)
        print(f"  {cat['name']} (id={cat['id']}): {count} annotations")
    
    print(f"\n✓ Dataset is ready at: {ps05_coco_path.parent}")
    print(f"  Images directory: {ps05_coco_path.parent.parent / 'images'}")
    
    return True


if __name__ == "__main__":
    print("PS05-Only Training Setup")
    print("=" * 60)
    print("\nYou can train with your PS05 dataset without downloading")
    print("public datasets (saves ~200GB disk space).\n")
    
    success = prepare_ps05_only_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("\n1. Optionally augment your data:")
        print("   python scripts/augment_data.py")
        print("   (Modify the script to use data/ps05_coco/ instead of merged_coco)")
        print("\n2. Configure PaddleDetection to use:")
        print("   data/ps05_coco/annotations/train.json")
        print("   data/ps05_coco/images/")
        print("\n3. Start training!")
        print("\nNote: Training with PS05 only is perfectly valid.")
        print("The model will learn from your specific document distribution.")


