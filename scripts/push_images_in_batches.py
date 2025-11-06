#!/usr/bin/env python3
"""
Script to push images to Git in smaller batches to avoid GitHub LFS quota/timeout issues.
Splits 8,000 images into batches of 500 images each.
"""

import subprocess
import os
from pathlib import Path

def get_image_files():
    """Get all PNG image files from data/ps05_coco/images/"""
    image_dir = Path("data/ps05_coco/images")
    images = sorted([f for f in image_dir.glob("*.png")])
    return images

def push_batch(batch_num, image_files):
    """Add and commit a batch of images"""
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}: {len(image_files)} images")
    print(f"{'='*60}")
    
    # Add images
    for img in image_files:
        subprocess.run(["git", "add", str(img)], check=True)
    
    # Commit
    commit_msg = f"Add PS05 images batch {batch_num} ({len(image_files)} images)"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    
    # Push
    print(f"\nPushing batch {batch_num}...")
    try:
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"✓ Batch {batch_num} pushed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Batch {batch_num} push failed: {e}")
        return False

def main():
    """Main function to push images in batches"""
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    # Get all images
    all_images = get_image_files()
    total_images = len(all_images)
    print(f"Total images to push: {total_images}")
    
    # Batch size (500 images per batch = ~175 MB per batch)
    batch_size = 500
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"Will push in {num_batches} batches of ~{batch_size} images each")
    
    # Process batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = all_images[start_idx:end_idx]
        
        success = push_batch(i + 1, batch_images)
        
        if not success:
            print(f"\n⚠ Batch {i + 1} failed. You can retry later.")
            print(f"Remaining images: {total_images - end_idx}")
            break
        
        # Small delay between batches
        import time
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print("Batch push complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

