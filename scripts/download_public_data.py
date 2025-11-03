"""
Step 3: Download public COCO-formatted datasets using HuggingFace Hub

This script downloads:
- DocLayNet (ibm/doclaynet)
- PubLayNet (BUPT-PRIV/PubLayNet or alternative)

And saves them to the data/ directory.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

# Output directories
DATA_DIR = Path("data")
DOCLAYNET_DIR = DATA_DIR / "doclaynet"
PUBLAYNET_DIR = DATA_DIR / "publaynet"


def download_doclaynet(output_dir):
    """
    Download DocLayNet dataset from HuggingFace.
    
    Args:
        output_dir: Path to save DocLayNet data
    """
    print("=" * 60)
    print("Downloading DocLayNet dataset...")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # DocLayNet is available as a dataset
        print("Loading DocLayNet dataset from HuggingFace...")
        dataset = load_dataset("docling-project/DocLayNet-v1.1", "COCO")
        
        # Save training split
        if "train" in dataset:
            train_dir = output_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            
            annotations_dir = output_dir / "annotations"
            images_dir = output_dir / "images"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"DocLayNet train split contains {len(dataset['train'])} samples")
            print("Downloading images and annotations...")
            
            # Note: DocLayNet COCO format may require specific handling
            # The dataset might already be in COCO format or need conversion
            print("DocLayNet dataset structure:")
            print(f"  Features: {dataset['train'].features}")
            
            # Save the dataset (this will download and cache locally)
            # For COCO format, we typically need annotations/train.json and images/
            print("\nNote: DocLayNet will be cached by HuggingFace datasets.")
            print(f"Cache location: ~/.cache/huggingface/datasets/")
            print("\nTo access the data programmatically:")
            print("  from datasets import load_dataset")
            print("  dataset = load_dataset('ibm/doclaynet', 'COCO')")
            print("\nThe dataset will be automatically downloaded on first use.")
            
        print("✓ DocLayNet download initiated successfully")
        
    except Exception as e:
        print(f"Error downloading DocLayNet: {e}")
        print("\nTrying alternative method...")
        try:
            # Alternative: Direct snapshot download
            print("Trying snapshot_download...")
            snapshot_download(
                repo_id="ibm/doclaynet",
                repo_type="dataset",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False
            )
            print("✓ DocLayNet downloaded via snapshot_download")
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            print("Please check the HuggingFace repository: https://huggingface.co/datasets/ibm/doclaynet")


def download_publaynet(output_dir):
    """
    Download PubLayNet dataset from HuggingFace.
    
    Args:
        output_dir: Path to save PubLayNet data
    """
    print("\n" + "=" * 60)
    print("Downloading PubLayNet dataset...")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different possible repository names
    possible_repos = [
        "psyche/publaynet"
    ]
    
    downloaded = False
    for repo_id in possible_repos:
        try:
            print(f"\nTrying repository: {repo_id}...")
            dataset = load_dataset(repo_id)
            
            if dataset:
                print(f"✓ Successfully loaded dataset from {repo_id}")
                print(f"Dataset splits: {list(dataset.keys())}")
                
                if "train" in dataset:
                    print(f"Train split size: {len(dataset['train'])}")
                
                print("\nPubLayNet dataset structure:")
                print(f"  Features: {dataset.get('train', {}).features if 'train' in dataset else 'N/A'}")
                
                downloaded = True
                break
        except Exception as e:
            print(f"  Not available at {repo_id}: {e}")
            continue
    
    if not downloaded:
        print("\n⚠ Could not find PubLayNet on HuggingFace.")
        print("PubLayNet might need to be downloaded manually from:")
        print("  https://github.com/ibm-aur-nlp/PubLayNet")
        print("  or")
        print("  https://www.kaggle.com/datasets/eward96/publaynet")
        print("\nPlease download and place it in the data/publaynet/ directory.")
    
    return downloaded


def main():
    """Main function to download all public datasets."""
    print("Starting download of public datasets...")
    print(f"Output directory: {DATA_DIR.absolute()}")
    print()
    
    # Download DocLayNet
    download_doclaynet(DOCLAYNET_DIR)
    
    # Download PubLayNet
    # download_publaynet(PUBLAYNET_DIR)
    
    print("\n" + "=" * 60)
    print("Download process completed!")
    print("=" * 60)
    print("\nNote: Some datasets may be cached by HuggingFace.")
    print("If you need the actual files, you may need to:")
    print("  1. Check the HuggingFace cache: ~/.cache/huggingface/datasets/")
    print("  2. Or download manually and place in the respective data/ directories")
    print("\nFor merging, ensure the datasets are in COCO format with:")
    print("  - annotations/train.json")
    print("  - images/ folder")


if __name__ == "__main__":
    main()
