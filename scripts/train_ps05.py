"""
Training script wrapper for PS05 document layout detection model.

This script:
- Validates environment and dataset
- Runs training with proper logging
- Handles errors gracefully
- Saves training logs to file
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("=" * 60)
    print("Prerequisites Check")
    print("=" * 60)
    
    issues = []
    
    # Check conda environment
    if not os.environ.get('CONDA_DEFAULT_ENV'):
        issues.append("WARNING: Not in conda environment. Recommended: conda activate doc-comp")
    
    # Check dataset
    dataset_path = Path("data/ps05_coco/annotations/train.json")
    if not dataset_path.exists():
        issues.append(f"ERROR: Dataset not found at {dataset_path}")
    
    # Check images directory
    images_dir = Path("data/ps05_coco/images")
    if not images_dir.exists():
        issues.append(f"ERROR: Images directory not found at {images_dir}")
    else:
        image_count = len(list(images_dir.glob("*.png")))
        if image_count == 0:
            issues.append(f"WARNING: No PNG images found in {images_dir}")
        else:
            print(f"  ✓ Found {image_count} images")
    
    # Check config files
    config_path = Path("PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml")
    if not config_path.exists():
        issues.append(f"ERROR: Config file not found at {config_path}")
    else:
        print(f"  ✓ Config file found")
    
    # Check PaddleDetection
    paddle_dir = Path("PaddleDetection")
    if not paddle_dir.exists():
        issues.append("ERROR: PaddleDetection directory not found")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        if any("ERROR" in i for i in issues):
            print("\nPlease fix errors before proceeding.")
            return False
    else:
        print("  ✓ All prerequisites met!")
    
    print("=" * 60)
    return True

def run_training(config_path, use_gpu=True, resume=None, batch_size=None, log_dir=None):
    """Run the training command."""
    
    # Set up logging directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"training_logs/ps05_training_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"GPU: {use_gpu}")
    print(f"Log directory: {log_dir}")
    print(f"Log file: {log_file}")
    if resume:
        print(f"Resuming from: {resume}")
    if batch_size:
        print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")
    
    # Ensure we're in the project root directory
    # Change to PaddleDetection directory for training
    original_dir = os.getcwd()
    paddle_dir = Path("PaddleDetection").resolve()
    
    if not paddle_dir.exists():
        print(f"ERROR: PaddleDetection directory not found at {paddle_dir}")
        return False
    
    # Build training command (config path relative to PaddleDetection directory)
    # If config_path is absolute or starts with PaddleDetection/, make it relative
    if Path(config_path).is_absolute():
        try:
            config_rel_path = Path(config_path).relative_to(paddle_dir)
        except ValueError:
            # If not relative to paddle_dir, use as-is
            config_rel_path = Path(config_path)
    elif str(config_path).startswith("PaddleDetection/"):
        config_rel_path = str(config_path).replace("PaddleDetection/", "")
    else:
        config_rel_path = config_path
    
    cmd = [
        sys.executable,
        "tools/train.py",
        "-c", str(config_rel_path),
        "-o", f"use_gpu={use_gpu}"
    ]
    
    if resume:
        cmd.extend(["-r", resume])
    
    if batch_size:
        cmd.extend(["-o", f"TrainReader.batch_size={batch_size}"])
    
    # Print command
    print(f"Running command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run training with logging
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write(f"PS05 Training Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            # Change to PaddleDetection directory before running
            os.chdir(str(paddle_dir))
            
            # Run training and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(paddle_dir)
            )
            
            # Stream output to both console and file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\nERROR: Training failed with return code {process.returncode}")
                print(f"Check log file: {log_file}")
                os.chdir(original_dir)  # Restore original directory
                return False
            else:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n")
                print(f"\n✓ Training completed successfully!")
                print(f"Logs saved to: {log_file}")
                os.chdir(original_dir)  # Restore original directory
                return True
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial logs saved to: {log_file}")
        os.chdir(original_dir)  # Restore original directory
        return False
    except Exception as e:
        print(f"\nERROR: Failed to run training: {e}")
        print(f"Check log file: {log_file}")
        os.chdir(original_dir)  # Restore original directory
        return False

def main():
    parser = argparse.ArgumentParser(description="Train PS05 document layout detection model")
    parser.add_argument(
        "-c", "--config",
        default="PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "-r", "--resume",
        help="Resume training from checkpoint path"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        help="Override batch size (default: from config)"
    )
    parser.add_argument(
        "--log-dir",
        help="Directory to save training logs"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    # Change to project root if needed
    if not Path("PaddleDetection").exists() and Path("../PaddleDetection").exists():
        os.chdir("..")
    
    # Check prerequisites
    if not args.skip_check:
        if not check_prerequisites():
            sys.exit(1)
    
    # Run training
    success = run_training(
        config_path=args.config,
        use_gpu=not args.cpu,
        resume=args.resume,
        batch_size=args.batch_size,
        log_dir=args.log_dir
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

