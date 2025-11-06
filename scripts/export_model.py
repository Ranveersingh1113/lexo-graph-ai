#!/usr/bin/env python3
"""
Export PaddleDetection trained model to inference format.

This script exports a trained checkpoint to a static inference model
that can be used for deployment and inference.

Usage:
    python scripts/export_model.py -c <config_path> -w <weights_path> [-o <output_dir>]
    
Example:
    python scripts/export_model.py \
        -c PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml \
        -w PaddleDetection/output/ppyoloe_plus_crn_s_80e_ps05/model_final.pdparams \
        -o models/inference/ppyoloe_ps05
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def export_model(config_path, weights_path, output_dir=None, use_gpu=True):
    """
    Export PaddleDetection model to inference format.
    
    Args:
        config_path: Path to training config file
        weights_path: Path to trained model weights (.pdparams)
        output_dir: Output directory for inference model (default: models/inference)
        use_gpu: Whether to use GPU for export
    """
    # Change to PaddleDetection directory
    original_dir = os.getcwd()
    paddle_dir = Path("PaddleDetection").resolve()
    
    if not paddle_dir.exists():
        print(f"ERROR: PaddleDetection directory not found at {paddle_dir}")
        return False
    
    # Make config path relative to PaddleDetection
    if Path(config_path).is_absolute():
        try:
            config_rel_path = Path(config_path).relative_to(paddle_dir)
        except ValueError:
            config_rel_path = Path(config_path)
    elif str(config_path).startswith("PaddleDetection/"):
        config_rel_path = str(config_path).replace("PaddleDetection/", "")
    else:
        config_rel_path = config_path
    
    # Set default output directory
    if output_dir is None:
        model_name = Path(config_rel_path).stem
        output_dir = f"models/inference/{model_name}"
    
    # Build export command
    cmd = [
        sys.executable,
        "tools/export_model.py",
        "-c", str(config_rel_path),
        "-o", f"weights={weights_path}",
        "-o", f"use_gpu={use_gpu}",
        "--output_dir", output_dir
    ]
    
    print("=" * 60)
    print("Exporting Model to Inference Format")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {output_dir}")
    print(f"GPU: {use_gpu}")
    print("=" * 60)
    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}\n")
    
    try:
        os.chdir(str(paddle_dir))
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✓ Model exported successfully!")
            print("=" * 60)
            print(f"\nInference model saved to: {paddle_dir / output_dir}")
            print(f"\nFiles created:")
            print(f"  - model.pdmodel (model structure)")
            print(f"  - model.pdiparams (model weights)")
            print(f"  - model.pdiparams.info (weights info)")
            print(f"  - infer_cfg.yml (inference config)")
            print("\nYou can now use this model for inference with run_stage1.py")
            os.chdir(original_dir)
            return True
        else:
            print(f"\nERROR: Export failed with return code {result.returncode}")
            os.chdir(original_dir)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Export failed: {e}")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"\nERROR: Failed to export model: {e}")
        os.chdir(original_dir)
        return False

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory."""
    output_path = Path("PaddleDetection") / output_dir
    
    # Look for model_final.pdparams
    final_model = output_path / "model_final.pdparams"
    if final_model.exists():
        return str(final_model)
    
    # Look for best model
    best_model = output_path / "best_model.pdparams"
    if best_model.exists():
        return str(best_model)
    
    # Look for latest epoch
    pdparam_files = list(output_path.glob("*.pdparams"))
    if pdparam_files:
        # Sort by modification time
        latest = max(pdparam_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Export PaddleDetection model to inference format"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_s_80e_ps05.yml",
        help="Path to training config file"
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        help="Path to trained model weights (.pdparams). "
             "If not provided, will try to find latest checkpoint from config."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for inference model (default: models/inference/<model_name>)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    args = parser.parse_args()
    
    # Find weights if not provided
    if args.weights is None:
        # Try to extract output dir from config
        config_path = Path(args.config)
        if config_path.exists():
            # Read config to find weights path
            with open(config_path, 'r') as f:
                content = f.read()
                # Look for weights: output/... pattern
                import re
                match = re.search(r'weights:\s*(output/[^\s]+)', content)
                if match:
                    output_dir_pattern = match.group(1)
                    weights = find_latest_checkpoint(output_dir_pattern)
                    if weights:
                        print(f"Found weights: {weights}")
                        args.weights = weights
                    else:
                        print(f"ERROR: Could not find weights in {output_dir_pattern}")
                        print("Please specify weights path with -w/--weights")
                        sys.exit(1)
        else:
            print("ERROR: Config file not found. Please specify weights with -w/--weights")
            sys.exit(1)
    
    # Check if weights file exists
    if not Path(args.weights).exists():
        print(f"ERROR: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Export model
    success = export_model(
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output_dir,
        use_gpu=not args.cpu
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


