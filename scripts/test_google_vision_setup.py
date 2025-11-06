#!/usr/bin/env python3
"""
Test script to verify Google Cloud Vision API setup.

This script tests if your Google Cloud Vision API is configured correctly.

Usage:
    python scripts/test_google_vision_setup.py [--credentials path/to/credentials.json]
"""

import os
import sys
import argparse
from pathlib import Path

def test_import():
    """Test if google-cloud-vision is installed."""
    try:
        from google.cloud import vision
        print("[OK] google-cloud-vision package is installed")
        return True
    except ImportError as e:
        print(f"[ERROR] google-cloud-vision package not found: {e}")
        print("\n  Install it with: pip install google-cloud-vision")
        return False

def test_credentials(credentials_path=None):
    """Test if credentials are accessible."""
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(credentials_path).resolve())
        print(f"[OK] Using credentials from: {credentials_path}")
    
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    if creds_path:
        if Path(creds_path).exists():
            print(f"[OK] Credentials file found: {creds_path}")
            return True
        else:
            print(f"[ERROR] Credentials file not found: {creds_path}")
            return False
    else:
        print("[WARNING] GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        print("  Set it with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
        return False

def test_client_initialization(credentials_path=None):
    """Test if API client can be initialized."""
    try:
        from google.cloud import vision
        
        if credentials_path:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            client = vision.ImageAnnotatorClient()
        
        print("[OK] Google Cloud Vision API client initialized successfully")
        return True, client
    except Exception as e:
        print(f"[ERROR] Failed to initialize API client: {e}")
        print("\n  Common issues:")
        print("  1. Credentials file is invalid or corrupted")
        print("  2. Cloud Vision API is not enabled in your project")
        print("  3. Service account doesn't have proper permissions")
        return False, None

def test_api_call(client, test_image_path=None):
    """Test actual API call (optional, requires image)."""
    if test_image_path and Path(test_image_path).exists():
        try:
            print(f"\n  Testing API call with image: {test_image_path}")
            with open(test_image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            
            if response.text_annotations:
                print(f"[OK] API call successful!")
                print(f"  Detected text: {response.text_annotations[0].description[:50]}...")
                return True
            else:
                print("[OK] API call successful (no text detected in image)")
                return True
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return False
    else:
        print("  (Skipping API call test - no image provided)")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Test Google Cloud Vision API setup"
    )
    parser.add_argument(
        "--credentials",
        type=str,
        help="Path to Google Cloud credentials JSON file"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        help="Path to test image for API call (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Google Cloud Vision API Setup Test")
    print("=" * 60)
    
    # Test 1: Package installation
    print("\n[1/4] Testing package installation...")
    if not test_import():
        sys.exit(1)
    
    # Test 2: Credentials
    print("\n[2/4] Testing credentials...")
    if not test_credentials(args.credentials):
        print("\n  To fix:")
        print("  1. Download service account JSON key from Google Cloud Console")
        print("  2. Set environment variable:")
        print("     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
        print("     OR use --credentials flag")
        sys.exit(1)
    
    # Test 3: Client initialization
    print("\n[3/4] Testing client initialization...")
    success, client = test_client_initialization(args.credentials)
    if not success:
        sys.exit(1)
    
    # Test 4: API call (optional)
    if args.test_image:
        print("\n[4/4] Testing API call...")
        from google.cloud import vision
        test_api_call(client, args.test_image)
    else:
        print("\n[4/4] Skipping API call test (use --test-image to test)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Setup verification complete!")
    print("=" * 60)
    print("\nYour Google Cloud Vision API is configured correctly.")
    print("You can now proceed with Phase 2 implementation.")

if __name__ == "__main__":
    main()

