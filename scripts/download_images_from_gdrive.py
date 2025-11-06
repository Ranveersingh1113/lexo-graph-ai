#!/usr/bin/env python3
"""
Download PS05 COCO images from Google Drive.
Requires: pip install pydrive2

Usage:
1. Run upload_images_to_gdrive.py first to get the folder ID
2. Set GDRIVE_FOLDER_ID environment variable or pass as argument
3. Run this script to download all images
"""

import os
import sys
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

def setup_drive():
    """Setup Google Drive authentication"""
    gauth = GoogleAuth()
    
    # Check if credentials.json exists, if so copy it to client_secrets.json
    if os.path.exists("credentials.json") and not os.path.exists("client_secrets.json"):
        import shutil
        shutil.copy("credentials.json", "client_secrets.json")
        print("Copied credentials.json to client_secrets.json")
    
    # Try to load saved client credentials
    if os.path.exists("gdrive_credentials.json"):
        gauth.LoadCredentialsFile("gdrive_credentials.json")
    
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("gdrive_credentials.json")
    
    return GoogleDrive(gauth)

def download_images(folder_id):
    """Download all images from Google Drive folder"""
    images_dir = Path("data/ps05_coco/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up Google Drive authentication...")
    try:
        drive = setup_drive()
    except Exception as e:
        print(f"Error setting up Google Drive: {e}")
        print("\nMake sure you have:")
        print("1. credentials.json in project root")
        print("2. pydrive2 installed: pip install pydrive2")
        return
    
    # List all files in the folder
    print(f"\nFetching file list from folder ID: {folder_id}")
    query = f"'{folder_id}' in parents and trashed=false and mimeType='image/png'"
    file_list = drive.ListFile({'q': query}).GetList()
    
    total_files = len(file_list)
    print(f"Found {total_files} images to download")
    
    if total_files == 0:
        print("No images found in folder!")
        return
    
    # Download images
    downloaded = 0
    skipped = 0
    
    for file_drive in tqdm(file_list, desc="Downloading images"):
        file_path = images_dir / file_drive['title']
        
        # Skip if already exists
        if file_path.exists():
            skipped += 1
            continue
        
        try:
            file_drive.GetContentFile(str(file_path))
            downloaded += 1
        except Exception as e:
            print(f"\nError downloading {file_drive['title']}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Total: {total_files}")
    print(f"{'='*60}")

def main():
    """Main function"""
    # Get folder ID from environment or argument
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    
    if len(sys.argv) > 1:
        folder_id = sys.argv[1]
    
    if not folder_id:
        print("Error: Google Drive folder ID not provided!")
        print("\nUsage:")
        print("  python download_images_from_gdrive.py <FOLDER_ID>")
        print("  OR")
        print("  set GDRIVE_FOLDER_ID=<FOLDER_ID>")
        print("  python download_images_from_gdrive.py")
        print("\nGet the folder ID from upload_images_to_gdrive.py output")
        return
    
    download_images(folder_id)

if __name__ == "__main__":
    main()

