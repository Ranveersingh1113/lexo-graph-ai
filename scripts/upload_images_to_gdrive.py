#!/usr/bin/env python3
"""
Upload PS05 COCO images to Google Drive.
Requires: pip install pydrive2

Setup:
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials.json and place in project root
6. Run this script - first time will open browser for authentication
"""

import os
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

def setup_drive():
    """Setup Google Drive authentication"""
    gauth = GoogleAuth()
    
    # Try to load saved client credentials
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

def create_folder(drive, folder_name, parent_folder_id=None):
    """Create a folder in Google Drive, return folder ID"""
    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        folder_metadata['parents'] = [{'id': parent_folder_id}]
    
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder['id']

def find_or_create_folder(drive, folder_name, parent_id=None):
    """Find existing folder or create new one"""
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    file_list = drive.ListFile({'q': query}).GetList()
    
    if file_list:
        return file_list[0]['id']
    else:
        return create_folder(drive, folder_name, parent_id)

def upload_images():
    """Upload all images from data/ps05_coco/images/ to Google Drive"""
    images_dir = Path("data/ps05_coco/images")
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist!")
        return
    
    # Get all PNG files
    image_files = list(images_dir.glob("*.png"))
    total_images = len(image_files)
    
    if total_images == 0:
        print("No images found!")
        return
    
    print(f"Found {total_images} images to upload")
    
    # Setup Google Drive
    print("\nSetting up Google Drive authentication...")
    try:
        drive = setup_drive()
    except Exception as e:
        print(f"Error setting up Google Drive: {e}")
        print("\nMake sure you have:")
        print("1. credentials.json in project root")
        print("2. pydrive2 installed: pip install pydrive2")
        return
    
    # Create folder structure
    print("\nCreating folder structure in Google Drive...")
    base_folder_id = find_or_create_folder(drive, "lexo-graph-ai")
    ps05_folder_id = find_or_create_folder(drive, "ps05_coco", base_folder_id)
    images_folder_id = find_or_create_folder(drive, "images", ps05_folder_id)
    
    print(f"Uploading to: lexo-graph-ai/ps05_coco/images/")
    print(f"Folder ID: {images_folder_id}")
    
    # Upload images
    uploaded = 0
    skipped = 0
    
    for img_file in tqdm(image_files, desc="Uploading images"):
        try:
            # Check if file already exists
            query = f"title='{img_file.name}' and '{images_folder_id}' in parents and trashed=false"
            existing = drive.ListFile({'q': query}).GetList()
            
            if existing:
                print(f"\nSkipping {img_file.name} (already exists)")
                skipped += 1
                continue
            
            # Create file metadata
            file_drive = drive.CreateFile({
                'title': img_file.name,
                'parents': [{'id': images_folder_id}]
            })
            
            # Set content and upload
            file_drive.SetContentFile(str(img_file))
            file_drive.Upload()
            
            uploaded += 1
            
        except Exception as e:
            print(f"\nError uploading {img_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Total: {total_images}")
    print(f"{'='*60}")
    print(f"\nGoogle Drive folder ID: {images_folder_id}")
    print("Save this ID for downloading images later!")

if __name__ == "__main__":
    upload_images()

