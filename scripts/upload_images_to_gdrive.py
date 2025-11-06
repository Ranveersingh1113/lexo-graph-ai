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
    # Check if credentials.json exists, if so copy it to client_secrets.json
    if os.path.exists("credentials.json") and not os.path.exists("client_secrets.json"):
        import shutil
        shutil.copy("credentials.json", "client_secrets.json")
        print("Copied credentials.json to client_secrets.json")
    
    # Check if settings.yaml exists, create it if not
    if not os.path.exists("settings.yaml"):
        settings_content = """client_config_backend: file
client_config_file: client_secrets.json
save_credentials: True
save_credentials_backend: file
save_credentials_file: gdrive_credentials.json
get_refresh_token: True
"""
        with open("settings.yaml", "w") as f:
            f.write(settings_content)
        print("Created settings.yaml configuration file")
    
    gauth = GoogleAuth()
    
    # Load existing credentials if they exist
    if os.path.exists("gdrive_credentials.json"):
        try:
            gauth.LoadCredentialsFile("gdrive_credentials.json")
        except Exception:
            # If loading fails, credentials will be None
            pass
    
    if gauth.credentials is None:
        # Authenticate if they're not there
        # Request offline access to get refresh token
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline', 'prompt': 'consent'})
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        try:
            gauth.Refresh()
        except Exception as e:
            # If refresh fails, re-authenticate
            print(f"Token refresh failed: {e}")
            print("Re-authenticating...")
            gauth.GetFlow()
            gauth.flow.params.update({'access_type': 'offline', 'prompt': 'consent'})
            gauth.LocalWebserverAuth()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    
    # Save the current credentials to a file (this ensures refresh token is saved)
    gauth.SaveCredentialsFile("gdrive_credentials.json")
    
    return GoogleDrive(gauth), gauth

def refresh_token_if_needed(gauth, drive):
    """Refresh OAuth token if expired, return updated drive instance"""
    try:
        if gauth.access_token_expired:
            print("\nToken expired, refreshing...")
            # Check if we have a refresh token
            if gauth.credentials and gauth.credentials.refresh_token:
                gauth.Refresh()
                gauth.SaveCredentialsFile("gdrive_credentials.json")
                # Create a new drive instance with refreshed auth
                return GoogleDrive(gauth), gauth
            else:
                # No refresh token, need to re-authenticate
                print("No refresh token found. Re-authenticating with offline access...")
                gauth.GetFlow()
                gauth.flow.params.update({'access_type': 'offline', 'prompt': 'consent'})
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile("gdrive_credentials.json")
                return GoogleDrive(gauth), gauth
    except Exception as e:
        error_str = str(e)
        if "refresh_token" in error_str.lower() or "invalid_grant" in error_str.lower():
            print(f"\nError refreshing token: {e}")
            print("Re-authenticating with offline access...")
            gauth.GetFlow()
            gauth.flow.params.update({'access_type': 'offline', 'prompt': 'consent'})
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile("gdrive_credentials.json")
            return GoogleDrive(gauth), gauth
        else:
            # Re-raise if it's a different error
            raise
    return drive, gauth

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
        drive, gauth = setup_drive()
    except Exception as e:
        error_str = str(e)
        print(f"Error setting up Google Drive: {e}")
        
        if "refresh_token" in error_str.lower():
            print("\n⚠ No refresh token found. This usually means:")
            print("1. The credentials file was created without offline access")
            print("2. Solution: Delete gdrive_credentials.json and run the script again")
            print("   The script will re-authenticate and request offline access (refresh token)")
            print("\nTo fix:")
            print("  - Delete: gdrive_credentials.json (in project root)")
            print("  - Run the script again - it will open browser for authentication")
            print("  - Make sure to approve ALL permissions when prompted")
        else:
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
    failed = 0
    
    # Refresh token every 50 uploads to prevent expiration
    refresh_interval = 50
    
    for idx, img_file in enumerate(tqdm(image_files, desc="Uploading images")):
        try:
            # Refresh token periodically or if expired
            if idx > 0 and (idx % refresh_interval == 0 or gauth.access_token_expired):
                drive, gauth = refresh_token_if_needed(gauth, drive)
            
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
            error_str = str(e)
            # Check if it's a token expiration error
            if "invalid_grant" in error_str or "Token expired" in error_str or "access_token_expired" in error_str:
                print(f"\nToken expired during upload. Refreshing and retrying {img_file.name}...")
                try:
                    drive, gauth = refresh_token_if_needed(gauth, drive)
                    
                    # Retry the upload
                    query = f"title='{img_file.name}' and '{images_folder_id}' in parents and trashed=false"
                    existing = drive.ListFile({'q': query}).GetList()
                    
                    if existing:
                        print(f"Skipping {img_file.name} (already exists)")
                        skipped += 1
                    else:
                        file_drive = drive.CreateFile({
                            'title': img_file.name,
                            'parents': [{'id': images_folder_id}]
                        })
                        file_drive.SetContentFile(str(img_file))
                        file_drive.Upload()
                        uploaded += 1
                        print(f"Successfully uploaded {img_file.name} after token refresh")
                except Exception as retry_error:
                    print(f"\nError uploading {img_file.name} after retry: {retry_error}")
                    failed += 1
            else:
                print(f"\nError uploading {img_file.name}: {e}")
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total: {total_images}")
    print(f"{'='*60}")
    if failed > 0:
        print(f"\n⚠ Warning: {failed} files failed to upload. You may need to run the script again.")
    print(f"\nGoogle Drive folder ID: {images_folder_id}")
    print("Save this ID for downloading images later!")

if __name__ == "__main__":
    upload_images()

