# Google Drive Setup Instructions

## Step 1: Install Required Package

```bash
pip install pydrive2
```

## Step 2: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Drive API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

## Step 3: Create OAuth Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in app name, user support email
   - Add your email to test users
   - Save
4. Create OAuth client ID:
   - Application type: **Desktop app**
   - Name: "lexo-graph-ai"
   - Click "Create"
5. Download the credentials JSON file
6. Rename it to `credentials.json` and place it in the project root

## Step 4: Upload Images

```bash
python scripts/upload_images_to_gdrive.py
```

- First time will open a browser for authentication
- Grant permissions to access Google Drive
- Images will be uploaded to: `lexo-graph-ai/ps05_coco/images/`
- **Save the folder ID** shown at the end for downloading later

## Step 5: Download Images (on another machine)

```bash
# Set the folder ID from upload script output
export GDRIVE_FOLDER_ID="your_folder_id_here"
python scripts/download_images_from_gdrive.py

# OR pass as argument
python scripts/download_images_from_gdrive.py <FOLDER_ID>
```

## Notes

- The `gdrive_credentials.json` file is created automatically after first authentication
- You can add `gdrive_credentials.json` to `.gitignore` (it's already there)
- Keep `credentials.json` secure - it contains your OAuth client credentials

