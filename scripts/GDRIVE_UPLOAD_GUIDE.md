# Google Drive Upload Guide - Step by Step

## Prerequisites Check
- ✅ 8000 images found in `data/ps05_coco/images/`
- ❌ pydrive2 not installed
- ❌ credentials.json not found

## Step 1: Install Required Package

```bash
pip install pydrive2
```

## Step 2: Setup Google Cloud Project

### 2.1 Create/Select Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on project dropdown (top left)
3. Click "New Project" or select existing project
4. Name it: "lexo-graph-ai" (or any name)
5. Click "Create"

### 2.2 Enable Google Drive API
1. In the project, go to "APIs & Services" > "Library"
2. Search for "Google Drive API"
3. Click on "Google Drive API"
4. Click "Enable"
5. Wait for it to enable (may take a minute)

### 2.3 Configure OAuth Consent Screen
1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" user type
3. Click "Create"
4. Fill in:
   - App name: `lexo-graph-ai`
   - User support email: your email
   - Developer contact: your email
5. Click "Save and Continue"
6. On "Scopes" page, click "Save and Continue"
7. On "Test users" page:
   - Click "Add Users"
   - Add your email address
   - Click "Add"
8. Click "Save and Continue"
9. Review and click "Back to Dashboard"

### 2.4 Create OAuth Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "+ Create Credentials" > "OAuth client ID"
3. Application type: **Desktop app**
4. Name: `lexo-graph-ai-desktop`
5. Click "Create"
6. A popup will appear with credentials
7. Click "Download JSON"
8. Save the file as `credentials.json` in the project root:
   ```
   C:\Users\NIDHI\OneDrive\Desktop\lexo-graph-ai\credentials.json
   ```

## Step 3: Upload Images

```bash
python scripts/upload_images_to_gdrive.py
```

**First time:**
- A browser window will open
- Sign in with your Google account
- Click "Advanced" if you see a warning
- Click "Go to lexo-graph-ai (unsafe)" or "Continue"
- Click "Allow" to grant permissions
- You'll see "The authentication flow has completed"

**After authentication:**
- Script will create folder structure: `lexo-graph-ai/ps05_coco/images/`
- Uploads all 8000 images (this may take 1-2 hours)
- At the end, it will print: **"Google Drive folder ID: XXXXXXXX"**

## Step 4: Save the Folder ID

**IMPORTANT:** Copy and save the folder ID shown at the end!

Example:
```
Google Drive folder ID: 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
```

Save it in:
- A text file
- The README.md
- Or create a `.env` file with: `GDRIVE_FOLDER_ID=your_folder_id_here`

## Step 5: Verify Upload

1. Go to [Google Drive](https://drive.google.com)
2. Navigate to: `lexo-graph-ai` > `ps05_coco` > `images`
3. Verify images are there (should see ~8000 PNG files)

## Step 6: Download on Another Machine

When you need to download on another machine:

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd lexo-graph-ai
   ```

2. Install dependencies:
   ```bash
   pip install pydrive2
   ```

3. Copy `credentials.json` to the new machine (or create new credentials)

4. Download images:
   ```bash
   # Option 1: Set environment variable
   export GDRIVE_FOLDER_ID="your_folder_id_here"
   python scripts/download_images_from_gdrive.py
   
   # Option 2: Pass as argument
   python scripts/download_images_from_gdrive.py your_folder_id_here
   ```

## Troubleshooting

### "ModuleNotFoundError: No module named 'pydrive2'"
```bash
pip install pydrive2
```

### "credentials.json not found"
- Make sure you downloaded the JSON file from Google Cloud Console
- Place it in the project root directory

### "Access denied" or "OAuth error"
- Make sure you added yourself as a test user in OAuth consent screen
- Try deleting `gdrive_credentials.json` and re-authenticating

### Upload is slow
- Normal: Uploading 8000 images (~2-3 GB) takes time
- Script will continue even if interrupted (skips already uploaded files)
- Check Google Drive quota (free tier has 15 GB)

### Folder ID not saved
- Run the script again - it will show the folder ID
- Or check Google Drive URL: `https://drive.google.com/drive/folders/FOLDER_ID`

## Notes

- The script creates `gdrive_credentials.json` automatically after first auth
- Add `gdrive_credentials.json` to `.gitignore` (already there)
- **Never commit `credentials.json` to Git** - it contains sensitive info
- Images are stored permanently in Google Drive
- You can share the folder with team members if needed


