# Google Cloud Vision API Setup Guide

This guide will help you set up Google Cloud Vision API for Phase 2 OCR integration.

## Prerequisites

- Google account
- Internet connection
- Basic familiarity with Google Cloud Console

## Step-by-Step Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top (next to "Google Cloud")
3. Click **"New Project"**
4. Enter project name: `lexo-graph-ai-ocr` (or any name you prefer)
5. Click **"Create"**
6. Wait for project creation (usually takes a few seconds)
7. Select your newly created project from the dropdown

### Step 2: Enable Cloud Vision API

1. In the Google Cloud Console, go to **"APIs & Services"** → **"Library"**
2. Search for **"Cloud Vision API"**
3. Click on **"Cloud Vision API"** from the results
4. Click **"Enable"** button
5. Wait for the API to be enabled (usually takes 1-2 minutes)

### Step 3: Create Service Account

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"Create Credentials"** → **"Service Account"**
3. Fill in the details:
   - **Service account name**: `ocr-service-account`
   - **Service account ID**: (auto-generated, you can leave it)
   - **Description**: `Service account for PS-05 OCR integration`
4. Click **"Create and Continue"**
5. Skip the optional steps (Grant access, Grant users access) and click **"Done"**

### Step 4: Create and Download JSON Key

1. In the **"Credentials"** page, find your service account (it should appear in the list)
2. Click on the service account email (e.g., `ocr-service-account@your-project.iam.gserviceaccount.com`)
3. Go to the **"Keys"** tab
4. Click **"Add Key"** → **"Create new key"**
5. Select **"JSON"** as the key type
6. Click **"Create"**
7. The JSON key file will be downloaded automatically
8. **IMPORTANT**: Save this file securely! It contains credentials that allow access to your Google Cloud resources.

### Step 5: Save Credentials File

1. Move the downloaded JSON file to your project:
   ```
   Move to: lexo-graph-ai/config/google_cloud_credentials.json
   ```
   (Or any secure location in your project)

2. **IMPORTANT SECURITY**: 
   - Add this file to `.gitignore` to prevent committing credentials to git
   - Never share this file publicly
   - Keep it secure on your local machine

### Step 6: Install Required Package

Run the following command in your terminal:

```bash
pip install google-cloud-vision
```

Or if you're in a conda environment:

```bash
conda activate doc-comp
pip install google-cloud-vision
```

### Step 7: Verify Setup

You can verify your setup by running:

```bash
python -c "from google.cloud import vision; print('Google Cloud Vision API is installed correctly!')"
```

If you see no errors, the package is installed correctly.

## Configuration

### Option 1: Environment Variable (Recommended)

Set the path to your credentials file as an environment variable:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\NIDHI\OneDrive\Desktop\lexo-graph-ai\config\google_cloud_credentials.json"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\NIDHI\OneDrive\Desktop\lexo-graph-credentials.json
```

**Linux/Mac:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google_cloud_credentials.json"
```

To make it permanent, add to your shell profile (`.bashrc`, `.zshrc`, etc.)

### Option 2: Specify in Code/Config

We'll create a config file that points to your credentials path.

## Quick Test

After setup, you can test the API with this simple script:

```python
from google.cloud import vision

# Initialize client
client = vision.ImageAnnotatorClient()

# Test with a simple image (you can use any image path)
image_path = "test_image.png"  # Replace with actual image path

with open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.text_detection(image=image)

if response.text_annotations:
    print("OCR Test Successful!")
    print(f"Detected text: {response.text_annotations[0].description}")
else:
    print("No text detected in image")
```

## Cost Information

### Free Tier
- **First 1,000 units per month**: FREE
- 1 image = 1 unit (regardless of text density)

### Paid Tier
- **1,001 - 5,000,000 units**: $1.50 per 1,000 units
- **5,000,001+ units**: $0.60 per 1,000 units

### Monitoring Usage
1. Go to **"APIs & Services"** → **"Dashboard"**
2. Select **"Cloud Vision API"**
3. View usage metrics and quotas

**Recommendation**: The free tier is sufficient for development and testing. Monitor usage if processing many images.

## Troubleshooting

### Error: "Could not automatically determine credentials"
**Solution**: Make sure `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set correctly, or specify credentials path in code.

### Error: "Permission denied" or "API not enabled"
**Solution**: 
1. Verify Cloud Vision API is enabled in your project
2. Check that your service account has proper permissions
3. Ensure you're using the correct project

### Error: "ModuleNotFoundError: No module named 'google.cloud'"
**Solution**: Install the package: `pip install google-cloud-vision`

### Error: "Invalid credentials"
**Solution**: 
1. Verify the JSON key file is valid (not corrupted)
2. Check that the service account has proper permissions
3. Re-download the key if needed

## Security Best Practices

1. ✅ **Never commit credentials to git** - Add to `.gitignore`
2. ✅ **Use service accounts** instead of user accounts for applications
3. ✅ **Rotate keys periodically** (every 90 days recommended)
4. ✅ **Limit permissions** - Only grant necessary permissions
5. ✅ **Monitor usage** - Set up billing alerts in Google Cloud Console

## Next Steps

Once setup is complete:
1. Verify the setup with the test script above
2. Confirm credentials are working
3. Proceed with Phase 2 implementation

## Support

- [Google Cloud Vision API Documentation](https://cloud.google.com/vision/docs)
- [Python Client Library](https://googleapis.dev/python/vision/latest/)
- [Pricing Information](https://cloud.google.com/vision/pricing)

