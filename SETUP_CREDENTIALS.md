# Credentials Setup Guide

This guide explains how to set up credentials for the application without committing them to git.

## Overview

The application uses two types of credentials:
1. **Google Cloud Vision API** - For OCR (Stage 2)
2. **Google Drive API** - For data upload/download (optional)

Both are stored in the `config/` directory but are **excluded from git** for security.

---

## Google Cloud Vision API Credentials

### Step 1: Get Your Credentials File

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select your project
3. Enable Cloud Vision API
4. Create service account
5. Download JSON key file

### Step 2: Save Credentials Locally

**Option A: Save with specific name (Recommended)**
```bash
# Copy your downloaded JSON file to:
config/google_cloud_credentials.json
```

**Option B: Use any filename**
```bash
# Copy to config/ folder with any name
config/your-credentials-file.json
```

Then update `config/ocr_config.yaml`:
```yaml
google_cloud:
  credentials_path: "config/your-credentials-file.json"
```

### Step 3: Verify Setup

```bash
python scripts/test_google_vision_setup.py --credentials config/google_cloud_credentials.json
```

### Step 4: Update Config File

Edit `config/ocr_config.yaml` if using a different filename:
```yaml
google_cloud:
  credentials_path: "config/google_cloud_credentials.json"  # Update this path
```

---

## Google Drive API Credentials (Optional)

### Step 1: Get Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Drive API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download JSON file

### Step 2: Save Credentials

```bash
# Copy downloaded file to:
config/credentials.json
```

OR use the existing `client_secrets.json` if you have it.

### Step 3: Configure

The scripts will automatically use `credentials.json` or `client_secrets.json` from the project root.

---

## Accessing Credentials After Git Clone

When you clone the repository on a new machine:

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd lexo-graph-ai
```

### Step 2: Set Up Credentials

**For Google Cloud Vision:**
```bash
# Copy your credentials JSON file to:
config/google_cloud_credentials.json
```

**For Google Drive (if needed):**
```bash
# Copy your credentials JSON file to:
config/credentials.json
```

### Step 3: Verify

```bash
# Test Google Cloud Vision setup
python scripts/test_google_vision_setup.py --credentials config/google_cloud_credentials.json
```

---

## Security Best Practices

### ✅ DO:
- Keep credentials in `config/` directory
- Add credentials to `.gitignore` (already done)
- Use environment variables if needed
- Rotate credentials periodically
- Use service accounts (not user accounts)

### ❌ DON'T:
- Commit credentials to git
- Share credentials publicly
- Use credentials in public repositories
- Store credentials in code

---

## Environment Variables (Alternative)

You can also use environment variables instead of file paths:

### Google Cloud Vision

**Set environment variable:**
```bash
# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\credentials.json"

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Then in config:**
```yaml
google_cloud:
  credentials_path: ""  # Leave empty to use environment variable
```

---

## Troubleshooting

### Error: "Credentials file not found"

**Solution:**
1. Check file exists: `ls config/google_cloud_credentials.json`
2. Verify path in `config/ocr_config.yaml`
3. Check file permissions

### Error: "Invalid credentials"

**Solution:**
1. Verify credentials file is valid JSON
2. Check service account has proper permissions
3. Ensure Cloud Vision API is enabled
4. Re-download credentials if needed

### File is in .gitignore but I need it

**Solution:**
- Credentials are intentionally excluded from git
- Each user/clone needs their own credentials
- Use the template file as reference
- Follow setup instructions above

---

## Template Files

The repository includes template files:

- `config/google_cloud_credentials.json.template` - Template for Google Cloud credentials
- `config/credentials.json.template` - Template for Google Drive credentials
- `config/ocr_config.yaml.template` - Template for OCR config

**Note:** Template files are safe to commit. They don't contain real credentials.

---

## Quick Setup Checklist

After cloning the repository:

- [ ] Copy Google Cloud Vision credentials to `config/google_cloud_credentials.json`
- [ ] Update `config/ocr_config.yaml` if using different filename
- [ ] Test setup: `python scripts/test_google_vision_setup.py --credentials config/google_cloud_credentials.json`
- [ ] (Optional) Copy Google Drive credentials to `config/credentials.json`
- [ ] Verify all credentials are working

---

## For Team Members

If you're working in a team:

1. **Each member** needs their own credentials
2. **Don't share** credentials via git, email, or chat
3. **Use** environment variables for CI/CD pipelines
4. **Rotate** credentials if compromised
5. **Follow** security best practices

---

## Summary

- Credentials are stored locally in `config/` directory
- They are excluded from git (`.gitignore`)
- Each user needs to set up their own credentials
- Template files show the expected format
- Environment variables are an alternative option

**The credentials file path is already configured in `config/ocr_config.yaml` - just place your credentials file at that path!**

