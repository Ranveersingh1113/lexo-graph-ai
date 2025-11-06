# Git Commit Guide

This guide explains how to commit your code while keeping credentials secure.

## ✅ What Will Be Committed

### Code Files
- All Python scripts in `scripts/`
- All source code in `src/`
- Configuration templates and YAML files (safe)

### Documentation
- All `.md` files (README, phase docs, setup guides)
- Project structure documentation

### Configuration
- `config/*.yaml` files (safe, no credentials)
- `config/*.template` files (templates only)
- `settings.yaml` (PyDrive config, safe)

### Ignored (NOT Committed)
- `config/*.json` (except templates) - Contains credentials
- `data/ps05_coco/images/` - Large image files
- `training_logs/` - Training logs
- Model outputs (`*.pdparams`, `output/`)
- Credentials files

## 🔐 Credentials Safety

### Verified Safe
- ✅ `config/ocr_config.yaml` - Only contains paths, no secrets
- ✅ `config/stage3_config.yaml` - Model selection, no secrets
- ✅ `config/*.template` - Template files only
- ✅ `settings.yaml` - PyDrive config, no secrets

### Excluded from Git
- ❌ `config/psyched-circuit-477317-j9-ded75231e471.json` - Your actual credentials
- ❌ `config/google_cloud_credentials.json` - Credentials file
- ❌ `credentials.json` - Google Drive credentials
- ❌ `client_secrets.json` - OAuth secrets

## 📝 Staging Files

### Step 1: Review Changes

```bash
git status
```

### Step 2: Stage All Safe Files

```bash
# Stage all new files and changes
git add .

# Verify what will be committed
git status
```

### Step 3: Verify No Credentials

```bash
# Check that credentials are NOT staged
git status --porcelain config/ | grep -v template | grep -v yaml
# Should show nothing (or only untracked, which is fine)

# Double-check specific credential file
git check-ignore config/psyched-circuit-477317-j9-ded75231e471.json
# Should show that it's ignored
```

### Step 4: Commit

```bash
git commit -m "Add complete pipeline implementation: Phase 1-3

- Phase 1: Layout detection inference pipeline
- Phase 2: Multilingual OCR with Google Cloud Vision
- Phase 3: Table & Figure processing with evaluation framework
- Add comprehensive documentation for all phases
- Add credential templates and setup guides
- Organize project structure"
```

### Step 5: Push

```bash
git push origin main
```

## 🚨 Before Pushing Checklist

- [ ] Verify credentials are NOT in staged files
- [ ] Check `.gitignore` includes all credential patterns
- [ ] Ensure `config/*.json` (except templates) are ignored
- [ ] Review `git status` output
- [ ] Test that credentials are properly ignored

## 📋 What Gets Committed

### New Files Added
- `src/` - All source code modules
- `scripts/` - All pipeline scripts
- `config/*.yaml` - Configuration files (safe)
- `config/*.template` - Template files
- All documentation `.md` files
- `.gitignore` - Updated ignore rules
- `.gitattributes` - File attributes

### Modified Files
- `README.md` - Updated with complete pipeline info
- `scripts/download_images_from_gdrive.py` - OAuth improvements
- `scripts/upload_images_to_gdrive.py` - OAuth improvements

## 🔍 Verification Commands

```bash
# Check what will be committed
git diff --cached --name-only

# Check for potential credential files
git diff --cached --name-only | grep -i credential
git diff --cached --name-only | grep -i secret
git diff --cached --name-only | grep -i key

# Verify .gitignore is working
git check-ignore config/psyched-circuit-477317-j9-ded75231e471.json
git check-ignore config/google_cloud_credentials.json
```

## 🆘 If Credentials Were Accidentally Staged

If you accidentally staged credentials:

```bash
# Unstage the file
git reset HEAD config/psyched-circuit-477317-j9-ded75231e471.json

# Add to .gitignore if not already there
echo "config/psyched-circuit-477317-j9-ded75231e471.json" >> .gitignore

# Verify it's ignored
git check-ignore config/psyched-circuit-477317-j9-ded75231e471.json
```

## 📚 After Push

### For Team Members / New Clones

After someone clones the repository:

1. **They need to set up credentials:**
   ```bash
   # Copy their credentials file to:
   config/google_cloud_credentials.json
   ```

2. **Update config:**
   - Edit `config/ocr_config.yaml` if needed
   - Path should point to their credentials file

3. **See `SETUP_CREDENTIALS.md` for full instructions**

## ✅ Summary

- ✅ All code is safe to commit
- ✅ All documentation is safe to commit
- ✅ Configuration files are safe (no secrets)
- ✅ Credentials are properly excluded
- ✅ Templates are provided for reference
- ✅ Setup guides explain credential configuration

**You're ready to commit!**

