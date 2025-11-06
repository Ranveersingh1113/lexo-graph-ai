# Fix OAuth 403 Error: access_denied

## Problem
You're seeing: "lexo-graph-ai has not completed the Google verification process"

## Solution: Add Yourself as a Test User

### Step 1: Go to OAuth Consent Screen
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (lexo-graph-ai)
3. Navigate to: **APIs & Services** > **OAuth consent screen**

### Step 2: Add Test Users
1. Scroll down to the **"Test users"** section
2. Click **"+ ADD USERS"** button
3. Enter your **Google account email** (the one you're signing in with)
4. Click **"Add"**
5. The email should appear in the test users list

### Step 3: Save and Retry
1. Make sure you click **"Save"** if there's a save button
2. Wait 1-2 minutes for changes to propagate
3. Run the upload script again:
   ```bash
   python scripts/upload_images_to_gdrive.py
   ```

## Alternative: If Test Users Section is Missing

If you don't see the "Test users" section:

1. Check the **"Publishing status"** at the top
2. It should say **"Testing"** (not "In production")
3. If it says something else, click **"BACK TO DASHBOARD"** and then:
   - Make sure you completed all steps in the OAuth consent screen setup
   - Go through the setup wizard again if needed

## Quick Checklist

- [ ] OAuth consent screen is configured
- [ ] App is in "Testing" mode
- [ ] Your email is added to "Test users" list
- [ ] Waited 1-2 minutes after adding user
- [ ] Using the same Google account email that was added

## Still Having Issues?

If you still get the error after adding yourself:

1. **Double-check the email**: Make sure you're signing in with the EXACT email you added
2. **Check user type**: Make sure you selected "External" (not "Internal") during setup
3. **Try different account**: Add a different Google account and try signing in with that
4. **Wait longer**: Sometimes it takes 5-10 minutes for changes to take effect

