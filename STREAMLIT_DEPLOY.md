# 🚀 Deploy to Streamlit Cloud - Step by Step Guide

## ✅ Prerequisites

- ✅ Your GitHub repository: `29SalahMo/web-automated-summarizer-for-articles`
- ✅ Streamlit account connected to GitHub
- ✅ All code pushed to GitHub

## 📋 Step-by-Step Deployment

### Step 1: Go to Streamlit Cloud

1. Visit: **https://share.streamlit.io/**
2. Sign in with your GitHub account (if not already signed in)

### Step 2: Create New App

1. Click **"New app"** button (top right)
2. You'll see a form to configure your app

### Step 3: Configure Your App

Fill in the form:

- **Repository**: Select `29SalahMo/web-automated-summarizer-for-articles`
- **Branch**: Select `main`
- **Main file path**: Enter `app.py`
- **App URL** (optional): Choose a custom URL like `web-summarizer` (optional)

### Step 4: Advanced Settings (Optional but Recommended)

Click **"Advanced settings"** and configure:

- **Python version**: `3.11` (or latest available)
- **Memory**: Select higher memory if available (models need RAM)
- **Auto-reload**: Enable (so changes auto-deploy)

### Step 5: Deploy!

1. Click **"Deploy"** button
2. Wait for deployment (first time takes 5-10 minutes)
3. Watch the logs for progress

### Step 6: Access Your App

Once deployed, you'll get a URL like:
```
https://web-summarizer.streamlit.app
```

## 🔧 Troubleshooting

### If deployment fails:

1. **Check logs** in Streamlit Cloud dashboard
2. **Common issues**:
   - Missing dependencies → Check `requirements.txt`
   - Memory issues → Models are large, may need more memory
   - Import errors → Check that all packages are in `requirements.txt`

### If app shows "Oh no" error:

1. Check the logs in Streamlit Cloud
2. Look for specific error messages
3. Common fixes:
   - Ensure `app.py` is the main file
   - Check that all imports are available
   - Verify Python version compatibility

## 📝 Important Notes

- **First deployment**: Takes 5-10 minutes (downloading models)
- **Subsequent deployments**: Faster (2-3 minutes)
- **Model loading**: Happens on first use, not at startup
- **Memory**: Free tier has limits, may need to upgrade for large models

## 🎯 Your App Structure

Make sure these files are in your GitHub repo:

```
web-automated-summarizer-for-articles/
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← Dependencies
├── README.md                 ← Documentation
└── .streamlit/
    └── config.toml           ← Streamlit config (optional)
```

## ✅ Verification Checklist

Before deploying, ensure:

- [ ] `app.py` exists and is the main file
- [ ] `requirements.txt` has all dependencies
- [ ] Code is pushed to GitHub `main` branch
- [ ] No syntax errors in `app.py`
- [ ] All imports are available in `requirements.txt`

## 🚀 After Deployment

Once your app is live:

1. Test with a short English text first
2. Test file upload (PDF/DOCX/TXT)
3. Test Arabic summarization
4. Share your app URL!

---

**Need help?** Check Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
