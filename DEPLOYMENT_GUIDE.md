# Heroku Deployment Guide

## Quick Deploy via Web Interface (Recommended - No CLI needed)

### Step 1: Create Heroku Account
1. Go to https://www.heroku.com
2. Sign up for a free account (if you don't have one)

### Step 2: Create New App
1. Log in to Heroku Dashboard: https://dashboard.heroku.com
2. Click "New" → "Create new app"
3. Choose an app name (e.g., `web-summarizer-app` or `your-name-summarizer`)
4. Select region (United States or Europe)
5. Click "Create app"

### Step 3: Connect GitHub Repository
1. In your Heroku app dashboard, go to the "Deploy" tab
2. Under "Deployment method", select "GitHub"
3. Click "Connect to GitHub" and authorize Heroku
4. Search for your repository: `web-automated-summarizer-for-articles`
5. Click "Connect" next to your repository

### Step 4: Enable Automatic Deploys
1. In the "Deploy" tab, scroll to "Automatic deploys"
2. Select the branch: `main`
3. Click "Enable Automatic Deploys"
4. (Optional) Check "Wait for CI to pass" if you have CI set up

### Step 5: Manual Deploy (First Time)
1. Scroll to "Manual deploy" section
2. Select branch: `main`
3. Click "Deploy Branch"
4. Wait for deployment to complete (this may take 5-10 minutes)

### Step 6: View Your App
1. Once deployment completes, click "View" or "Open app"
2. Your app will be available at: `https://your-app-name.herokuapp.com`

## Important Notes

⚠️ **First Deployment Notes:**
- The first deployment will take longer (10-15 minutes) because Heroku needs to:
  - Download and install all Python dependencies
  - Download AI models (several GB) from Hugging Face
  - Build the application

⚠️ **Memory & Performance:**
- Free tier Heroku dynos have 512MB RAM
- Large AI models may require upgrading to a paid dyno
- Consider using Heroku's "Eco" dyno ($5/month) for better performance

⚠️ **Model Loading:**
- Models are downloaded on first request (cold start)
- First request after deployment may take 2-3 minutes
- Subsequent requests will be faster

## Alternative: Deploy via Heroku CLI

If you prefer using command line:

1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

## Troubleshooting

### If deployment fails:
1. Check build logs in Heroku dashboard
2. Ensure all files are committed to GitHub
3. Verify `Procfile`, `requirements.txt`, and `runtime.txt` are correct

### If app crashes:
1. Check logs: `heroku logs --tail` (or view in dashboard)
2. Verify models are loading correctly
3. Check memory usage (may need larger dyno)

### If models don't load:
1. First request always takes longer
2. Check internet connectivity on Heroku dyno
3. Verify Hugging Face model names are correct

## Your App URL

Once deployed, your app will be available at:
**https://your-app-name.herokuapp.com**

Replace `your-app-name` with the name you chose when creating the Heroku app.
