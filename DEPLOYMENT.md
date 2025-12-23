# Deployment Guide - YouTube Summarizer

## Quick Deploy Links

| Service | Purpose | Deploy |
|---------|---------|--------|
| Hugging Face | Backend (FastAPI / Docker) | [Create HF Space](https://huggingface.co/new-space) |
| Vercel | Frontend (React) | [Deploy to Vercel](https://vercel.com/new) |

---

## Step 1: Deploy Backend to Hugging Face Spaces

1. Go to [Hugging Face - New Space](https://huggingface.co/new-space)
2. **Settings**:
   - **Space Name**: `youtube-summarizer-api`
   - **SDK**: `Docker`
   - **Template**: `Blank`
   - **Visibility**: `Public` (or Private if preferred)
3. Click **"Create Space"**
4. Connect your GitHub repository:
   - Go to **Settings** tab in your new Space
   - Scroll to **"Connected GitHub Repository"**
   - Connect `Abdulrahmann-Omar/YouTube-Summarizer-1`
5. **Set Environment Variables**:
   - Go to **Settings** tab → **"Variables and secrets"**
   - Click **"New secret"**
   - Name: `GEMINI_API_KEY`
   - Value: `AIzaSyA8TEqjomna43KP2YDGa7dBxFENHyXks54` (or your personal key)
6. **Deploy**:
   - HF will automatically detect the root `Dockerfile` and start building.
7. **Get API URL**:
   - Once running, click **"Direct URL"** (found in the top right menu options `...` or under the App tab).
   - It will look like: `https://[username]-[space-name].hf.space`

---

## Step 2: Deploy Frontend to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/new)
2. Import: `Abdulrahmann-Omar/YouTube-Summarizer-1`
3. Configure:
   - **Framework**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. **Add Environment Variable**:
   - `VITE_API_URL`: `https://[username]-[space-name].hf.space` (from Step 1)
5. Click **"Deploy"**

---

## Environment Variables Summary

| Variable | Service | Value |
|----------|---------|-------|
| `GEMINI_API_KEY` | HF Space | Get from [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `VITE_API_URL` | Vercel | Your Hugging Face Space URL |

---

## Verify Deployment

1. **Backend Health**: `https://[username]-[space-name].hf.space/health`
2. **API Docs**: `https://[username]-[space-name].hf.space/docs`
3. **Frontend**: `https://your-app.vercel.app`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build Error | Ensure `Dockerfile` is in the root directory and correctly references `backend/` |
| API not responding | Check HF Space logs for errors |
| Gemini errors | Verify `GEMINI_API_KEY` is set in HF Secrets |
| Build failures | Check `backend/requirements.txt` for any incompatible versions |
