# Deployment Guide - YouTube Summarizer

## Quick Deploy Links

| Service | Purpose | Deploy |
|---------|---------|--------|
| Render | Backend (FastAPI) | [Deploy to Render](https://dashboard.render.com/blueprints) |
| Vercel | Frontend (React) | [Deploy to Vercel](https://vercel.com/new) |

---

## Step 1: Deploy Backend to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New" → "Blueprint"**
3. Connect your GitHub repo: `Abdulrahmann-Omar/YouTube-Summarizer-1`
4. Render will detect `render.yaml` automatically
5. Set environment variables:
   - `GEMINI_API_KEY`: Your Google Gemini API key
6. Click **"Apply"**
7. Wait for deployment (~3-5 minutes)
8. Copy your backend URL: `https://youtube-summarizer-api.onrender.com`

---

## Step 2: Deploy Frontend to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/new)
2. Import: `Abdulrahmann-Omar/YouTube-Summarizer-1`
3. Configure:
   - **Framework**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add environment variable:
   - `VITE_API_URL`: `https://youtube-summarizer-api.onrender.com`
5. Click **"Deploy"**

---

## Alternative: Single Vercel Deployment (Full Stack)

For simpler deployment, you can deploy both frontend and backend on Vercel:

1. Go to [Vercel Dashboard](https://vercel.com/new)
2. Import your repo
3. Set environment variables:
   - `GEMINI_API_KEY`: Your Gemini API key
4. Vercel will use the root `vercel.json` if present

---

## Environment Variables Summary

| Variable | Service | Value |
|----------|---------|-------|
| `GEMINI_API_KEY` | Backend | Get from [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `VITE_API_URL` | Frontend | Your Render backend URL |
| `CORS_ORIGINS` | Backend | Your Vercel frontend URL |

---

## Verify Deployment

1. **Backend Health**: `https://your-backend.onrender.com/health`
2. **API Docs**: `https://your-backend.onrender.com/docs`
3. **Frontend**: `https://your-app.vercel.app`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CORS errors | Add frontend URL to `CORS_ORIGINS` in Render |
| API not responding | Check Render logs for errors |
| Gemini errors | Verify `GEMINI_API_KEY` is set correctly |
| Build failures | Check `npm install` in frontend folder |
