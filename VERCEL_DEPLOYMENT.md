# Vercel Full-Stack Deployment Guide

## Quick Steps to Deploy

### 1. Commit and Push Latest Changes
```powershell
git add .
git commit -m "Configure for Vercel full-stack deployment"
git push origin main
```

### 2. Deploy to Vercel

1. Go to **[vercel.com/new](https://vercel.com/new)**
2. Click **"Import Git Repository"**
3. Select: `Abdulrahmann-Omar/YouTube-Summarizer-1`
4. Keep **Root Directory** as `./` (blank)
5. **Framework Preset**: Select `Vite`
6. Click **"Environment Variables"** and add:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: `AIzaSyA8TEqjomna43KP2YDGa7dBxFENHyXks54`
7. Click **"Deploy"**

### 3. Wait for Build (~3-5 minutes)

Vercel will:
- Build the frontend (React/Vite)
- Set up the backend (FastAPI as serverless functions)

### 4. Access Your App

Once deployed, you'll get a URL like:
- `https://youtube-summarizer-1.vercel.app`

---

## Important Notes

### Backend Limitations on Vercel
- **Serverless**: Each API request runs in an isolated function
- **Cold starts**: First request may be slow (~5-10s) if the function is "cold"
- **Fine-tuned models**: Will NOT work on Vercel free tier due to memory/size limits
  - DistilBART and T5-small require persistent storage and GPU
  - Only Gemini AI, TF-IDF, Frequency, and Gensim will work

### Recommended Split Deployment
For best results with fine-tuned models:
1. **Frontend → Vercel** (fast, free)
2. **Backend → Hugging Face Spaces** (supports Docker, larger models)

See detailed guide in `DEPLOYMENT.md`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check that `frontend/package.json` exists and has `build` script |
| API 500 errors | Check Vercel function logs for Python errors |
| Gemini not working | Verify `GEMINI_API_KEY` in Vercel dashboard → Settings → Environment Variables |
| Frontend 404 | Ensure `vercel.json` routes are correct |

---

## Verify Deployment

1. **Frontend**: Visit your Vercel URL
2. **Backend Health**: `https://your-app.vercel.app/health`
3. **API Docs**: `https://your-app.vercel.app/docs`
