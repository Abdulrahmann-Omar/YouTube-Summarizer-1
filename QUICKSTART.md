# Quick Start Guide

Get YouTube Summarizer up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 16+
- Google Gemini API key (free at [Google AI Studio](https://makersuite.google.com/app/apikey))

## Installation

### Option 1: Automated Setup (Recommended for Unix/Mac)

```bash
# 1. Setup backend
chmod +x setup_backend.sh
./setup_backend.sh

# 2. Setup frontend
chmod +x setup_frontend.sh
./setup_frontend.sh

# 3. Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_api_key_here
SPACY_MODEL=en_core_web_sm
DEFAULT_FRACTION=0.3
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DEBUG_MODE=True
LOG_LEVEL=INFO
EOF

# 4. Run development servers
chmod +x run_dev.sh
./run_dev.sh
```

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Frontend:**
```bash
cd frontend
npm install
```

**Environment:**
Create `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SPACY_MODEL=en_core_web_sm
```

## Running

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

**Access:** Open http://localhost:3000

## First Use

1. **Get Gemini API Key**:
   - Visit https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy and add to `.env` file

2. **Test with a video**:
   - Paste a YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)
   - Select "Gemini AI" method
   - Click "Generate Summary"

3. **Try Q&A**:
   - After summary is generated
   - Click "Ask Questions"
   - Ask anything about the video content!

## Troubleshooting

**Backend won't start:**
- Check Python version: `python --version` (need 3.8+)
- Ensure virtual environment is activated
- Verify all packages installed: `pip list`

**Frontend won't start:**
- Check Node version: `node --version` (need 16+)
- Clear cache: `rm -rf node_modules && npm install`

**No summaries generating:**
- Verify `GEMINI_API_KEY` in `.env`
- Check backend logs for errors
- Try a different video with English captions

**WebSocket errors:**
- Ensure backend is running on port 8000
- Check CORS settings in `.env`
- Refresh the browser page

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Explore different summarization methods
- Try the Q&A chat feature
- Adjust summary length with the slider
- Export summaries to TXT/JSON

## Support

- Check [README.md](README.md) for detailed docs
- Open an issue on GitHub for bugs
- Read API docs at http://localhost:8000/docs

---

Happy summarizing! 🎉

