<div align="center">

# 🎬 YouTube Summarizer

### AI-Powered Video Summarization with Interactive Q&A

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4.svg?logo=google)](https://ai.google.dev/)

Transform lengthy YouTube videos into concise, intelligent summaries using advanced NLP and Google Gemini AI. Ask follow-up questions through an interactive chat interface.

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## ✨ Demo

<div align="center">
<img src="gui.gif" alt="YouTube Summarizer Demo" width="700"/>
</div>

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered Summarization
- **Google Gemini AI** - State-of-the-art abstractive summaries
- **TF-IDF** - Keyword-focused extraction with n-grams
- **Frequency-Based** - Position-aware sentence scoring
- **TextRank** - Graph-based extractive summarization

</td>
<td width="50%">

### 💬 Interactive Q&A Chat
- Real-time WebSocket communication
- Context-aware answers about video content
- Conversation history tracking
- Confidence scoring for responses

</td>
</tr>
<tr>
<td width="50%">

### 🔬 Advanced NLP Processing
- spaCy-powered tokenization & preprocessing
- Named Entity Recognition (NER)
- Key topic extraction
- Smart sentence segmentation

</td>
<td width="50%">

### 🎨 Modern User Interface
- Beautiful responsive design
- Dark/Light mode toggle
- Real-time video previews
- Export as TXT or JSON

</td>
</tr>
</table>

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- [Google Gemini API Key](https://makersuite.google.com/app/apikey) (free tier available)

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/el3ttar3/YouTube-Summarizer.git
cd youtube-summarizer

# Create environment file
cp env.template .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

```bash
# Terminal 1 - Backend
cd backend && source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

🎉 **Open [http://localhost:3000](http://localhost:3000) and start summarizing!**

---

## 📁 Project Structure

```
youtube-summarizer/
├── 🔧 backend/
│   ├── main.py                    # FastAPI application & endpoints
│   ├── config.py                  # Environment configuration
│   ├── services/
│   │   ├── youtube_service.py     # Caption extraction (yt-dlp)
│   │   ├── nlp_service.py         # NLP preprocessing (spaCy)
│   │   ├── summarization_service.py  # TF-IDF, Frequency, TextRank
│   │   └── gemini_service.py      # Google Gemini AI integration
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── utils/
│       └── text_preprocessing.py  # Text cleaning utilities
│
├── 🎨 frontend/
│   ├── src/
│   │   ├── App.jsx                # Main application component
│   │   ├── components/
│   │   │   ├── Header.jsx         # Navigation & theme toggle
│   │   │   ├── VideoInput.jsx     # URL input & validation
│   │   │   ├── SummarizationOptions.jsx  # Method selection
│   │   │   ├── SummaryDisplay.jsx # Results display
│   │   │   └── QAChat.jsx         # Interactive Q&A chat
│   │   ├── services/
│   │   │   └── api.js             # API client
│   │   └── styles/                # CSS modules
│   ├── vite.config.js
│   └── package.json
│
├── docker-compose.yml             # Docker orchestration
├── env.template                   # Environment template
└── README.md
```

---

## 🎯 Usage

### Basic Workflow

1. **Paste YouTube URL** → Enter any YouTube video link
2. **Choose Method** → Select summarization algorithm (Gemini AI recommended)
3. **Adjust Length** → Use slider to set summary length (10%-80%)
4. **Generate** → Click "Generate Summary" and wait for processing
5. **Ask Questions** → Open chat to ask follow-up questions about the video
6. **Export** → Download your summary as TXT or JSON

### Summarization Methods Comparison

| Method | Speed | Quality | Type | Best For |
|--------|:-----:|:-------:|:----:|----------|
| **Gemini AI** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Abstractive | High-quality, context-aware summaries |
| **TF-IDF** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Extractive | Quick, keyword-focused summaries |
| **Frequency** | ⚡⚡⚡⚡ | ⭐⭐⭐ | Extractive | Balanced importance scoring |
| **TextRank** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Extractive | Graph-based sentence ranking |

---

## 🔌 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/summarize` | POST | Generate video summary |
| `/api/video-info` | GET | Get video metadata |
| `/api/qa` | POST | Ask questions about video |

### Summarize Video

```bash
curl -X POST http://localhost:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "method": "gemini",
    "fraction": 0.3
  }'
```

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/unique-client-id');
ws.send(JSON.stringify({
  question: "What is the main topic?",
  video_context: "Video transcript..."
}));
```

📚 **Full API docs available at** [http://localhost:8000/docs](http://localhost:8000/docs) when running

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
open http://localhost:3000
```

---

## ⚙️ Configuration

Create a `.env` file from the template:

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Optional - spaCy model (sm=fast, trf=accurate)
SPACY_MODEL=en_core_web_sm

# Optional - Summarization settings
DEFAULT_FRACTION=0.3
MAX_SUMMARY_LENGTH=1000

# Optional - Server settings
BACKEND_PORT=8000
DEBUG_MODE=True
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="33%">

**Backend**

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![spaCy](https://img.shields.io/badge/-spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white)

</td>
<td align="center" width="33%">

**Frontend**

![React](https://img.shields.io/badge/-React-61DAFB?style=flat-square&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/-Vite-646CFF?style=flat-square&logo=vite&logoColor=white)
![CSS3](https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)

</td>
<td align="center" width="33%">

**AI & NLP**

![Google Gemini](https://img.shields.io/badge/-Gemini-4285F4?style=flat-square&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-sklearn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Gensim](https://img.shields.io/badge/-Gensim-2C8EBB?style=flat-square)

</td>
</tr>
</table>

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ "Gemini API not available"</b></summary>

- Verify `GEMINI_API_KEY` is set in your `.env` file
- Check the key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
</details>

<details>
<summary><b>❌ "spaCy model not found"</b></summary>

```bash
python -m spacy download en_core_web_sm
```
</details>

<details>
<summary><b>❌ "Video captions not available"</b></summary>

- The video may not have captions/subtitles enabled
- Try a different video with English captions
</details>

<details>
<summary><b>❌ WebSocket connection fails</b></summary>

- Ensure backend is running on port 8000
- Check CORS settings match your frontend URL
</details>

---

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Playlist batch processing
- [ ] Summary quality scoring
- [ ] User authentication & history
- [ ] Browser extension
- [ ] Mobile application
- [ ] Integration with Coursera, Udemy, etc.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [spaCy](https://spacy.io/) - Industrial-strength NLP
- [Google Gemini](https://ai.google.dev/) - Advanced AI capabilities
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube caption extraction
- [Gensim](https://radimrehurek.com/gensim/) - Topic modeling

---

<div align="center">

**[⬆ Back to Top](#-youtube-summarizer)**

Made for NLP-project by developers who watch too many hidi-YouTube videos

</div>
