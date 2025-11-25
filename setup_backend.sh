#!/bin/bash
# Backend setup script for YouTube Summarizer

echo "🚀 Setting up YouTube Summarizer Backend..."

# Create virtual environment
echo "📦 Creating virtual environment..."
cd backend
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "✨ Backend setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your GEMINI_API_KEY"
echo "2. Run: source backend/venv/bin/activate"
echo "3. Run: cd backend && python -m uvicorn main:app --reload"
echo ""

