# Backend Dockerfile for YouTube Summarizer
# Optimization: Use python slim image for smaller size
FROM python:3.11-slim

WORKDIR /app

# Optimization: Install system dependencies in a single layer and clean up
# Added git because some python packages might need it (though unlikely for wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimization: Copy and install base requirements first (better caching)
COPY requirements-base.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# Optimization: Install volatile requirements separately
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optimization: Download models in a separate layer
# Using en_core_web_sm (small model) as identified in config
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy application code last
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

