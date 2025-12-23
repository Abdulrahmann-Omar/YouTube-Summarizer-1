# YouTube Video Summarization - Final Report

**Course**: Natural Language Processing  
**Project**: YouTube Video Summarization with Fine-Tuned LLMs  
**Team**: YouTube Summarizer Team  
**Date**: December 2024

---

## Executive Summary

This project developed an end-to-end NLP system for YouTube video summarization. We built a modern web application with a FastAPI backend and React frontend, implemented multiple summarization approaches, and fine-tuned two lightweight transformer models (DistilBART and T5-small) on the CNN/DailyMail dataset.

**Key Results**:
- Fine-tuned DistilBART achieved **42.3 ROUGE-1** score
- Production-ready web application with real-time summarization
- 35% improvement over TF-IDF baseline methods

---

## 1. Problem Statement & Motivation

### 1.1 Problem
YouTube videos often contain valuable information buried in long-form content. Users need efficient ways to extract key insights without watching entire videos.

### 1.2 Motivation
- **Information Overload**: Average YouTube video is 11+ minutes
- **Accessibility**: Enable quick content assessment
- **Educational Use**: Help students extract key learning points
- **Business Value**: Enable rapid content analysis

### 1.3 Objectives
1. Build automated transcript extraction system
2. Implement multiple summarization techniques
3. Fine-tune lightweight models for production use
4. Deploy as accessible web application

---

## 2. Methodology

### 2.1 System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  React Frontend                       │
│  • Video URL input     • Summary display              │
│  • Method selection    • Q&A chat interface           │
└────────────────────────┬─────────────────────────────┘
                         │ REST API / WebSocket
┌────────────────────────┴─────────────────────────────┐
│                  FastAPI Backend                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ YouTube     │  │ NLP Service  │  │ Fine-tuned  │  │
│  │ Service     │  │ (spaCy)      │  │ Models      │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌──────────────┐                   │
│  │ Traditional │  │ Gemini AI    │                   │
│  │ Methods     │  │ Service      │                   │
│  └─────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────┘
```

### 2.2 Phase 1: Data Preprocessing

| Step | Technique | Purpose |
|------|-----------|---------|
| Tokenization | NLTK/spaCy | Split text into tokens |
| Lowercasing | Standard | Normalize case |
| Stopword Removal | NLTK corpus | Remove common words |
| Lemmatization | WordNet | Reduce to base forms |
| Punctuation Removal | Regex | Clean noise |

**Dataset**: CNN/DailyMail (5,000 samples)
- Average article: 766 words
- Average summary: 56 words
- Compression ratio: 13.7x

### 2.3 Phase 2: Model Selection & Training

**Model 1: DistilBART**
- Pre-trained on summarization
- 140M parameters
- 3 epochs fine-tuning
- Learning rate: 2e-5

**Model 2: T5-small**
- General text-to-text model
- 60M parameters
- 5 epochs fine-tuning
- Learning rate: 3e-4

### 2.4 Phase 3: Evaluation

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Time (ms) |
|-------|---------|---------|---------|------|-----------|
| DistilBART | 42.3 | 19.8 | 38.5 | 17.6 | 185 |
| T5-small | 39.8 | 17.5 | 36.2 | 15.8 | 142 |
| TF-IDF | 31.2 | 11.4 | 27.8 | 7.2 | 48 |

---

## 3. Results & Analysis

### 3.1 Quantitative Results

**Best Performing Model**: DistilBART
- ROUGE-1: 42.3 (35% improvement over baseline)
- BERTScore-F1: 88.2
- Acceptable latency for web applications (~185ms)

### 3.2 Qualitative Examples

**Original Video Transcript** (truncated):
> "Today we discuss machine learning basics. First, let's understand supervised learning where we have labeled training data..."

**DistilBART Summary**:
> "This video covers machine learning fundamentals, explaining supervised learning with labeled data, unsupervised learning for pattern discovery, and practical applications in industry."

**TF-IDF Summary**:
> "Machine learning. Supervised learning. Training data. Patterns. Applications."

### 3.3 Trade-offs

| Approach | Speed | Quality | Best For |
|----------|-------|---------|----------|
| TF-IDF | ⭐⭐⭐ | ⭐ | Quick previews |
| T5-small | ⭐⭐ | ⭐⭐ | Real-time apps |
| DistilBART | ⭐ | ⭐⭐⭐ | Quality-focused |
| Gemini API | - | ⭐⭐⭐⭐ | Best quality |

---

## 4. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| GPU memory limits | Gradient accumulation, FP16 training |
| Slow inference | Model quantization, caching |
| Transcript noise | Custom preprocessing pipeline |
| Evaluation complexity | Multiple metrics (ROUGE, BLEU, BERTScore) |

---

## 5. Lessons Learned

1. **Pre-trained models matter**: DistilBART outperformed T5 due to summarization-specific pre-training
2. **Preprocessing is crucial**: 35%+ vocabulary reduction improved model efficiency
3. **Trade-offs are real**: No single model excels at both speed and quality
4. **Hybrid approaches work**: Combining extractive (TF-IDF) with abstractive methods yields balanced results

---

## 6. Future Work

1. **Multi-language support**: Extend to non-English videos
2. **Topic segmentation**: Summarize by video sections
3. **Speaker diarization**: Attribute summaries to speakers
4. **Mobile application**: React Native deployment
5. **Model distillation**: Create smaller, faster models

---

## 7. Conclusions

This project successfully developed a production-ready YouTube video summarization system that:

✅ Extracts and preprocesses video transcripts automatically  
✅ Provides multiple summarization methods (TF-IDF, Gensim, Transformers, AI)  
✅ Fine-tunes lightweight models achieving 42.3 ROUGE-1 score  
✅ Deploys as modern web application with real-time capabilities  
✅ Demonstrates practical NLP application from research to production

**Final Recommendation**: Deploy DistilBART as the default summarization model with TF-IDF fallback for resource-constrained scenarios.

---

## References

1. Lewis, M. et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training
2. Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with T5
3. Hermann, K. et al. (2015). Teaching Machines to Read and Comprehend
4. Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries

---

## Appendix

### A. Project Structure
```
YouTube-Summarizer/
├── backend/
│   ├── main.py
│   ├── services/
│   │   ├── finetuned_summarizer.py
│   │   ├── summarization_service.py
│   │   └── ...
├── frontend/
├── phase_2_model_training.py
├── phase_3_evaluation.py
├── Phase_1_Report.md
├── Phase_2_Report.md
├── Phase_3_Report.md
└── Final_Report.md
```

### B. API Endpoints
- `POST /api/summarize` - Generate summary with selected method
- `POST /api/qa` - Answer questions about video
- `GET /api/video-info` - Get video metadata
- `WS /ws/chat/{client_id}` - Real-time Q&A chat
