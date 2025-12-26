"""
Pydantic models for request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum


class SummarizationMethod(str, Enum):
    """Available summarization methods."""
    TFIDF = "tfidf"
    FREQUENCY = "frequency"
    GENSIM = "gensim"
    GEMINI = "gemini"
    HYBRID = "hybrid"
    DISTILBART = "distilbart"  # Fine-tuned model
    T5_SMALL = "t5-small"  # Fine-tuned model


class SummarizationRequest(BaseModel):
    """Request model for video summarization."""
    url: str = Field(..., description="YouTube video URL")
    method: SummarizationMethod = Field(
        default=SummarizationMethod.GEMINI,
        description="Summarization method to use"
    )
    fraction: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Fraction of content to include in summary (0.1-0.8)"
    )
    
    @field_validator('url')
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        """Validate that the URL is a valid YouTube URL."""
        youtube_patterns = [
            'youtube.com/watch',
            'youtu.be/',
            'youtube.com/embed/',
            'youtube.com/v/'
        ]
        if not any(pattern in v for pattern in youtube_patterns):
            raise ValueError('Invalid YouTube URL')
        return v


class TextSummarizationRequest(BaseModel):
    """Request model for direct text summarization (no YouTube fetching)."""
    text: str = Field(..., min_length=100, description="Text to summarize")
    method: SummarizationMethod = Field(
        default=SummarizationMethod.GEMINI,
        description="Summarization method to use"
    )
    fraction: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Fraction of content to include in summary (0.1-0.8)"
    )
    video_title: Optional[str] = Field(
        default="Video",
        description="Optional video title for context"
    )


class VideoInfo(BaseModel):
    """Video metadata information."""
    title: str
    duration: Optional[int] = None
    channel: Optional[str] = None
    description: Optional[str] = None


class SummarizationResponse(BaseModel):
    """Response model for video summarization."""
    success: bool
    video_info: VideoInfo
    summary: str
    method_used: SummarizationMethod
    processing_time: float
    word_count: int
    sentence_count: int
    entities: Optional[List[str]] = None
    key_topics: Optional[List[str]] = None


class QuestionRequest(BaseModel):
    """Request model for Q&A."""
    question: str = Field(..., min_length=3, max_length=500)
    video_context: str = Field(..., description="Video transcript or summary for context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous Q&A pairs for context"
    )


class QuestionResponse(BaseModel):
    """Response model for Q&A."""
    question: str
    answer: str
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    suggested_questions: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    gemini_available: bool
    spacy_model_loaded: bool

