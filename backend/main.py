"""
FastAPI main application for YouTube Summarizer.
Provides REST API and WebSocket endpoints for summarization and Q&A.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models.schemas import (
    SummarizationRequest,
    SummarizationResponse,
    QuestionRequest,
    QuestionResponse,
    ErrorResponse,
    HealthResponse,
    SummarizationMethod,
    VideoInfo
)
from services.youtube_service import get_youtube_service
from services.nlp_service import get_nlp_service
from services.summarization_service import get_summarization_service
from services.gemini_service import get_gemini_service
from services.finetuned_summarizer import get_finetuned_service, FinetunedModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    logger.info("Starting YouTube Summarizer API...")
    
    # Initialize services
    nlp_service = get_nlp_service()
    youtube_service = get_youtube_service()
    summarization_service = get_summarization_service()
    gemini_service = get_gemini_service()
    
    logger.info("Services initialized")
    
    yield
    
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="YouTube Summarizer API",
    description="Advanced YouTube video summarization with NLP and AI",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for Q&A chat."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.contexts: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.contexts:
            del self.contexts[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        """Send message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    def set_context(self, client_id: str, context: str):
        """Store context for client session."""
        self.contexts[client_id] = context
    
    def get_context(self, client_id: str) -> Optional[str]:
        """Retrieve context for client session."""
        return self.contexts.get(client_id)


manager = ConnectionManager()


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and service availability."""
    nlp_service = get_nlp_service()
    gemini_service = get_gemini_service()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        gemini_available=gemini_service.is_available(),
        spacy_model_loaded=nlp_service.is_model_loaded()
    )


# Main summarization endpoint
@app.post("/api/summarize", response_model=SummarizationResponse)
async def summarize_video(request: SummarizationRequest):
    """
    Summarize YouTube video using specified method.
    
    Args:
        request: SummarizationRequest with URL, method, and fraction
        
    Returns:
        SummarizationResponse with summary and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing video: {request.url} with method: {request.method}")
        
        # Extract captions and video info
        youtube_service = get_youtube_service()
        captions, video_info = await youtube_service.get_captions(request.url)
        
        if not captions or len(captions.strip()) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video captions are too short or unavailable"
            )
        
        # Get NLP service for preprocessing
        nlp_service = get_nlp_service()
        
        # Generate summary based on method
        if request.method == SummarizationMethod.GEMINI:
            gemini_service = get_gemini_service()
            if not gemini_service.is_available():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Gemini API not available. Please configure GEMINI_API_KEY"
                )
            
            summary, suggested_questions = await gemini_service.summarize_text(
                captions,
                request.fraction,
                video_info.title
            )
        elif request.method in [SummarizationMethod.DISTILBART, SummarizationMethod.T5_SMALL]:
            # Use fine-tuned models
            finetuned_service = get_finetuned_service()
            model_type = (
                FinetunedModel.DISTILBART 
                if request.method == SummarizationMethod.DISTILBART 
                else FinetunedModel.T5_SMALL
            )
            result = finetuned_service.summarize(captions, model_type)
            summary = result["summary"]
            suggested_questions = []
        else:
            # Use traditional methods
            summarization_service = get_summarization_service()
            summary = await summarization_service.summarize(
                captions,
                request.method,
                request.fraction
            )
            suggested_questions = []
        
        # Calculate statistics
        stats = nlp_service.calculate_text_statistics(summary)
        
        # Extract entities and topics
        entities = nlp_service.extract_key_entities(summary)
        topics = nlp_service.get_key_topics(summary, top_n=8)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Summary generated in {processing_time:.2f}s")
        
        return SummarizationResponse(
            success=True,
            video_info=video_info,
            summary=summary,
            method_used=request.method,
            processing_time=processing_time,
            word_count=stats['word_count'],
            sentence_count=stats['sentence_count'],
            entities=entities[:10] if entities else None,
            key_topics=topics if topics else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process video: {str(e)}"
        )


# Q&A endpoint
@app.post("/api/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer question about video content.
    
    Args:
        request: QuestionRequest with question and context
        
    Returns:
        QuestionResponse with answer and metadata
    """
    try:
        gemini_service = get_gemini_service()
        
        if not gemini_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini API not available"
            )
        
        answer, confidence = await gemini_service.answer_question(
            request.question,
            request.video_context,
            request.conversation_history
        )
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat/{client_id}")
async def websocket_chat(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time Q&A chat.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await manager.connect(websocket, client_id)
    gemini_service = get_gemini_service()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get('type')
            
            if message_type == 'init':
                # Initialize chat with video context
                context = data.get('context', '')
                manager.set_context(client_id, context)
                
                if gemini_service.is_available():
                    await gemini_service.start_chat_session(context)
                    await manager.send_message(client_id, {
                        'type': 'status',
                        'message': 'Chat session initialized'
                    })
                else:
                    await manager.send_message(client_id, {
                        'type': 'error',
                        'message': 'Gemini service not available'
                    })
            
            elif message_type == 'message':
                # Process chat message
                question = data.get('message', '')
                context = manager.get_context(client_id)
                
                if not context:
                    await manager.send_message(client_id, {
                        'type': 'error',
                        'message': 'No context set. Please initialize chat first.'
                    })
                    continue
                
                try:
                    answer, confidence = await gemini_service.answer_question(
                        question,
                        context,
                        data.get('history', [])
                    )
                    
                    await manager.send_message(client_id, {
                        'type': 'answer',
                        'question': question,
                        'answer': answer,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    await manager.send_message(client_id, {
                        'type': 'error',
                        'message': f'Failed to process message: {str(e)}'
                    })
            
            elif message_type == 'ping':
                # Heartbeat
                await manager.send_message(client_id, {'type': 'pong'})
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)


# Video info endpoint (lightweight)
@app.get("/api/video-info")
async def get_video_info(url: str):
    """
    Get video information without processing captions.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video metadata
    """
    try:
        youtube_service = get_youtube_service()
        video_info = await youtube_service.extract_video_info(url)
        return video_info
        
    except Exception as e:
        logger.error(f"Error fetching video info: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch video info: {str(e)}"
        )


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug_mode
    )

