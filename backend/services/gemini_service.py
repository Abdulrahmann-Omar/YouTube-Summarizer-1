"""
Google Gemini API integration for advanced summarization and Q&A.
Uses Gemini 1.5 Flash (free tier) for optimal performance.
"""

import logging
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Service for interacting with Google Gemini API.
    Provides summarization and Q&A capabilities.
    """
    
    def __init__(self):
        """Initialize Gemini service with API key."""
        self.model = None
        self.chat_session = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model with configuration."""
        try:
            if not settings.gemini_api_key or settings.gemini_api_key == "":
                logger.warning("Gemini API key not configured")
                return
            
            genai.configure(api_key=settings.gemini_api_key)
            
            # Use Gemini 1.5 Flash for free tier (fast and efficient)
            self.model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            logger.info("Gemini model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        return self.model is not None
    
    async def summarize_text(
        self,
        text: str,
        fraction: float = 0.3,
        video_title: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate abstractive summary using Gemini.
        
        Args:
            text: Input text to summarize
            fraction: Desired summary length (relative to original)
            video_title: Optional video title for context
            
        Returns:
            Tuple of (summary, suggested_questions)
        """
        if not self.is_available():
            raise RuntimeError("Gemini service not available")
        
        try:
            # Calculate target word count
            word_count = len(text.split())
            target_words = int(word_count * fraction)
            target_words = max(100, min(target_words, 800))  # Clamp between 100-800 words
            
            # Create optimized prompt for educational content
            title_context = f" about '{video_title}'" if video_title else ""
            
            prompt = f"""You are an expert at summarizing educational video content. 

Please provide a comprehensive, well-structured summary of this YouTube video transcript{title_context}. 

**Requirements:**
1. Create a summary of approximately {target_words} words
2. Focus on key concepts, main ideas, and important details
3. Use clear, concise language
4. Organize information logically with proper flow
5. Preserve technical terms and important names
6. Highlight actionable insights or takeaways

**Transcript:**
{text[:50000]}

**Summary:**"""

            # Generate summary
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Generate suggested questions
            questions = await self._generate_questions(text[:30000], summary)
            
            return summary, questions
            
        except Exception as e:
            logger.error(f"Gemini summarization error: {e}")
            raise RuntimeError(f"Failed to generate summary: {str(e)}")
    
    async def _generate_questions(
        self,
        context: str,
        summary: str
    ) -> List[str]:
        """
        Generate suggested questions based on content.
        
        Args:
            context: Original text
            summary: Generated summary
            
        Returns:
            List of suggested questions
        """
        try:
            prompt = f"""Based on this video content summary, generate 5 insightful questions that someone might ask to better understand the topic:

Summary: {summary[:1000]}

Generate questions that:
1. Explore key concepts in depth
2. Ask about practical applications
3. Seek clarification on complex topics
4. Connect ideas to real-world scenarios

Respond with ONLY the questions, one per line, numbered 1-5."""

            response = self.model.generate_content(prompt)
            questions_text = response.text.strip()
            
            # Parse questions
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                    # Remove number prefix
                    question = line.split('.', 1)[1].strip()
                    questions.append(question)
            
            return questions[:5]
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    async def answer_question(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, float]:
        """
        Answer question based on video context.
        
        Args:
            question: User question
            context: Video transcript/summary
            conversation_history: Previous Q&A pairs
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not self.is_available():
            raise RuntimeError("Gemini service not available")
        
        try:
            # Build conversation context
            history_text = ""
            if conversation_history:
                for item in conversation_history[-5:]:  # Last 5 exchanges
                    history_text += f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}\n\n"
            
            prompt = f"""You are an expert assistant helping someone understand a YouTube video.

**Video Content:**
{context[:40000]}

**Conversation History:**
{history_text}

**Current Question:**
{question}

**Instructions:**
1. Answer based ONLY on the video content provided
2. Be specific and cite relevant parts of the video
3. If the answer isn't in the video, say so clearly
4. Keep answers concise but comprehensive (2-4 sentences)
5. Use simple, clear language

**Answer:**"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Simple confidence scoring based on answer characteristics
            confidence = self._calculate_confidence(answer)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Gemini Q&A error: {e}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
    
    def _calculate_confidence(self, answer: str) -> float:
        """
        Calculate confidence score for answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence
        
        # Increase confidence if answer is substantive
        if len(answer.split()) > 20:
            confidence += 0.1
        
        # Decrease if answer indicates uncertainty
        uncertainty_phrases = [
            "i don't know",
            "not mentioned",
            "not in the video",
            "cannot determine",
            "unclear"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Increase if answer cites specifics
        if any(word in answer.lower() for word in ["specifically", "mentioned", "states", "explains"]):
            confidence += 0.1
        
        return max(0.1, min(confidence, 1.0))
    
    async def start_chat_session(self, context: str) -> None:
        """
        Start a new chat session with video context.
        
        Args:
            context: Video transcript or summary
        """
        if not self.is_available():
            raise RuntimeError("Gemini service not available")
        
        try:
            system_instruction = f"""You are a helpful assistant for a YouTube video summarizer.
The user is watching/reviewing this video content:

{context[:30000]}

Your role:
- Answer questions about the video content
- Provide clarifications and explanations
- Help the user understand key concepts
- Be conversational and friendly
- Always base answers on the video content provided"""

            self.chat_session = self.model.start_chat(history=[])
            
            # Prime the chat with context
            self.chat_session.send_message(
                f"I've just watched a video with this content: {context[:10000]}"
            )
            
        except Exception as e:
            logger.error(f"Error starting chat session: {e}")
            raise RuntimeError(f"Failed to start chat: {str(e)}")
    
    async def chat(self, message: str) -> str:
        """
        Send message in active chat session.
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        if not self.chat_session:
            raise RuntimeError("No active chat session")
        
        try:
            response = self.chat_session.send_message(message)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise RuntimeError(f"Failed to process chat message: {str(e)}")


# Global instance
_gemini_service: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Get or create global Gemini service instance."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service

