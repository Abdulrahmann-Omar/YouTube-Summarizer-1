"""
Enhanced NLP service with spaCy integration and advanced preprocessing.
"""

import spacy
from typing import Optional, Dict, List, Tuple
from functools import lru_cache
import logging

from config import settings
from utils.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class NLPService:
    """
    Service for NLP operations including preprocessing, tokenization,
    and text analysis using spaCy.
    """
    
    _instance: Optional['NLPService'] = None
    _nlp: Optional[spacy.language.Language] = None
    _preprocessor: Optional[TextPreprocessor] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize NLP service with spaCy model."""
        if self._nlp is None:
            self._load_model()
    
    def _load_model(self):
        """Load spaCy model based on configuration."""
        try:
            logger.info(f"Loading spaCy model: {settings.spacy_model}")
            self._nlp = spacy.load(settings.spacy_model)
            
            # Add sentence segmentation component if not present
            if 'sentencizer' not in self._nlp.pipe_names and 'parser' not in self._nlp.pipe_names:
                self._nlp.add_pipe('sentencizer')
            
            self._preprocessor = TextPreprocessor(self._nlp)
            logger.info("spaCy model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.info("Falling back to en_core_web_sm")
            try:
                self._nlp = spacy.load('en_core_web_sm')
                self._preprocessor = TextPreprocessor(self._nlp)
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load spaCy model. "
                    "Please install: python -m spacy download en_core_web_sm"
                )
    
    @property
    def nlp(self) -> spacy.language.Language:
        """Get spaCy model instance."""
        if self._nlp is None:
            self._load_model()
        return self._nlp
    
    @property
    def preprocessor(self) -> TextPreprocessor:
        """Get text preprocessor instance."""
        if self._preprocessor is None:
            self._load_model()
        return self._preprocessor
    
    def process_text(self, text: str) -> spacy.tokens.Doc:
        """
        Process text with spaCy pipeline.
        
        Args:
            text: Input text
            
        Returns:
            spaCy Doc object
        """
        return self.nlp(text)
    
    def preprocess_for_summarization(
        self, 
        text: str
    ) -> Tuple[str, spacy.tokens.Doc, Dict]:
        """
        Preprocess text for summarization with full pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Tuple of (cleaned_text, doc, metadata)
        """
        return self.preprocessor.preprocess_for_summarization(text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def extract_key_entities(self, text: str) -> List[str]:
        """
        Extract key named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity texts
        """
        doc = self.nlp(text)
        entities = self.preprocessor.extract_entities(doc)
        
        # Flatten and return unique entities
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        return list(set(all_entities))
    
    def get_key_topics(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key topics/terms from text.
        
        Args:
            text: Input text
            top_n: Number of top topics to return
            
        Returns:
            List of key topic strings
        """
        doc = self.nlp(text)
        key_terms = self.preprocessor.get_key_terms(doc, top_n=top_n)
        return [term for term, _ in key_terms]
    
    def calculate_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Calculate basic statistics about text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with statistics
        """
        doc = self.nlp(text)
        
        return {
            'word_count': len([token for token in doc if token.is_alpha]),
            'sentence_count': len(list(doc.sents)),
            'character_count': len(text),
            'unique_words': len(set([token.text.lower() for token in doc if token.is_alpha]))
        }
    
    def is_model_loaded(self) -> bool:
        """Check if spaCy model is loaded."""
        return self._nlp is not None


# Global instance
_nlp_service: Optional[NLPService] = None


def get_nlp_service() -> NLPService:
    """
    Get or create global NLP service instance.
    
    Returns:
        NLPService instance
    """
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service

