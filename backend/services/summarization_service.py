"""
Enhanced summarization service with multiple algorithms.
Includes TF-IDF, Frequency-based, Gensim, and Gemini-based summarization.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy

from services.nlp_service import get_nlp_service
from models.schemas import SummarizationMethod

logger = logging.getLogger(__name__)

# Try to import gensim summarization, but handle if it's not available
try:
    from gensim.summarization import summarize as gensim_summarize
    GENSIM_AVAILABLE = True
except (ImportError, AttributeError):
    GENSIM_AVAILABLE = False
    logger.warning("Gensim summarization not available. Gensim method will fall back to frequency-based.")


class SummarizationService:
    """
    Service providing multiple summarization algorithms with enhancements.
    """
    
    def __init__(self):
        """Initialize summarization service."""
        self.nlp_service = get_nlp_service()
    
    async def summarize(
        self,
        text: str,
        method: SummarizationMethod,
        fraction: float = 0.3
    ) -> str:
        """
        Summarize text using specified method.
        
        Args:
            text: Input text to summarize
            method: Summarization method to use
            fraction: Fraction of content to include (0.1-0.8)
            
        Returns:
            Summarized text
        """
        # Preprocess text
        cleaned_text, doc, metadata = self.nlp_service.preprocess_for_summarization(text)
        
        if method == SummarizationMethod.TFIDF:
            return self.tfidf_based(cleaned_text, doc, fraction)
        elif method == SummarizationMethod.FREQUENCY:
            return self.frequency_based(cleaned_text, doc, fraction)
        elif method == SummarizationMethod.GENSIM:
            return self.gensim_based(cleaned_text, doc, fraction)
        else:
            raise ValueError(f"Method {method} should be handled by Gemini service")
    
    def tfidf_based(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        fraction: float = 0.3
    ) -> str:
        """
        Enhanced TF-IDF based summarization with n-grams and entity weighting.
        
        Args:
            text: Cleaned text
            doc: spaCy document
            fraction: Fraction of sentences to include
            
        Returns:
            Summary text
        """
        try:
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            if len(sentences) == 0:
                return text[:500]  # Fallback
            
            # Calculate number of sentences to extract
            num_sentences = max(1, int(np.ceil(len(sentences) * fraction)))
            
            # If we have fewer sentences than requested, return all
            if len(sentences) <= num_sentences:
                return " ".join(sentences)
            
            # Create TF-IDF with unigrams and bigrams
            tfidf = TfidfVectorizer(
                stop_words='english',
                token_pattern=r'(?ui)\b\w*[a-z]+\w*\b',
                ngram_range=(1, 2),  # Include bigrams
                max_features=500
            )
            
            # Fit and transform
            X = tfidf.fit_transform(sentences)
            
            # Create dataframe with TF-IDF scores
            df = pd.DataFrame(
                data=X.todense(),
                columns=tfidf.get_feature_names_out()
            )
            
            # Calculate sentence scores (sum of TF-IDF values)
            sentence_scores = df.sum(axis=1)
            
            # Add position-based weighting (favor earlier sentences slightly)
            position_weights = np.array([
                1.0 + (0.3 * (1 - i / len(sentences)))
                for i in range(len(sentences))
            ])
            weighted_scores = sentence_scores * position_weights
            
            # Get top sentences
            top_indices = weighted_scores.nlargest(num_sentences).index.tolist()
            
            # Sort indices to maintain order
            top_indices.sort()
            
            # Extract and join summary sentences
            summary = " ".join([sentences[i] for i in top_indices])
            
            return summary
            
        except Exception as e:
            logger.error(f"TF-IDF summarization error: {e}")
            # Fallback to simple truncation
            return " ".join([sent.text for sent in doc.sents][:int(len(list(doc.sents)) * fraction)])
    
    def frequency_based(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        fraction: float = 0.3
    ) -> str:
        """
        Enhanced frequency-based summarization with position scoring.
        
        Args:
            text: Cleaned text
            doc: spaCy document
            fraction: Fraction of sentences to include
            
        Returns:
            Summary text
        """
        try:
            sentences = [sent for sent in doc.sents]
            
            if len(sentences) == 0:
                return text[:500]
            
            num_sentences = max(1, int(np.ceil(len(sentences) * fraction)))
            
            if len(sentences) <= num_sentences:
                return " ".join([sent.text.strip() for sent in sentences])
            
            # Extract and count words (lemmatized)
            words = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
                and not token.is_stop
                and len(token.text) > 2
            ]
            
            # Calculate weighted frequency
            word_freq = Counter(words)
            max_freq = max(word_freq.values()) if word_freq else 1
            
            word_weights = {
                word: freq / max_freq
                for word, freq in word_freq.items()
            }
            
            # Calculate sentence scores
            sentence_scores = []
            for idx, sent in enumerate(sentences):
                score = 0.0
                word_count = 0
                
                for token in sent:
                    if token.lemma_.lower() in word_weights:
                        score += word_weights[token.lemma_.lower()]
                        word_count += 1
                
                # Normalize by sentence length
                if word_count > 0:
                    score = score / word_count
                
                # Add position bonus (favor earlier sentences)
                position_bonus = 1.0 + (0.5 * (1 - idx / len(sentences)))
                score *= position_bonus
                
                # Penalty for very short or very long sentences
                sent_length = len([t for t in sent if t.is_alpha])
                if sent_length < 5:
                    score *= 0.5
                elif sent_length > 50:
                    score *= 0.8
                
                sentence_scores.append(score)
            
            # Get top sentences
            scored_sentences = list(enumerate(sentence_scores))
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in scored_sentences[:num_sentences]]
            
            # Sort to maintain order
            top_indices.sort()
            
            # Build summary
            summary = " ".join([sentences[i].text.strip() for i in top_indices])
            
            return summary
            
        except Exception as e:
            logger.error(f"Frequency-based summarization error: {e}")
            return " ".join([sent.text for sent in doc.sents][:int(len(list(doc.sents)) * fraction)])
    
    def gensim_based(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        fraction: float = 0.3
    ) -> str:
        """
        Gensim-based summarization using TextRank algorithm.
        Falls back to frequency-based if gensim is not available.
        
        Args:
            text: Cleaned text
            doc: spaCy document
            fraction: Fraction of content to include
            
        Returns:
            Summary text
        """
        # Check if gensim is available
        if not GENSIM_AVAILABLE:
            logger.info("Gensim not available, using frequency-based method")
            return self.frequency_based(text, doc, fraction)
        
        try:
            # Gensim requires sentence-per-line format
            sentences = [sent.text.strip() for sent in doc.sents]
            
            if len(sentences) <= 2:
                return " ".join(sentences)
            
            # Join with newlines for gensim
            formatted_text = "\n".join(sentences)
            
            # Try gensim summarization
            try:
                summary = gensim_summarize(
                    formatted_text,
                    ratio=fraction,
                    split=False
                )
                
                if summary and len(summary.strip()) > 0:
                    return summary
                else:
                    # If gensim returns empty, fall back to word_count method
                    word_count = int(len(text.split()) * fraction)
                    summary = gensim_summarize(
                        formatted_text,
                        word_count=max(50, word_count),
                        split=False
                    )
                    if summary:
                        return summary
                    
            except ValueError as ve:
                logger.warning(f"Gensim summarization failed, falling back: {ve}")
            
            # Fallback: use frequency-based
            return self.frequency_based(text, doc, fraction)
            
        except Exception as e:
            logger.error(f"Gensim summarization error: {e}")
            return self.frequency_based(text, doc, fraction)
    
    def hybrid_summarization(
        self,
        text: str,
        fraction: float = 0.3
    ) -> str:
        """
        Hybrid approach combining multiple methods.
        
        Args:
            text: Input text
            fraction: Fraction of content to include
            
        Returns:
            Summary text
        """
        cleaned_text, doc, metadata = self.nlp_service.preprocess_for_summarization(text)
        
        # Get summaries from multiple methods
        tfidf_summary = self.tfidf_based(cleaned_text, doc, fraction)
        freq_summary = self.frequency_based(cleaned_text, doc, fraction)
        
        # Extract sentences from both
        tfidf_sents = set(self.nlp_service.extract_sentences(tfidf_summary))
        freq_sents = set(self.nlp_service.extract_sentences(freq_summary))
        
        # Find common sentences (high confidence)
        common_sents = tfidf_sents & freq_sents
        
        # Add unique sentences from each method
        all_sents = list(common_sents) + list(tfidf_sents - common_sents)[:2] + list(freq_sents - common_sents)[:2]
        
        # Sort by original order
        all_sentences = [sent.text for sent in doc.sents]
        ordered_sents = []
        for sent in all_sentences:
            if sent in all_sents:
                ordered_sents.append(sent)
        
        return " ".join(ordered_sents[:int(len(all_sentences) * fraction) + 1])


# Global instance
_summarization_service = None


def get_summarization_service() -> SummarizationService:
    """Get or create global summarization service instance."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service

