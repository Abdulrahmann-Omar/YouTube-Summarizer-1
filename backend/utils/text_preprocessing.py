"""
Advanced text preprocessing and tokenization utilities.
Implements enhanced NLP preprocessing pipeline with noise removal,
normalization, and entity recognition.
"""

import re
import string
from typing import List, Tuple, Set, Dict
import spacy
from spacy.tokens import Doc
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Advanced text preprocessing class for cleaning and normalizing text.
    """
    
    def __init__(self, nlp: spacy.language.Language):
        """
        Initialize preprocessor with spaCy model.
        
        Args:
            nlp: Loaded spaCy language model
        """
        self.nlp = nlp
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def clean_youtube_captions(self, text: str) -> str:
        """
        Clean YouTube caption text by removing timestamps and formatting.
        
        Args:
            text: Raw caption text
            
        Returns:
            Cleaned text
        """
        # Remove timestamps (e.g., "00:00:00.000 --> 00:00:02.000")
        text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', '', text)
        
        # Remove timestamp markers (e.g., "[00:00]")
        text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
        
        # Remove duplicate whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove newlines and replace with spaces
        text = text.replace('\n', ' ')
        
        # Remove music/sound effects markers
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?music.*?\)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def remove_duplicate_sentences(self, text: str) -> str:
        """
        Remove duplicate or near-duplicate sentences.
        
        Args:
            text: Input text
            
        Returns:
            Text with duplicates removed
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Use set to track seen sentences (case-insensitive)
        seen = set()
        unique_sentences = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            if sent_lower not in seen and len(sent.split()) > 3:  # Skip very short sentences
                seen.add(sent_lower)
                unique_sentences.append(sent)
        
        return ' '.join(unique_sentences)
    
    def normalize_text(self, text: str, lowercase: bool = False) -> str:
        """
        Normalize text by handling contractions, special characters, etc.
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            
        Returns:
            Normalized text
        """
        # Expand common contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        if lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, doc: Doc) -> Dict[str, List[str]]:
        """
        Extract named entities from document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary mapping entity types to entity texts
        """
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def get_key_terms(self, doc: Doc, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract key terms using lemmatization and frequency.
        
        Args:
            doc: spaCy Doc object
            top_n: Number of top terms to return
            
        Returns:
            List of (term, frequency) tuples
        """
        # Extract lemmatized tokens, filtering stopwords and non-alphabetic
        terms = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha 
            and not token.is_stop 
            and len(token.text) > 2
            and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']
        ]
        
        # Count frequencies
        term_freq = Counter(terms)
        
        return term_freq.most_common(top_n)
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences using spaCy's sentence tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
    
    def preprocess_for_summarization(
        self, 
        text: str, 
        preserve_entities: bool = True
    ) -> Tuple[str, Doc, Dict[str, any]]:
        """
        Complete preprocessing pipeline for summarization.
        
        Args:
            text: Raw input text
            preserve_entities: Whether to extract and preserve entities
            
        Returns:
            Tuple of (cleaned_text, spacy_doc, metadata)
        """
        # Step 1: Clean YouTube-specific artifacts
        cleaned = self.clean_youtube_captions(text)
        
        # Step 2: Remove duplicates
        cleaned = self.remove_duplicate_sentences(cleaned)
        
        # Step 3: Normalize
        cleaned = self.normalize_text(cleaned)
        
        # Step 4: Process with spaCy
        doc = self.nlp(cleaned)
        
        # Step 5: Extract metadata
        metadata = {
            'sentence_count': len(list(doc.sents)),
            'word_count': len([token for token in doc if token.is_alpha]),
            'entities': self.extract_entities(doc) if preserve_entities else {},
            'key_terms': self.get_key_terms(doc, top_n=15)
        }
        
        return cleaned, doc, metadata
    
    def get_sentences_with_scores(
        self, 
        doc: Doc, 
        score_dict: Dict[int, float]
    ) -> List[Tuple[str, float, int]]:
        """
        Get sentences with their scores and positions.
        
        Args:
            doc: spaCy Doc object
            score_dict: Dictionary mapping sentence indices to scores
            
        Returns:
            List of (sentence_text, score, position) tuples
        """
        sentences = []
        for idx, sent in enumerate(doc.sents):
            score = score_dict.get(idx, 0.0)
            sentences.append((sent.text.strip(), score, idx))
        
        return sentences

