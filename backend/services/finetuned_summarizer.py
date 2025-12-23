"""
Fine-tuned Summarizer Service

This service handles inference with fine-tuned DistilBART and T5-small models
for YouTube video transcript summarization.
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
import torch

logger = logging.getLogger(__name__)


class FinetunedModel(str, Enum):
    """Available fine-tuned models."""
    DISTILBART = "distilbart"
    T5_SMALL = "t5-small"


class FinetunedSummarizerService:
    """Service for fine-tuned model inference."""
    
    def __init__(self):
        """Initialize the service with lazy model loading."""
        self.models: Dict[str, any] = {}
        self.tokenizers: Dict[str, any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths (from Hugging Face Hub or local)
        self.model_paths = {
            FinetunedModel.DISTILBART: os.getenv(
                "DISTILBART_MODEL_PATH", 
                "sshleifer/distilbart-cnn-12-6"  # Fallback to base model
            ),
            FinetunedModel.T5_SMALL: os.getenv(
                "T5_MODEL_PATH",
                "t5-small"  # Fallback to base model
            )
        }
        
        # Generation config
        self.max_input_length = 1024
        self.max_output_length = 150
        self.t5_prefix = "summarize: "
        
        logger.info(f"FinetunedSummarizerService initialized on {self.device}")
    
    def _load_model(self, model_type: FinetunedModel) -> Tuple[any, any]:
        """Lazy load model and tokenizer."""
        
        if model_type in self.models:
            return self.models[model_type], self.tokenizers[model_type]
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_path = self.model_paths[model_type]
        logger.info(f"Loading {model_type.value} from {model_path}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model = model.to(self.device)
            model.eval()
            
            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            
            logger.info(f"{model_type.value} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_type.value}: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        model_type: FinetunedModel = FinetunedModel.DISTILBART,
        max_length: Optional[int] = None,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0
    ) -> Dict:
        """
        Generate summary using fine-tuned model.
        
        Args:
            text: Input text to summarize
            model_type: Which model to use
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Beam search width
            length_penalty: Penalty for longer sequences
            
        Returns:
            Dict with summary, model info, and metrics
        """
        
        if not text or len(text.strip()) < 50:
            return {
                "summary": text,
                "model": model_type.value,
                "error": "Text too short for summarization"
            }
        
        model, tokenizer = self._load_model(model_type)
        
        # Prepare input
        input_text = text
        if model_type == FinetunedModel.T5_SMALL:
            input_text = self.t5_prefix + text
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length or self.max_output_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Decode
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "summary": summary,
            "model": model_type.value,
            "model_path": self.model_paths[model_type],
            "inference_time_ms": round(inference_time, 2),
            "input_length": len(text.split()),
            "output_length": len(summary.split()),
            "compression_ratio": round(len(text.split()) / max(len(summary.split()), 1), 2)
        }
    
    def compare_models(self, text: str) -> Dict:
        """
        Run inference on both models for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Dict with results from both models
        """
        
        results = {}
        
        for model_type in FinetunedModel:
            try:
                results[model_type.value] = self.summarize(text, model_type)
            except Exception as e:
                results[model_type.value] = {"error": str(e)}
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        
        info = {
            "device": str(self.device),
            "available_models": [m.value for m in FinetunedModel],
            "loaded_models": list(self.models.keys())
        }
        
        for model_type, model in self.models.items():
            info[f"{model_type.value}_params"] = model.num_parameters()
        
        return info
    
    def unload_models(self):
        """Unload all models to free memory."""
        
        self.models.clear()
        self.tokenizers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("All models unloaded")


# Global service instance
_finetuned_service: Optional[FinetunedSummarizerService] = None


def get_finetuned_service() -> FinetunedSummarizerService:
    """Get or create global service instance."""
    global _finetuned_service
    
    if _finetuned_service is None:
        _finetuned_service = FinetunedSummarizerService()
    
    return _finetuned_service
