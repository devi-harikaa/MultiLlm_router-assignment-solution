"""
Token Counter Utilities
"""
import tiktoken
from typing import Dict, Optional, Union

from utils.logger import get_logger

logger = get_logger(__name__)

# Cache for tiktoken encoders
_ENCODERS = {}

def get_encoder(model_name: str) -> Optional[tiktoken.Encoding]:
    """Get or create a tiktoken encoder for the specified model."""
    if model_name in _ENCODERS:
        return _ENCODERS[model_name]
    
    try:
        if "gpt" in model_name:
            encoder = tiktoken.encoding_for_model(model_name)
        else:
            # Default to cl100k_base for non-OpenAI models
            encoder = tiktoken.get_encoding("cl100k_base")
        
        _ENCODERS[model_name] = encoder
        return encoder
    
    except Exception as e:
        logger.warning(f"Failed to create encoder for {model_name}: {str(e)}")
        return None

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using the appropriate tokenizer.
    
    Args:
        text: The text to count tokens for
        model_name: Name of the model to use for token counting
        
    Returns:
        Number of tokens
    """
    encoder = get_encoder(model_name)
    
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
    
    # Fallback to approximate counting
    return approximate_token_count(text)

def approximate_token_count(text: str) -> int:
    """
    Approximate token count when a proper tokenizer isn't available.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate number of tokens
    """
    # Simple word-based approximation
    words = text.split()
    
    # Common approximation for English text: ~4/3 tokens per word
    return max(1, len(words) * 4 // 3)