"""
Base LLM Provider Class - All specific providers will inherit from this
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

class LLMProvider(ABC):
    """Base abstract class for all LLM providers."""
    
    def __init__(self, config: Dict):
        """
        Initialize the LLM provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'unknown')
        self.priority = config.get('priority', 999)
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 1)
        
        # Process API keys - replace ${ENV_VAR} with actual environment variables
        self._process_api_keys()
        
        logger.info(f"Initialized provider: {self.name}")
    
    def _process_api_keys(self):
        """Process API keys from environment variables if needed."""
        api_key = self.config.get('api_key')
        
        if api_key and isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
            # Extract environment variable name
            env_var = api_key[2:-1]
            # Replace with actual value
            self.config['api_key'] = os.environ.get(env_var, '')
            
            if not self.config['api_key']:
                logger.warning(f"Environment variable {env_var} not set for provider {self.name}")
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter for generation
            
        Returns:
            Dictionary containing:
            - response: The generated text
            - tokens: Token count information (prompt, completion, total)
            - additional metadata
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass