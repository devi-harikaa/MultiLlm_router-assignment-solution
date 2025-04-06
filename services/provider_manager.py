import json
import os
import time
from typing import Dict, List, Any, Optional
import importlib

from utils.logger import get_logger
from services.llm_provider import LLMProvider
from utils.cost_tracker import calculate_cost

logger = get_logger(__name__)

class ProviderManager:
    """
    Manages multiple LLM providers, handles routing, fallback, and tracking.
    """
    
    def __init__(self, config: Dict):
        """Initialize the provider manager with configuration."""
        self.config = config
        self.providers = []
        self.settings = config.get('settings', {})
        
        # Load all providers
        self._load_providers()
        
        # Sort providers by priority
        self.providers.sort(key=lambda p: p.priority)
        
        logger.info(f"Initialized {len(self.providers)} providers")
    
    def _load_providers(self):
        """Dynamically load and initialize providers from configuration."""
        provider_configs = self.config.get('providers', [])
        
        for provider_config in provider_configs:
            if not provider_config.get('enabled', True):
                continue
                
            provider_type = provider_config.get('type')
            if not provider_type:
                logger.warning(f"Provider missing 'type' field: {provider_config.get('name', 'unknown')}")
                continue
                
            try:
                # Dynamically import the provider module
                module_name = f"services.providers.{provider_type}_provider"
                module = importlib.import_module(module_name)
                
                # Get the provider class (assuming it follows the naming convention)
                class_name = f"{provider_type.capitalize()}Provider"
                provider_class = getattr(module, class_name)
                
                # Create provider instance
                provider = provider_class(provider_config)
                self.providers.append(provider)
                
                logger.info(f"Loaded provider: {provider.name}")
            
            except (ImportError, AttributeError, Exception) as e:
                logger.error(f"Failed to load provider {provider_type}: {str(e)}")
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> Dict:
        """
        Generate text using the most cost-effective provider with fallback logic.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with generation results, provider used, cost, etc.
        """
        if not max_tokens:
            max_tokens = self.settings.get('default_max_tokens', 100)
            
        if not temperature:
            temperature = self.settings.get('default_temperature', 0.7)
        
        # Try each provider in order of priority
        for provider in self.providers:
            try:
                logger.info(f"Attempting to generate with provider: {provider.name}")
                
                result = provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Calculate cost
                token_info = result.get('tokens', {})
                cost = calculate_cost(
                    provider_name=provider.name,
                    prompt_tokens=token_info.get('prompt', 0),
                    completion_tokens=token_info.get('completion', 0),
                    provider_config=provider.config
                )
                
                # Add cost to result
                result['cost'] = cost
                result['modelUsed'] = provider.name
                
                # Log usage
                self._log_usage(result)
                
                logger.info(f"Successfully generated with {provider.name}. "
                           f"Tokens: {token_info.get('total', 0)}, Cost: ${cost:.6f}")
                
                return result
                
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {str(e)}")
                continue
        
        # If we get here, all providers failed
        raise Exception("All providers failed to generate response")
    
    def _log_usage(self, result: Dict):
        """Log usage data to storage."""
        try:
            # Add timestamp
            result['timestamp'] = time.time()
            
            # Load existing logs
            logs_path = 'storage/usage_logs.json'
            logs = []
            
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as file:
                    try:
                        logs = json.load(file)
                    except json.JSONDecodeError:
                        logs = []
            
            # Append new log
            logs.append(result)
            
            # Write updated logs
            with open(logs_path, 'w') as file:
                json.dump(logs, file)
                
        except Exception as e:
            logger.error(f"Failed to log usage: {str(e)}")
    
    def get_provider_status(self) -> List[Dict]:
        """Get status of all providers."""
        return [
            {
                "name": provider.name,
                "enabled": provider.config.get('enabled', True),
                "priority": provider.priority
            }
            for provider in self.providers
        ]