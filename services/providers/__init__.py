"""
Provider package initialization file
"""
# Import all provider classes for easy access
from services.providers.openai_provider import OpenaiProvider
from services.providers.huggingface_provider import HuggingfaceProvider
from services.providers.llama_provider import LlamaProvider

# List of available provider classes
AVAILABLE_PROVIDERS = {
    'openai': OpenaiProvider,
    'huggingface': HuggingfaceProvider,
    'llama': LlamaProvider
}