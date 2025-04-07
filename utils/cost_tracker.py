"""
Cost Tracking Utilities
"""
from typing import Dict, Any, Union, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

def calculate_cost(
    provider_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider_config: Dict
) -> float:
    """
    Calculate the cost for a given number of tokens using provider-specific rates.
    
    Args:
        provider_name: Name of the provider
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        provider_config: Provider configuration containing cost rates
        
    Returns:
        Cost in USD
    """
    # Get cost rates from provider config
    cost_rates = provider_config.get('cost_per_1k_tokens', {})
    
    # Default rates (free)
    prompt_rate = cost_rates.get('prompt', 0.0)
    completion_rate = cost_rates.get('completion', 0.0)
    
    # Calculate costs
    prompt_cost = (prompt_tokens / 1000) * prompt_rate
    completion_cost = (completion_tokens / 1000) * completion_rate
    total_cost = prompt_cost + completion_cost
    
    logger.debug(f"Cost calculation for {provider_name}: "
                f"Prompt: {prompt_tokens} tokens (${prompt_cost:.6f}), "
                f"Completion: {completion_tokens} tokens (${completion_cost:.6f}), "
                f"Total: ${total_cost:.6f}")
    
    return total_cost

def estimate_cost(
    prompt: str,
    max_tokens: int,
    provider_config: Dict,
    approx_tokens_per_word: float = 1.33
) -> Dict[str, float]:
    """
    Estimate the cost before making an API call.
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens for completion
        provider_config: Provider configuration
        approx_tokens_per_word: Approximate tokens per word ratio
        
    Returns:
        Dictionary with estimated costs
    """
    # Approximate prompt tokens
    words = len(prompt.split())
    estimated_prompt_tokens = int(words * approx_tokens_per_word)
    
    # Get cost rates
    cost_rates = provider_config.get('cost_per_1k_tokens', {})
    prompt_rate = cost_rates.get('prompt', 0.0)
    completion_rate = cost_rates.get('completion', 0.0)
    
    # Calculate estimated costs
    estimated_prompt_cost = (estimated_prompt_tokens / 1000) * prompt_rate
    estimated_completion_cost = (max_tokens / 1000) * completion_rate
    estimated_total_cost = estimated_prompt_cost + estimated_completion_cost
    
    return {
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_prompt_cost": estimated_prompt_cost,
        "estimated_completion_tokens": max_tokens,
        "estimated_completion_cost": estimated_completion_cost,
        "estimated_total_cost": estimated_total_cost
    }

def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 0.1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"