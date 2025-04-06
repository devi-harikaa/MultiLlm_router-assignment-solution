"""
Tests for the Provider Manager
"""
import pytest
from unittest.mock import MagicMock, patch

from services.provider_manager import ProviderManager
from services.llm_provider import LLMProvider

# Sample test configuration
TEST_CONFIG = {
    'providers': [
        {
            'name': 'test_provider_1',
            'type': 'test',
            'enabled': True,
            'priority': 1,
            'cost_per_1k_tokens': {
                'prompt': 0.001,
                'completion': 0.002
            }
        },
        {
            'name': 'test_provider_2',
            'type': 'test',
            'enabled': True,
            'priority': 2,
            'cost_per_1k_tokens': {
                'prompt': 0.003,
                'completion': 0.004
            }
        }
    ],
    'settings': {
        'default_max_tokens': 100,
        'default_temperature': 0.7
    }
}

# Mock provider class for testing
class MockProvider(LLMProvider):
    def __init__(self, config):
        super().__init__(config)
        self.mock_response = {
            "response": "This is a test response",
            "tokens": {
                "prompt": 5,
                "completion": 10,
                "total": 15
            }
        }
    
    def generate(self, prompt, max_tokens, temperature):
        return self.mock_response
    
    def count_tokens(self, text):
        return len(text.split())

# Patch the dynamic import to return our mock provider
@pytest.fixture
def mock_importlib():
    with patch('importlib.import_module') as mock_import:
        # Create a mock module with our MockProvider class
        mock_module = MagicMock()
        mock_module.TestProvider = MockProvider
        mock_import.return_value = mock_module
        yield mock_import

@pytest.fixture
def provider_manager(mock_importlib):
    # Use the patched importlib to create a ProviderManager with mock providers
    manager = ProviderManager(TEST_CONFIG)
    return manager

def test_provider_loading(provider_manager):
    """Test that providers are loaded correctly."""
    assert len(provider_manager.providers) == 2
    assert provider_manager.providers[0].name == 'test_provider_1'
    assert provider_manager.providers[1].name == 'test_provider_2'
    assert provider_manager.providers[0].priority < provider_manager.providers[1].priority

def test_generate_success(provider_manager):
    """Test successful generation with the first provider."""
    result = provider_manager.generate("Test prompt")
    
    assert result['modelUsed'] == 'test_provider_1'
    assert 'response' in result
    assert 'cost' in result
    assert result['tokens']['total'] == 15

def test_generate_fallback(provider_manager):
    """Test fallback to the second provider when the first one fails."""
    # Make the first provider fail
    provider_manager.providers[0].generate = MagicMock(side_effect=Exception("Provider failed"))
    
    result = provider_manager.generate("Test prompt")
    
    assert result['modelUsed'] == 'test_provider_2'
    assert 'response' in result
    assert 'cost' in result
    assert result['tokens']['total'] == 15

def test_all_providers_fail(provider_manager):
    """Test behavior when all providers fail."""
    # Make all providers fail
    for provider in provider_manager.providers:
        provider.generate = MagicMock(side_effect=Exception("Provider failed"))
    
    with pytest.raises(Exception) as excinfo:
        provider_manager.generate("Test prompt")
    
    assert "All providers failed" in str(excinfo.value)

def test_cost_calculation(provider_manager):
    """Test that costs are calculated correctly."""
    result = provider_manager.generate("Test prompt")
    
    # Expected cost calculation for test_provider_1:
    # Prompt: 5 tokens * $0.001/1000 = $0.000005
    # Completion: 10 tokens * $0.002/1000 = $0.00002
    # Total: $0.000025
    expected_cost = 0.000025
    
    assert result['cost'] == pytest.approx(expected_cost)