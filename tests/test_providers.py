"""
Tests for LLM Providers
"""
import pytest
from unittest.mock import MagicMock, patch

from services.providers.groq_provider import GroqProvider
from services.providers.huggingface_provider import HuggingfaceProvider
from services.providers.llama_provider import LlamaProvider

# Test configurations
OPENAI_CONFIG = {
    'name': 'groq',
    'endpoint': "https://api.groq.com/openai/v1/chat/completions",
    'priority': '1',
    'cost_per_1k_tokens': '0.002',
    'api_key': "api_key",
    'model': "llama-3.1-8b-instant"
    }

HUGGINGFACE_CONFIG = {
    'name': 'huggingface_test',
    'type': 'huggingface',
    'enabled': True,
    'priority': 2,
    'api_key': 'test_key',
    'model': 'google/flan-t5-base',
    'cost_per_1k_tokens': {
        'prompt': 0.0,
        'completion': 0.0
    }
}

LLAMA_CONFIG = {
    'name': 'llama_test',
    'type': 'llama',
    'enabled': True,
    'priority': 3,
    'endpoint': 'http://localhost:11434/api/generate',
    'cost_per_1k_tokens': {
        'prompt': 0.0,
        'completion': 0.0
    }
}

@pytest.fixture
def mock_requests():
    with patch('requests.post') as mock_post:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        
        # Configure return values for different providers
        mock_post.return_value = mock_response
        
        # OpenAI response
        openai_json = {
            'choices': [{'text': 'This is a test response from OpenAI'}],
            'usage': {
                'prompt_tokens': 5,
                'completion_tokens': 10,
                'total_tokens': 15
            }
        }
        
        # Hugging Face response
        hf_json = [
            {'generated_text': 'This is a test response from Hugging Face'}
        ]
        
        # Llama response
        llama_json = {
            'content': 'This is a test response from Llama',
            'tokens_predicted': 10,
            'tokens_evaluated': 5
        }
        
        # Configure mock to return different responses based on URL
        def side_effect_function(*args, **kwargs):
            url = kwargs.get('url', '') or args[0]
            
            if 'openai.com' in str(url):
                mock_response.json.return_value = openai_json
            elif 'huggingface.co' in str(url):
                mock_response.json.return_value = hf_json
            elif 'localhost:8080' in str(url):
                mock_response.json.return_value = llama_json
                
            return mock_response
            
        mock_post.side_effect = side_effect_function
        
        yield mock_post

# OpenAI Provider Tests
@pytest.fixture
def openai_provider(mock_requests):
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        # Mock the tiktoken encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(5))  # Dummy token IDs
        mock_encoding.return_value = mock_encoder
        
        provider = GroqProvider(OPENAI_CONFIG)
        return provider

def test_openai_provider_init(openai_provider):
    """Test OpenAI provider initialization."""
    assert openai_provider.name == 'openai_test'
    assert openai_provider.model == 'gpt-3.5-turbo-instruct'
    assert openai_provider.api_key == 'test_key'

def test_openai_provider_generate(openai_provider, mock_requests):
    """Test OpenAI provider text generation."""
    result = openai_provider.generate(
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.7
    )
    
    assert 'response' in result
    assert result['response'] == 'This is a test response from OpenAI'
    assert result['tokens']['prompt'] == 5
    assert result['tokens']['completion'] == 10
    assert result['tokens']['total'] == 15
    
    # Verify API call
    mock_requests.assert_called_once()
    args, kwargs = mock_requests.call_args
    assert kwargs['json']['model'] == 'gpt-3.5-turbo-instruct'
    assert kwargs['json']['prompt'] == 'Test prompt'
    assert kwargs['json']['max_tokens'] == 100
    assert kwargs['json']['temperature'] == 0.7

# Hugging Face Provider Tests
@pytest.fixture
def hf_provider(mock_requests):
    provider = HuggingfaceProvider(HUGGINGFACE_CONFIG)
    return provider

def test_hf_provider_init(hf_provider):
    """Test Hugging Face provider initialization."""
    assert hf_provider.name == 'huggingface_test'
    assert hf_provider.model == 'google/flan-t5-base'
    assert hf_provider.api_key == 'test_key'

def test_hf_provider_generate(hf_provider, mock_requests):
    """Test Hugging Face provider text generation."""
    result = hf_provider.generate(
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.7
    )
    
    assert 'response' in result
    assert result['response'] == 'This is a test response from Hugging Face'
    assert 'tokens' in result
    assert result['tokens']['total'] > 0
    
    # Verify API call
    mock_requests.assert_called_once()
    args, kwargs = mock_requests.call_args
    assert 'google/flan-t5-base' in args[0]
    assert kwargs['json']['inputs'] == 'Test prompt'
    assert kwargs['json']['parameters']['max_new_tokens'] == 100
    assert kwargs['json']['parameters']['temperature'] == 0.7

# Llama Provider Tests
@pytest.fixture
def llama_provider(mock_requests):
    provider = LlamaProvider(LLAMA_CONFIG)
    return provider

def test_llama_provider_init(llama_provider):
    """Test Llama provider initialization."""
    assert llama_provider.name == 'llama_test'
    assert llama_provider.endpoint == 'http://localhost:8080/completion'

def test_llama_provider_generate(llama_provider, mock_requests):
    """Test Llama provider text generation."""
    result = llama_provider.generate(
        prompt="Test prompt",
        max_tokens=100,
        temperature=0.7
    )
    
    assert 'response' in result
    assert result['response'] == 'This is a test response from Llama'
    assert result['tokens']['prompt'] == 5
    assert result['tokens']['completion'] == 10
    assert result['tokens']['total'] == 15
    
    # Verify API call
    mock_requests.assert_called_once()
    args, kwargs = mock_requests.call_args
    assert 'localhost:8080/completion' in args[0]
    assert kwargs['json']['prompt'] == 'Test prompt'
    assert kwargs['json']['n_predict'] == 100
    assert kwargs['json']['temperature'] == 0.7