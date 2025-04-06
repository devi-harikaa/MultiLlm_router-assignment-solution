import time
import os
import requests
import tiktoken
from typing import Dict, Any
from services.llm_provider import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenaiProvider(LLMProvider):
    """Provider for OpenAI API."""

    def __init__(self, config: Dict):
        """Initialize OpenAI provider with configuration."""
        super().__init__(config)

        self.model = config.get('model', 'gpt-3.5-turbo')
        self.api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("Missing required environment variable: OPENAI_API_KEY")

        # Load tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            logger.warning(f"tiktoken encoding not found for model '{self.model}', falling back to 'cl100k_base'")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.retry_count = config.get('retry_count', 3)
        self.timeout = config.get('timeout', 10)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate text using OpenAI's Chat Completion API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        api_url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        prompt_tokens = self.count_tokens(prompt)
        retries = 0
        last_error = None

        while retries <= self.retry_count:
            try:
                response = requests.post(api_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                choices = result.get("choices", [])
                if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
                    raise ValueError("Unexpected response format from OpenAI API")

                completion_text = choices[0]["message"]["content"]

                # Use usage info if available, else estimate
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", self.count_tokens(completion_text))
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                return {
                    "response": completion_text,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens
                    }
                }

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"OpenAI API request failed (attempt {retries + 1}/{self.retry_count + 1}): {last_error}")
                retries += 1
                if retries <= self.retry_count:
                    time.sleep(2 ** retries * 0.5)  # Exponential backoff

        raise Exception(f"OpenAI provider failed after {self.retry_count + 1} attempts: {last_error}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a given text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split()) * 4 // 3  # Fallback estimate
