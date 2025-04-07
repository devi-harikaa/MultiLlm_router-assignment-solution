import time
import os
import requests
from typing import Dict, Any
from services.llm_provider import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

class GroqProvider(LLMProvider):
    """Provider for Groq OpenAI-compatible API."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = config.get("model", "llama-3.1-8b-instant")
        self.api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Missing Groq API key in config or environment variable GROQ_API_KEY")

        self.api_url = config.get("endpoint", "https://api.groq.com/openai/v1/chat/completions")
        self.retry_count = config.get("retry_count", 3)
        self.timeout = config.get("timeout", 10)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        retries = 0
        last_error = None
        start_time = time.time()

        while retries <= self.retry_count:
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                message = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                duration = time.time() - start_time

                return {
                    "response": message,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens
                    },
                    "time": duration
                }

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Groq API request failed (attempt {retries + 1}/{self.retry_count + 1}): {last_error}")
                retries += 1
                if retries <= self.retry_count:
                    time.sleep(2 ** retries * 0.5)

        raise Exception(f"Groq provider failed after {self.retry_count + 1} attempts: {last_error}")
    def count_tokens(self, prompt: str) -> int:
        """
        Estimate the number of tokens in the prompt.
        This is a rough estimate assuming 1 token ≈ 4 characters.
        """
        return len(prompt) // 4
