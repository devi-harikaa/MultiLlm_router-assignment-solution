import time
import requests
from typing import Dict, Any

from services.llm_provider import LLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)

class LlamaProvider(LLMProvider):
    """Provider for local Ollama server (e.g. llama2, mistral)."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.endpoint = config.get('endpoint', 'http://localhost:11434/api/generate')
        self.model = config.get('model', 'llama2')

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        prompt_tokens = self.count_tokens(prompt)

        retries = 0
        last_error = None

        while retries <= self.retry_count:
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                completion_text = result.get('response', '')

                # Estimate tokens since Ollama doesn't return token counts
                completion_tokens = self.count_tokens(completion_text)
                total_tokens = prompt_tokens + completion_tokens

                return {
                    "response": completion_text,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens
                    }
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Ollama request failed (attempt {retries+1}/{self.retry_count+1}): {last_error}")
                retries += 1
                time.sleep(2 ** retries * 0.5)

        raise Exception(f"Ollama provider failed after {self.retry_count+1} attempts: {last_error}")

    def count_tokens(self, text: str) -> int:
        words = text.split()
        return len(words) * 4 // 3 or 1
