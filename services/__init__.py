from services.llm_provider import LLMProvider
from services.providers.huggingface_provider import logger


import requests


import time
from typing import Any, Dict


class HuggingfaceProvider(LLMProvider):
    """Provider for Hugging Face Inference API with accurate token count."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = config.get('model', 'google/flan-t5-base')
        self.api_key = config.get('api_key', '')

        if not self.api_key:
            raise ValueError("Missing Hugging Face API key in config.")

        self.retry_count = config.get('retry_count', 3)
        self.timeout = config.get('timeout', 10)

        # Load tokenizer for accurate token count
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for model {self.model}. Falling back to estimate. Error: {e}")
            self.tokenizer = None

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        prompt_tokens = self.count_tokens(prompt)

        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0
            }
        }

        retries = 0
        last_error = None

        while retries <= self.retry_count:
            try:
                response = requests.post(api_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                # Extract response text
                if isinstance(result, list) and "generated_text" in result[0]:
                    completion_text = result[0]["generated_text"]
                elif isinstance(result, dict) and "generated_text" in result:
                    completion_text = result["generated_text"]
                else:
                    completion_text = str(result)

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
                logger.warning(f"Hugging Face API request failed (attempt {retries + 1}/{self.retry_count + 1}): {last_error}")
                retries += 1

                if retries <= self.retry_count:
                    time.sleep(2 ** retries * 0.5)
                    if "is currently loading" in last_error.lower():
                        time.sleep(5)

        raise Exception(f"Hugging Face provider failed after {self.retry_count + 1} attempts: {last_error}")

    def count_tokens(self, text: str) -> int:
        """Count tokens using actual model tokenizer."""
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            else:
                # Fallback estimate
                return max(len(text.split()) * 3 // 4, 1)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return 1