"""
Ollama Client - Local LLM inference via Ollama.
Ollama must be running on the host (not in Docker).
  - Local dev: http://localhost:11434
  - Inside Docker container: http://host.docker.internal:11434
"""
import os
import logging
import requests
import json

from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def is_available(self) -> bool:
        """Check if Ollama model is configured and the server responds."""
        if not self.model:
            return False
        return self.check_health()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout_s: int = 300,
    ) -> str:
        """
        Generate text using the Ollama chat API.

        Args:
            prompt: The prompt text
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout_s: Request timeout in seconds

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama API timed out after {timeout_s}s")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            raise Exception(f"Ollama generate error: {e}")

    def generate_fast(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        timeout_s: int = 60,
    ) -> str:
        """Generate using smaller token budget for auxiliary tasks (query expansion, grading, rewrite)."""
        return self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    def check_health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
