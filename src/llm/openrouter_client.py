"""
OpenRouter API Client - Access to multiple LLM providers
Get API key at: https://openrouter.ai/keys
Free models available, or pay-as-you-go for premium models.
"""
import os
import logging
import requests
import json
from typing import Optional

from src.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "Climate RAG")
        
        # Default model matches what the rest of the project (README, DEMO,
        # eval logs) advertises. Override with OPENROUTER_MODEL in .env.
        self.model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
        self.fast_model = os.getenv("OPENROUTER_FAST_MODEL", "anthropic/claude-sonnet-4.6")
    
    def is_available(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return bool(self.api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout_s: int = 300,
    ) -> str:
        """
        Generate text using OpenRouter API.
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout_s: Request timeout in seconds
            
        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Optional headers for OpenRouter rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_s,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"OpenRouter API timed out after {timeout_s}s")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            raise Exception(f"OpenRouter API error: {e} - {error_detail}")
        except Exception as e:
            raise Exception(f"OpenRouter generate error: {e}")
    
    def generate_fast(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        timeout_s: int = 30,
    ) -> str:
        """Generate using the fast model (Sonnet) for auxiliary tasks like query expansion."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.fast_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_s,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"OpenRouter fast generate error: {e}")

    def check_health(self) -> bool:
        """Check if OpenRouter API is accessible."""
        if not self.api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
