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

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "Climate RAG")
        
        # Model options (in order of recommendation for RAG):
        # Free/cheap models:
        # - meta-llama/llama-3.1-8b-instruct:free (FREE, good quality)
        # - google/gemma-2-9b-it:free (FREE, good for structured data)
        # - mistralai/mistral-7b-instruct:free (FREE, fast)
        # Premium models:
        # - openai/gpt-4o-mini ($0.15/1M input) - best value
        # - anthropic/claude-3-haiku ($0.25/1M input) - very good
        # - openai/gpt-4o ($2.50/1M input) - best quality
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    
    def is_available(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return bool(self.api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        timeout_s: int = 30,
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
            except:
                pass
            raise Exception(f"OpenRouter API error: {e} - {error_detail}")
        except Exception as e:
            raise Exception(f"OpenRouter generate error: {e}")
    
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
