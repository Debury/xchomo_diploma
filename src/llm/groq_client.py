"""
Groq API Client - Fast LLM inference
Free tier: 30 requests/min, 6000 tokens/min
Get API key at: https://console.groq.com
"""
import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class GroqClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        # Fast models on Groq (in order of speed):
        # - llama-3.1-8b-instant (fastest, good quality)
        # - llama-3.1-70b-versatile (slower, best quality)
        # - mixtral-8x7b-32768 (good balance)
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    def is_available(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        timeout_s: int = 30,
    ) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout_s: Request timeout in seconds
            
        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
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
                json=payload,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Groq API timed out after {timeout_s}s")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = resp.json().get("error", {}).get("message", "")
            except:
                pass
            raise Exception(f"Groq API error: {e} - {error_detail}")
        except Exception as e:
            raise Exception(f"Groq generate error: {e}")
    
    def check_health(self) -> bool:
        """Check if Groq API is accessible."""
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
