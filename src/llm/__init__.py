"""
LLM module for RAG answer generation.

Provides integration with Ollama for local LLM inference.
"""

from .ollama_client import OllamaClient

__all__ = ["OllamaClient"]
