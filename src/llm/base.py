"""Abstract base class for LLM API clients."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for all LLM clients (Ollama, Groq, OpenRouter)."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
        timeout_s: int = 120,
    ) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def check_health(self) -> bool:
        """Check if the LLM service is accessible."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is configured and ready."""
        ...
