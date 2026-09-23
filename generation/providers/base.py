from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for all Universal-RAG language-model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from a prompt."""

    def list_models(self) -> list[str]:
        """Return models available from this provider."""
        return []

    def validate_connection(self) -> bool:
        """Return True when the provider is reachable/configured."""
        return True

    def close(self) -> None:
        """Release provider resources."""
        return None
