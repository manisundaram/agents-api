"""LLM provider factory."""

from typing import Any, Dict, Optional
from .base import BaseLLMProvider, LLMError
from .openai_provider import OpenAIProvider


PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
}


def create_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseLLMProvider:
    """Create an LLM provider instance.
    
    Args:
        provider_type: The type of provider (e.g., 'openai', 'anthropic', 'gemini')
        config: Configuration dictionary for the provider
    
    Returns:
        An instance of the specified provider
        
    Raises:
        LLMError: If the provider type is not supported or configuration is invalid
    """
    provider_type = provider_type.lower().strip()
    
    if provider_type not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise LLMError(f"Unsupported provider type: {provider_type}. Available: {available}")
    
    provider_class = PROVIDER_REGISTRY[provider_type]
    try:
        return provider_class(config)
    except Exception as e:
        raise LLMError(f"Failed to initialize {provider_type} provider: {str(e)}") from e


def get_available_providers() -> list[str]:
    """Get list of available provider types."""
    return list(PROVIDER_REGISTRY.keys())


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """Register a new provider type."""
    PROVIDER_REGISTRY[name] = provider_class