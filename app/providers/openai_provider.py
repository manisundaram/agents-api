"""OpenAI LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional
from .base import BaseLLMProvider, LLMError, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = self.config.get("api_key")
        if not self._api_key:
            raise LLMError("OpenAI API key is required")

        self._base_url = self.config.get("base_url")
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> List[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url
                )
            except ImportError:
                raise LLMError("OpenAI package not installed. Run: pip install openai")
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        try:
            client = self._get_client()
            
            # Use default model if not specified
            if model is None:
                model = "gpt-4o-mini"
            
            # Validate model
            if model not in self.supported_models:
                raise LLMError(f"Unsupported model: {model}")

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            choice = response.choices[0]
            usage_info = response.usage
            
            return LLMResponse(
                content=choice.message.content,
                usage={
                    "prompt_tokens": usage_info.prompt_tokens if usage_info else 0,
                    "completion_tokens": usage_info.completion_tokens if usage_info else 0,
                    "total_tokens": usage_info.total_tokens if usage_info else 0,
                } if usage_info else None,
                model=response.model,
                finish_reason=choice.finish_reason,
                metadata={
                    "response_id": response.id,
                    "created": response.created,
                }
            )

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"OpenAI generation failed: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        try:
            # Simple test with minimal tokens
            test_messages = [{"role": "user", "content": "Hi"}]
            response = await self.generate(
                messages=test_messages,
                model="gpt-4o-mini",
                max_tokens=5
            )
            return bool(response.content)
        except Exception:
            return False