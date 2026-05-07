"""Model routing for cost-optimized LLM selection with fallback logic."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from ai_service_kit.logging import Logger

from .config import get_settings
from .providers import create_provider
from .models import RouteDecision


class ModelRouter:
    """Routes requests to appropriate models based on complexity and cost optimization."""
    
    def __init__(self):
        self.settings = get_settings()
        self._cheap_provider = None
        self._expensive_provider = None
        self._fallback_providers: Dict[str, Any] = {}
        
        # Route decision cache for similar requests
        self._route_cache: Dict[str, RouteDecision] = {}
        
        Logger.info("Model router initialized")
    
    async def _ensure_providers(self):
        """Ensure providers are initialized."""
        if self._cheap_provider is None:
            try:
                cheap_config = self.settings.cheap_model_config()
                cheap_provider_name = cheap_config["provider"].strip().lower()
                cheap_provider_config = {
                    **self._get_provider_credentials(cheap_provider_name),
                    "model": cheap_config["model"],
                }
                self._cheap_provider = create_provider(
                    provider_type=cheap_provider_name,
                    config=cheap_provider_config,
                )
                Logger.debug(f"Initialized cheap model provider: {cheap_config['provider']}/{cheap_config['model']}")
            except Exception as e:
                Logger.error(f"Failed to initialize cheap provider: {e}")
                
        if self._expensive_provider is None:
            try:
                expensive_config = self.settings.expensive_model_config()
                expensive_provider_name = expensive_config["provider"].strip().lower()
                expensive_provider_config = {
                    **self._get_provider_credentials(expensive_provider_name),
                    "model": expensive_config["model"],
                }
                self._expensive_provider = create_provider(
                    provider_type=expensive_provider_name,
                    config=expensive_provider_config,
                )
                Logger.debug(f"Initialized expensive model provider: {expensive_config['provider']}/{expensive_config['model']}")
            except Exception as e:
                Logger.error(f"Failed to initialize expensive provider: {e}")
    
    async def route_request(
        self,
        prompt: str,
        complexity_hint: str | None = None,
        force_cheap: bool = False,
        force_expensive: bool = False,
        **generation_kwargs
    ) -> tuple[Any, RouteDecision]:
        """Route request to appropriate model and return provider + decision."""
        
        await self._ensure_providers()
        
        # Handle forced routing
        if force_expensive:
            decision = RouteDecision(
                selected_provider=self.settings.expensive_model_provider,
                selected_model=self.settings.expensive_model,
                reasoning="Expensive model forced by caller",
                route_type="expensive"
            )
            return self._expensive_provider, decision
        
        if force_cheap:
            decision = RouteDecision(
                selected_provider=self.settings.cheap_model_provider,
                selected_model=self.settings.cheap_model,
                reasoning="Cheap model forced by caller",
                route_type="cheap"
            )
            return self._cheap_provider, decision
        
        # Determine complexity and route accordingly
        complexity = await self._analyze_complexity(prompt, complexity_hint)
        
        if complexity == "high":
            provider = self._expensive_provider
            route_type = "expensive"
            selected_provider = self.settings.expensive_model_provider
            selected_model = self.settings.expensive_model
            reasoning = f"High complexity task requiring advanced reasoning: {complexity_hint or 'complex prompt structure'}"
        else:
            provider = self._cheap_provider
            route_type = "cheap"
            selected_provider = self.settings.cheap_model_provider
            selected_model = self.settings.cheap_model
            reasoning = f"Low/medium complexity task suitable for efficient model: {complexity_hint or 'simple prompt structure'}"
        
        decision = RouteDecision(
            selected_provider=selected_provider,
            selected_model=selected_model,
            reasoning=reasoning,
            route_type=route_type
        )
        
        Logger.debug(f"Routed to {route_type} model: {selected_provider}/{selected_model}")
        return provider, decision
    
    async def generate_with_fallback(
        self,
        prompt: str,
        complexity_hint: str | None = None,
        max_retries: int = 2,
        **generation_kwargs
    ) -> tuple[str, RouteDecision]:
        """Generate response with automatic fallback on failure."""
        
        start_time = time.time()
        last_error = None
        
        # Try primary route
        try:
            provider, decision = await self.route_request(prompt, complexity_hint, **generation_kwargs)
            
            if provider is None:
                raise RuntimeError(f"Provider not available: {decision.selected_provider}")
            
            response = await provider.generate(prompt=prompt, **generation_kwargs)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            Logger.debug(f"Generation successful with {decision.selected_provider}/{decision.selected_model}")
            return response_text, decision
            
        except Exception as e:
            Logger.warning(f"Primary model failed: {e}")
            last_error = e
        
        # Try fallback to the other model type
        try:
            fallback_force_expensive = decision.route_type == "cheap"  # If cheap failed, try expensive
            fallback_force_cheap = decision.route_type == "expensive"  # If expensive failed, try cheap
            
            fallback_provider, fallback_decision = await self.route_request(
                prompt=prompt,
                complexity_hint=complexity_hint,
                force_expensive=fallback_force_expensive,
                force_cheap=fallback_force_cheap,
                **generation_kwargs
            )
            
            if fallback_provider is None:
                raise RuntimeError(f"Fallback provider not available: {fallback_decision.selected_provider}")
            
            response = await fallback_provider.generate(prompt=prompt, **generation_kwargs)
            
            # Update decision to reflect fallback
            fallback_decision.fallback_used = True
            fallback_decision.reasoning += f" (fallback after {decision.selected_provider} failed)"
            fallback_decision.route_type = "fallback"
            
            Logger.info(f"Fallback successful with {fallback_decision.selected_provider}/{fallback_decision.selected_model}")
            return response, fallback_decision
            
        except Exception as e:
            Logger.error(f"Fallback model also failed: {e}")
            last_error = e
        
        # Try alternative providers if available
        for retry in range(max_retries):
            try:
                alternative_provider = await self._get_alternative_provider()
                if alternative_provider is None:
                    break
                
                response = await alternative_provider.generate(prompt=prompt, **generation_kwargs)
                response_text = response.content if hasattr(response, "content") else str(response)
                
                alt_decision = RouteDecision(
                    selected_provider="alternative",
                    selected_model="fallback",
                    reasoning=f"Alternative provider used after primary and fallback failed (retry {retry + 1})",
                    fallback_used=True,
                    route_type="fallback"
                )
                
                Logger.info(f"Alternative provider successful on retry {retry + 1}")
                return response_text, alt_decision
                
            except Exception as e:
                Logger.error(f"Alternative provider retry {retry + 1} failed: {e}")
                last_error = e
        
        # All options exhausted
        execution_time = int((time.time() - start_time) * 1000)
        Logger.error(f"All model routing options exhausted after {execution_time}ms")
        
        failed_decision = RouteDecision(
            selected_provider="none",
            selected_model="none",
            reasoning=f"All providers failed. Last error: {str(last_error)}",
            fallback_used=True,
            route_type="fallback"
        )
        
        raise RuntimeError(f"All model providers failed. Last error: {str(last_error)}")
    
    async def _analyze_complexity(self, prompt: str, hint: str | None = None) -> str:
        """Analyze prompt complexity to determine appropriate model."""
        
        # Use hint if provided
        if hint:
            hint_lower = hint.lower()
            if any(word in hint_lower for word in ["complex", "difficult", "advanced", "detailed", "analysis"]):
                return "high"
            elif any(word in hint_lower for word in ["simple", "basic", "easy", "quick"]):
                return "low"
        
        # Analyze prompt characteristics
        complexity_score = 0
        
        # Length factor
        if len(prompt) > 1000:
            complexity_score += 2
        elif len(prompt) > 500:
            complexity_score += 1
        
        # Complexity indicators
        high_complexity_indicators = [
            "analyze", "compare", "evaluate", "synthesize", "reasoning", "logic",
            "complex", "detailed", "comprehensive", "multi-step", "chain of thought",
            "reasoning", "explain why", "justify", "critique", "assess"
        ]
        
        for indicator in high_complexity_indicators:
            if indicator in prompt.lower():
                complexity_score += 1
        
        # Question complexity
        question_count = prompt.count('?')
        if question_count > 2:
            complexity_score += 1
        
        # Code or technical content
        if any(keyword in prompt.lower() for keyword in ["code", "programming", "algorithm", "technical", "implementation"]):
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    async def _get_alternative_provider(self) -> Any:
        """Get an alternative provider for fallback scenarios."""
        try:
            # In a real implementation, this would cycle through different providers
            # For now, we'll return the main provider as fallback
            if self._cheap_provider and self._expensive_provider:
                # Alternate between available providers
                return self._cheap_provider if hasattr(self, '_last_alternative') else self._expensive_provider
            return None
        except Exception as e:
            Logger.error(f"Failed to get alternative provider: {e}")
            return None
    
    def _get_provider_credentials(self, provider_type: str) -> Dict[str, Any]:
        """Get credentials for a specific provider type."""
        if provider_type.lower() == "openai":
            return {
                "api_key": self.settings.openai_api_key,
            }
        elif provider_type.lower() == "gemini":
            return {
                "api_key": self.settings.gemini_api_key,
            }
        elif provider_type.lower() == "claude":
            return {
                "api_key": self.settings.anthropic_api_key,
            }
        else:
            Logger.warning(f"Unknown provider type: {provider_type}")
            return {}
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        return {
            "cheap_provider_available": self._cheap_provider is not None,
            "expensive_provider_available": self._expensive_provider is not None,
            "cache_size": len(self._route_cache),
            "cheap_model_config": self.settings.cheap_model_config(),
            "expensive_model_config": self.settings.expensive_model_config(),
            "routing_enabled": True,
        }
    
    async def clear_cache(self):
        """Clear the routing decision cache."""
        self._route_cache.clear()
        Logger.info("Router cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all configured providers."""
        results = {}
        
        await self._ensure_providers()
        
        # Test cheap provider
        if self._cheap_provider:
            try:
                test_response = await asyncio.wait_for(
                    self._cheap_provider.generate("Test", max_tokens=1),
                    timeout=10.0
                )
                results["cheap_provider"] = {
                    "status": "healthy",
                    "provider": self.settings.cheap_model_provider,
                    "model": self.settings.cheap_model
                }
            except Exception as e:
                results["cheap_provider"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "provider": self.settings.cheap_model_provider,
                    "model": self.settings.cheap_model
                }
        else:
            results["cheap_provider"] = {
                "status": "not_initialized",
                "provider": self.settings.cheap_model_provider,
                "model": self.settings.cheap_model
            }
        
        # Test expensive provider
        if self._expensive_provider:
            try:
                test_response = await asyncio.wait_for(
                    self._expensive_provider.generate("Test", max_tokens=1),
                    timeout=10.0
                )
                results["expensive_provider"] = {
                    "status": "healthy",
                    "provider": self.settings.expensive_model_provider,
                    "model": self.settings.expensive_model
                }
            except Exception as e:
                results["expensive_provider"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "provider": self.settings.expensive_model_provider,
                    "model": self.settings.expensive_model
                }
        else:
            results["expensive_provider"] = {
                "status": "not_initialized",
                "provider": self.settings.expensive_model_provider,
                "model": self.settings.expensive_model
            }
        
        return results