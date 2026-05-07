from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_service_kit.health import (
    BaseHealthCheck,
    CheckResult,
    ComponentKind,
    ComponentStatus,
    HealthStatus,
    NoOpMetricsCollector,
    ProviderDiagnosticsResult,
    ServiceContext,
    VectorStoreDiagnosticsResult,
)
from ai_service_kit.logging import Logger

from .config import Settings

SUPPORTED_PROVIDERS = ("openai", "gemini", "claude")
SUPPORTED_VECTORSTORES = ("chroma",)


@dataclass(slots=True)
class ProviderRuntime:
    name: str
    model: str | None
    embedding_model: str | None
    configured: bool
    supported: bool
    initialized: bool


@dataclass(slots=True)
class VectorStoreRuntime:
    backend: str
    default_collection_name: str
    configured: bool
    supported: bool
    initialized: bool
    collections_count: int


@dataclass(slots=True)
class AgentsRuntime:
    agent_max_steps: int
    semantic_cache_enabled: bool
    cheap_model_provider: str
    expensive_model_provider: str
    external_services_configured: bool
    tools_available: int


class ConfigurationHealthCheck(BaseHealthCheck):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "configuration"

    async def run(self) -> CheckResult:
        try:
            Logger.debug("Running configuration health check")
            provider_name = self._settings.provider_type.strip().lower()
            vectorstore_backend = self._settings.vectorstore_backend.strip().lower()
            errors: list[str] = []

            if provider_name not in SUPPORTED_PROVIDERS:
                error_msg = f"Unsupported provider: {provider_name}"
                Logger.warning(error_msg)
                errors.append(error_msg)
            elif not self._settings.selected_provider_api_key():
                error_msg = f"Missing API key for provider: {provider_name}"
                Logger.warning(error_msg)
                errors.append(error_msg)

            if vectorstore_backend not in SUPPORTED_VECTORSTORES:
                error_msg = f"Unsupported vector store backend: {vectorstore_backend}"
                Logger.warning(error_msg)
                errors.append(error_msg)

            # Check agent-specific configuration
            if self._settings.agent_max_steps <= 0:
                error_msg = "Invalid agent max steps configuration"
                Logger.warning(error_msg)
                errors.append(error_msg)
                
            if self._settings.semantic_cache_threshold <= 0 or self._settings.semantic_cache_threshold > 1:
                error_msg = "Invalid semantic cache threshold"
                Logger.warning(error_msg)
                errors.append(error_msg)

            if not errors:
                status = HealthStatus.HEALTHY
                summary = "Operational configuration is valid"
                Logger.debug("Configuration health check passed")
            elif any(error.startswith("Unsupported") for error in errors):
                status = HealthStatus.CRITICAL
                summary = "Operational configuration contains unsupported components"
                Logger.error(f"Configuration health check critical: {summary}")
            else:
                status = HealthStatus.DEGRADED
                summary = "Operational configuration is incomplete"
                Logger.warning(f"Configuration health check degraded: {summary}")

            return CheckResult(
                name=self.name,
                status=status,
                summary=summary,
                details={
                    "provider_type": provider_name,
                    "available_providers": list(SUPPORTED_PROVIDERS),
                    "vectorstore_backend": vectorstore_backend,
                    "available_vectorstores": list(SUPPORTED_VECTORSTORES),
                    "agent_max_steps": self._settings.agent_max_steps,
                    "semantic_cache_enabled": self._settings.semantic_cache_enabled,
                    "cheap_model": self._settings.cheap_model,
                    "expensive_model": self._settings.expensive_model,
                    "external_services": {
                        "semantic_search_api": self._settings.semantic_search_api_url,
                        "rag_api": self._settings.rag_api_url
                    }
                },
                errors=tuple(errors),
            )
        except Exception as e:
            Logger.error(f"Configuration health check failed: {e}", exc_info=True)
            return CheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                summary="Configuration health check failed",
                details={},
                errors=(str(e),),
            )


class ExternalServicesHealthCheck(BaseHealthCheck):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "external_services"

    async def run(self) -> CheckResult:
        try:
            Logger.debug("Running external services health check")
            errors: list[str] = []
            services_status = {}
            
            # Check semantic search API
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self._settings.semantic_search_api_url}/ping")
                    if response.status_code == 200:
                        services_status["semantic_search"] = "healthy"
                    else:
                        services_status["semantic_search"] = f"unhealthy (status: {response.status_code})"
                        errors.append(f"Semantic search API unhealthy: {response.status_code}")
            except Exception as e:
                services_status["semantic_search"] = f"unreachable ({str(e)[:50]})"
                errors.append(f"Semantic search API unreachable: {str(e)[:100]}")

            # Check RAG API
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self._settings.rag_api_url}/ping")
                    if response.status_code == 200:
                        services_status["rag_api"] = "healthy"
                    else:
                        services_status["rag_api"] = f"unhealthy (status: {response.status_code})"
                        errors.append(f"RAG API unhealthy: {response.status_code}")
            except Exception as e:
                services_status["rag_api"] = f"unreachable ({str(e)[:50]})"
                errors.append(f"RAG API unreachable: {str(e)[:100]}")

            if not errors:
                status = HealthStatus.HEALTHY
                summary = "All external services are healthy"
                Logger.debug("External services health check passed")
            elif len(errors) == len(services_status):
                status = HealthStatus.CRITICAL
                summary = "All external services are unavailable"
                Logger.error(f"External services health check critical: {summary}")
            else:
                status = HealthStatus.DEGRADED
                summary = f"{len(errors)} external services are unavailable"
                Logger.warning(f"External services health check degraded: {summary}")

            return CheckResult(
                name=self.name,
                status=status,
                summary=summary,
                details=services_status,
                errors=tuple(errors),
            )
        except Exception as e:
            Logger.error(f"External services health check failed: {e}", exc_info=True)
            return CheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                summary="External services health check failed",
                details={},
                errors=(str(e),),
            )


def _build_provider_runtime(settings: Settings) -> ProviderRuntime:
    try:
        provider_name = settings.provider_type.strip().lower()
        supported = provider_name in SUPPORTED_PROVIDERS
        configured = supported and bool(settings.selected_provider_api_key())
        initialized = configured
        
        Logger.debug(f"Built provider runtime: {provider_name} (configured: {configured}, supported: {supported})")
        
        return ProviderRuntime(
            name=provider_name,
            model=settings.selected_provider_model(),
            embedding_model=settings.selected_embedding_model(),
            configured=configured,
            supported=supported,
            initialized=initialized,
        )
    except Exception as e:
        Logger.error(f"Failed to build provider runtime: {e}", exc_info=True)
        return ProviderRuntime(
            name="unknown",
            model=None,
            embedding_model=None,
            configured=False,
            supported=False,
            initialized=False,
        )


def _build_vectorstore_runtime(settings: Settings) -> VectorStoreRuntime:
    try:
        backend = settings.vectorstore_backend.strip().lower()
        supported = backend in SUPPORTED_VECTORSTORES
        configured = supported and bool(settings.default_collection_name)
        initialized = configured
        
        Logger.debug(f"Built vectorstore runtime: {backend} (configured: {configured}, supported: {supported})")
        
        return VectorStoreRuntime(
            backend=backend,
            default_collection_name=settings.default_collection_name,
            configured=configured,
            supported=supported,
            initialized=initialized,
            collections_count=1 if configured else 0,
        )
    except Exception as e:
        Logger.error(f"Failed to build vectorstore runtime: {e}", exc_info=True)
        return VectorStoreRuntime(
            backend="unknown",
            default_collection_name="",
            configured=False,
            supported=False,
            initialized=False,
            collections_count=0,
        )


def _build_agents_runtime(settings: Settings) -> AgentsRuntime:
    try:
        external_services_configured = (
            bool(settings.semantic_search_api_url) and 
            bool(settings.rag_api_url)
        )
        
        # Count available tools
        from .tools import AVAILABLE_TOOLS
        tools_available = len(AVAILABLE_TOOLS)
        
        Logger.debug(f"Built agents runtime: external_services={external_services_configured}, tools={tools_available}")
        
        return AgentsRuntime(
            agent_max_steps=settings.agent_max_steps,
            semantic_cache_enabled=settings.semantic_cache_enabled,
            cheap_model_provider=settings.cheap_model_provider,
            expensive_model_provider=settings.expensive_model_provider,
            external_services_configured=external_services_configured,
            tools_available=tools_available,
        )
    except Exception as e:
        Logger.error(f"Failed to build agents runtime: {e}", exc_info=True)
        return AgentsRuntime(
            agent_max_steps=10,
            semantic_cache_enabled=False,
            cheap_model_provider="unknown",
            expensive_model_provider="unknown",
            external_services_configured=False,
            tools_available=0,
        )


def _provider_status(provider_runtime: ProviderRuntime) -> ComponentStatus:
    if not provider_runtime.supported:
        status = HealthStatus.CRITICAL
        error = f"Unsupported provider: {provider_runtime.name}"
    elif not provider_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Missing credentials for provider: {provider_runtime.name}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=provider_runtime.name,
        kind=ComponentKind.PROVIDER,
        status=status,
        configured=provider_runtime.configured,
        available=provider_runtime.supported,
        initialized=provider_runtime.initialized,
        details={
            "model": provider_runtime.model,
            "embedding_model": provider_runtime.embedding_model
        },
        error=error,
    )


def _vectorstore_status(vectorstore_runtime: VectorStoreRuntime) -> ComponentStatus:
    if not vectorstore_runtime.supported:
        status = HealthStatus.CRITICAL
        error = f"Unsupported vector store backend: {vectorstore_runtime.backend}"
    elif not vectorstore_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Vector store backend is missing required configuration: {vectorstore_runtime.backend}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=vectorstore_runtime.backend,
        kind=ComponentKind.VECTORSTORE,
        status=status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
        error=error,
    )


def _provider_diagnostics(provider_runtime: ProviderRuntime) -> ProviderDiagnosticsResult:
    provider_status = _provider_status(provider_runtime)
    return ProviderDiagnosticsResult(
        provider=provider_runtime.name,
        status=provider_status.status,
        configured=provider_runtime.configured,
        available=provider_runtime.supported,
        initialized=provider_runtime.initialized,
        models_available=(provider_runtime.model,) if provider_runtime.model else (),
        error=provider_status.error,
        details={
            "model": provider_runtime.model,
            "embedding_model": provider_runtime.embedding_model
        },
    )


def _vectorstore_diagnostics(vectorstore_runtime: VectorStoreRuntime) -> VectorStoreDiagnosticsResult:
    vectorstore_status = _vectorstore_status(vectorstore_runtime)
    return VectorStoreDiagnosticsResult(
        backend=vectorstore_runtime.backend,
        status=vectorstore_status.status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        collections_count=vectorstore_runtime.collections_count,
        default_collection=vectorstore_runtime.default_collection_name,
        error=vectorstore_status.error,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
    )


def debug_snapshot(context: ServiceContext) -> dict[str, Any]:
    configuration = context.configuration()
    return {
        "service_name": context.service_name,
        "service_version": context.service_version,
        "configuration": asdict(configuration),
        "metrics_collector": type(context.metrics_collector).__name__,
        "health_checks": [getattr(check, "name", type(check).__name__) for check in context.health_checks],
        "diagnostics_checks": [getattr(check, "name", type(check).__name__) for check in context.diagnostics_checks],
    }


def build_service_context(settings: Settings) -> ServiceContext:
    try:
        Logger.info(f"Building service context for {settings.app_name} v{settings.app_version}")
        
        provider_runtime = _build_provider_runtime(settings)
        vectorstore_runtime = _build_vectorstore_runtime(settings)
        agents_runtime = _build_agents_runtime(settings)
        metrics_collector = NoOpMetricsCollector()
        configuration_check = ConfigurationHealthCheck(settings)
        external_services_check = ExternalServicesHealthCheck(settings)

        async def provider_status_resolver() -> tuple[ComponentStatus, ...]:
            try:
                result = (_provider_status(provider_runtime),)
                Logger.debug("Provider status resolved successfully")
                return result
            except Exception as e:
                Logger.error(f"Provider status resolution failed: {e}", exc_info=True)
                raise

        async def vectorstore_status_resolver() -> tuple[ComponentStatus, ...]:
            try:
                result = (_vectorstore_status(vectorstore_runtime),)
                Logger.debug("Vectorstore status resolved successfully")
                return result
            except Exception as e:
                Logger.error(f"Vectorstore status resolution failed: {e}", exc_info=True)
                raise

        async def provider_diagnostics_resolver() -> tuple[ProviderDiagnosticsResult, ...]:
            try:
                result = (_provider_diagnostics(provider_runtime),)
                Logger.debug("Provider diagnostics resolved successfully")
                return result
            except Exception as e:
                Logger.error(f"Provider diagnostics resolution failed: {e}", exc_info=True)
                raise

        async def vectorstore_diagnostics_resolver() -> tuple[VectorStoreDiagnosticsResult, ...]:
            try:
                result = (_vectorstore_diagnostics(vectorstore_runtime),)
                Logger.debug("Vectorstore diagnostics resolved successfully")
                return result
            except Exception as e:
                Logger.error(f"Vectorstore diagnostics resolution failed: {e}", exc_info=True)
                raise

        async def benchmarks_resolver() -> dict[str, Any]:
            try:
                result = {
                    "bootstrap": {
                        "provider_initialized": provider_runtime.initialized,
                        "vectorstore_initialized": vectorstore_runtime.initialized,
                    },
                    "agents": {
                        "max_steps": agents_runtime.agent_max_steps,
                        "semantic_cache_enabled": agents_runtime.semantic_cache_enabled,
                        "tools_available": agents_runtime.tools_available,
                        "external_services_configured": agents_runtime.external_services_configured,
                    }
                }
                Logger.debug("Benchmarks resolved successfully")
                return result
            except Exception as e:
                Logger.error(f"Benchmarks resolution failed: {e}", exc_info=True)
                raise

        context = ServiceContext(
            service_name=settings.app_name,
            service_version=settings.app_version,
            provider=provider_runtime.name,
            available_providers=SUPPORTED_PROVIDERS,
            vectorstore=vectorstore_runtime.backend,
            available_vectorstores=SUPPORTED_VECTORSTORES,
            mock_mode=settings.mock_mode,
            debug_mode=settings.app_debug,
            cors_enabled=settings.enable_cors,
            masked_secrets=settings.masked_secrets(),
            settings=settings.operational_settings(),
            metrics_collector=metrics_collector,
            health_checks=(configuration_check, external_services_check),
            diagnostics_checks=(configuration_check, external_services_check),
            provider_status_resolver=provider_status_resolver,
            vectorstore_status_resolver=vectorstore_status_resolver,
            provider_diagnostics_resolver=provider_diagnostics_resolver,
            vectorstore_diagnostics_resolver=vectorstore_diagnostics_resolver,
            benchmarks_resolver=benchmarks_resolver,
        )
        
        Logger.info(f"Service context built successfully for provider: {provider_runtime.name}, vectorstore: {vectorstore_runtime.backend}, agents: {agents_runtime.tools_available} tools")
        return context
        
    except Exception as e:
        Logger.error(f"Failed to build service context: {e}", exc_info=True)
        raise