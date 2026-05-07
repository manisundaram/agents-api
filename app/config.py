from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ai_service_kit.utils import mask_secret
from ai_service_kit.logging import Logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = Field(default="agents-api", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")
    
    # Enhanced logging configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    file_log_level: str = Field(default="DEBUG", alias="FILE_LOG_LEVEL")
    error_file_level: str = Field(default="ERROR", alias="ERROR_FILE_LEVEL")
    log_structured: bool = Field(default=True, alias="LOG_STRUCTURED")
    log_console: bool = Field(default=True, alias="LOG_CONSOLE")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    log_max_file_size: int = Field(default=10485760, alias="LOG_MAX_FILE_SIZE")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")
    cloud_logging_providers: list[str] = Field(default_factory=list, alias="CLOUD_LOGGING_PROVIDERS")
    
    # Cloud logging - AWS
    aws_logging_enabled: bool = Field(default=False, alias="AWS_LOGGING_ENABLED")
    aws_logging_level: str = Field(default="ERROR", alias="AWS_LOGGING_LEVEL")
    aws_log_group: str = Field(default="/agents-api/production", alias="AWS_LOG_GROUP")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    
    # Cloud logging - Datadog
    datadog_logging_enabled: bool = Field(default=False, alias="DATADOG_LOGGING_ENABLED")
    datadog_logging_level: str = Field(default="INFO", alias="DATADOG_LOGGING_LEVEL")
    datadog_api_key: str | None = Field(default=None, alias="DATADOG_API_KEY")
    
    # Provider configuration
    provider_type: str = Field(default="openai", alias="PROVIDER_TYPE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(default="models/embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-5-haiku-latest", alias="CLAUDE_MODEL")
    
    # Agent configuration
    agent_max_steps: int = Field(default=10, alias="AGENT_MAX_STEPS")
    agent_temperature: float = Field(default=0.1, alias="AGENT_TEMPERATURE")
    agent_timeout_seconds: int = Field(default=300, alias="AGENT_TIMEOUT_SECONDS")
    
    # Model routing configuration
    cheap_model_provider: str = Field(default="openai", alias="CHEAP_MODEL_PROVIDER")
    cheap_model: str = Field(default="gpt-4o-mini", alias="CHEAP_MODEL")
    expensive_model_provider: str = Field(default="openai", alias="EXPENSIVE_MODEL_PROVIDER") 
    expensive_model: str = Field(default="gpt-4o", alias="EXPENSIVE_MODEL")
    
    # External service URLs
    semantic_search_api_url: str = Field(default="http://localhost:8001", alias="SEMANTIC_SEARCH_API_URL")
    rag_api_url: str = Field(default="http://localhost:8002", alias="RAG_API_URL")
    
    # Semantic cache configuration
    semantic_cache_enabled: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")
    semantic_cache_threshold: float = Field(default=0.85, alias="SEMANTIC_CACHE_THRESHOLD")
    semantic_cache_max_entries: int = Field(default=1000, alias="SEMANTIC_CACHE_MAX_ENTRIES")
    
    # Vector store configuration
    vectorstore_backend: str = Field(default="chroma", alias="VECTORSTORE_BACKEND")
    default_collection_name: str = Field(default="agents_cache", alias="DEFAULT_COLLECTION_NAME")
    
    # CORS configuration
    enable_cors: bool = Field(default=True, alias="ENABLE_CORS")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item.strip() for item in value if item.strip()]
        return [item.strip() for item in value.split(",") if item.strip()]
    
    @field_validator("cloud_logging_providers", mode="before")
    @classmethod
    def parse_cloud_providers(cls, value: str | list[str] | None) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [item.strip().lower() for item in value if item.strip()]
        return [item.strip().lower() for item in value.split(",") if item.strip()]

    def provider_config(self) -> dict[str, Any]:
        provider_name = self.provider_type.strip().lower()
        try:
            if provider_name == "openai":
                return {
                    "api_key": self.openai_api_key,
                    "model": self.openai_model,
                    "embedding_model": self.openai_embedding_model,
                }
            if provider_name == "gemini":
                return {
                    "api_key": self.gemini_api_key,
                    "model": self.gemini_model,
                    "embedding_model": self.gemini_embedding_model,
                }
            if provider_name == "claude":
                return {
                    "api_key": self.anthropic_api_key,
                    "model": self.claude_model,
                }
            Logger.warning(f"Unknown provider type: {provider_name}")
            return {}
        except Exception as e:
            Logger.error(f"Failed to build provider config for {provider_name}: {e}", exc_info=True)
            return {}

    def selected_provider_model(self) -> str | None:
        return self.provider_config().get("model")

    def selected_provider_api_key(self) -> str | None:
        return self.provider_config().get("api_key")
        
    def selected_embedding_model(self) -> str | None:
        return self.provider_config().get("embedding_model")

    def cheap_model_config(self) -> dict[str, Any]:
        """Get configuration for the cheap model used in routing."""
        return {
            "provider": self.cheap_model_provider,
            "model": self.cheap_model,
        }

    def expensive_model_config(self) -> dict[str, Any]:
        """Get configuration for the expensive model used in routing."""
        return {
            "provider": self.expensive_model_provider,
            "model": self.expensive_model,
        }

    def operational_settings(self) -> dict[str, Any]:
        return {
            "app_env": self.app_env,
            "app_debug": self.app_debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "mock_mode": self.mock_mode,
            "provider_type": self.provider_type,
            "provider_model": self.selected_provider_model(),
            "embedding_model": self.selected_embedding_model(),
            "vectorstore_backend": self.vectorstore_backend,
            "default_collection_name": self.default_collection_name,
            "enable_cors": self.enable_cors,
            "cors_origins": list(self.cors_origins),
            "agent_max_steps": self.agent_max_steps,
            "agent_temperature": self.agent_temperature,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "semantic_search_api_url": self.semantic_search_api_url,
            "rag_api_url": self.rag_api_url,
        }

    def masked_secrets(self) -> dict[str, str | None]:
        return {
            "openai_api_key": mask_secret(self.openai_api_key),
            "gemini_api_key": mask_secret(self.gemini_api_key),
            "anthropic_api_key": mask_secret(self.anthropic_api_key),
            "datadog_api_key": mask_secret(self.datadog_api_key),
        }

    def masked_debug_config(self) -> Mapping[str, Any]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env,
            "app_debug": self.app_debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "file_log_level": self.file_log_level,
            "error_file_level": self.error_file_level,
            "log_structured": self.log_structured,
            "log_console": self.log_console,
            "log_dir": self.log_dir,
            "cloud_logging_providers": self.cloud_logging_providers,
            "mock_mode": self.mock_mode,
            "provider_type": self.provider_type,
            "openai_api_key": self.masked_secrets()["openai_api_key"],
            "openai_model": self.openai_model,
            "openai_embedding_model": self.openai_embedding_model,
            "gemini_api_key": self.masked_secrets()["gemini_api_key"],
            "gemini_model": self.gemini_model,
            "gemini_embedding_model": self.gemini_embedding_model,
            "anthropic_api_key": self.masked_secrets()["anthropic_api_key"],
            "claude_model": self.claude_model,
            "datadog_api_key": self.masked_secrets()["datadog_api_key"],
            "vectorstore_backend": self.vectorstore_backend,
            "default_collection_name": self.default_collection_name,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origins,
            "agent_max_steps": self.agent_max_steps,
            "agent_temperature": self.agent_temperature,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "cheap_model": self.cheap_model,
            "expensive_model": self.expensive_model,
            "semantic_search_api_url": self.semantic_search_api_url,
            "rag_api_url": self.rag_api_url,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "semantic_cache_threshold": self.semantic_cache_threshold,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        settings = Settings()
        # Note: Logger may not be configured yet during initial setup
        return settings
    except Exception as e:
        # Use print since Logger might not be configured yet
        print(f"Failed to load settings: {e}")
        raise