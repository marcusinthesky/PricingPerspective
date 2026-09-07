"""Typed environment and dotenv settings for pipeline runtime boundaries."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from pydantic import Field, PositiveInt, SecretStr, field_validator  # noqa: TC002
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTINGS_CONFIG = SettingsConfigDict(
    case_sensitive=True,
    dotenv_filtering="only_existing",
    env_file=".env",
    env_file_encoding="utf-8",
    env_ignore_empty=True,
    extra="ignore",
    frozen=True,
)


class OpenRouterSettings(BaseSettings):
    """Authentication settings for paid OpenRouter embedding requests."""

    model_config = _SETTINGS_CONFIG

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENROUTER_API_KEY",
    )


class PipelineRuntimeSettings(BaseSettings):
    """Non-secret process controls shared by numerical pipeline stages."""

    model_config = _SETTINGS_CONFIG

    jax_cache_dir: Path = Field(
        default=Path(".jax_cache"),
        validation_alias="PP_JAX_CACHE_DIR",
    )
    blas_limit: PositiveInt | None = Field(
        default=2,
        validation_alias="PP_BLAS_LIMIT",
    )

    @field_validator("blas_limit", mode="before")
    @classmethod
    def _zero_disables_blas_limit(cls, value: object) -> object:
        """Interpret an empty or zero limit as an explicit opt-out."""
        if isinstance(value, str):
            value = value.strip()
        return None if value in {"", "0", 0} else value


@cache
def openrouter_settings() -> OpenRouterSettings:
    """Load and cache provider settings once for this process."""
    return OpenRouterSettings()


@cache
def runtime_settings() -> PipelineRuntimeSettings:
    """Load and cache numerical runtime settings once for this process."""
    return PipelineRuntimeSettings()


__all__ = [
    "OpenRouterSettings",
    "PipelineRuntimeSettings",
    "openrouter_settings",
    "runtime_settings",
]
