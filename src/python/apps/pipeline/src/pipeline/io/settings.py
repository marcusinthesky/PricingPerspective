"""Typed environment and dotenv settings for pipeline runtime boundaries."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from pydantic import Field, PositiveInt, field_validator  # noqa: TC002
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
def runtime_settings() -> PipelineRuntimeSettings:
    """Load and cache numerical runtime settings once for this process."""
    return PipelineRuntimeSettings()


class _BenchmarkThreadSettings(BaseSettings):
    """Actual non-secret process controls, read for provenance without dotenv."""

    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=None, extra="ignore", frozen=True
    )

    omp: str | None = Field(default=None, validation_alias="OMP_NUM_THREADS")
    openblas: str | None = Field(default=None, validation_alias="OPENBLAS_NUM_THREADS")
    mkl: str | None = Field(default=None, validation_alias="MKL_NUM_THREADS")
    numexpr: str | None = Field(default=None, validation_alias="NUMEXPR_NUM_THREADS")
    jax_platforms: str | None = Field(default=None, validation_alias="JAX_PLATFORMS")
    xla_flags: str | None = Field(default=None, validation_alias="XLA_FLAGS")


def benchmark_thread_settings() -> dict[str, str | None]:
    """Snapshot non-secret numerical process controls for measurement provenance.

    These are recorded values, not a second configuration owner. The helper
    neither changes process settings nor infers effective hardware parallelism
    from a requested thread cap.
    """
    actual = _BenchmarkThreadSettings()
    return {
        "OMP_NUM_THREADS": actual.omp,
        "OPENBLAS_NUM_THREADS": actual.openblas,
        "MKL_NUM_THREADS": actual.mkl,
        "NUMEXPR_NUM_THREADS": actual.numexpr,
        "JAX_PLATFORMS": actual.jax_platforms,
        "XLA_FLAGS": actual.xla_flags,
    }


__all__ = [
    "PipelineRuntimeSettings",
    "benchmark_thread_settings",
    "runtime_settings",
]
