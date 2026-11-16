"""Typed environment and dotenv settings tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pipeline.io.settings import PipelineRuntimeSettings
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

BLAS_LIMIT = 3


@pytest.mark.parametrize("value", ["", "0", 0])
def test_zero_or_empty_blas_limit_disables_cap(value: str | int) -> None:
    """Retain the established zero-means-unlimited runtime convention."""
    settings = PipelineRuntimeSettings(
        _env_file=None,
        PP_BLAS_LIMIT=value,
    )
    assert settings.blas_limit is None


def test_runtime_settings_validate_paths_and_positive_limits(tmp_path: Path) -> None:
    """Parse runtime values into their operational types."""
    settings = PipelineRuntimeSettings(
        _env_file=None,
        PP_JAX_CACHE_DIR=str(tmp_path / "jax"),
        PP_BLAS_LIMIT=str(BLAS_LIMIT),
    )
    assert settings.jax_cache_dir == tmp_path / "jax"
    assert settings.blas_limit == BLAS_LIMIT

    with pytest.raises(ValidationError):
        PipelineRuntimeSettings(_env_file=None, PP_BLAS_LIMIT="-1")


def test_settings_models_remain_frozen() -> None:
    """Prevent runtime configuration from changing after validation."""
    assert PipelineRuntimeSettings.model_config["frozen"] is True
