"""Cold production processes own precision before certified numerical work."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import jax
import numpy as np
import pytest
from hotelling.dynamics.engine import canonical_capacity
from pipeline._kernels.paper6.recovery import HistoryMeasurement
from pipeline.cli.commands.p6 import recovery
from pipeline.stages.papers.paper6.recovery import campaign
from pipeline.stages.papers.paper6.recovery.campaign import _measure

if TYPE_CHECKING:
    from typing import Literal


def _cold_process_probe(
    boundary: Literal["cli", "worker"], output: Path
) -> tuple[str, str | None]:
    """Run in spawn's fresh interpreter with float64 disabled in its environment."""
    assert not jax.config.jax_enable_x64
    dtypes: list[str] = []

    def numerical_work() -> None:
        assert jax.config.jax_enable_x64
        dtypes.append(str(canonical_capacity(4).dtype))

    if boundary == "cli":

        def run(*_args: object, **_kwargs: object) -> dict[str, object]:
            numerical_work()
            return {"status": "complete", "failures": []}

        with patch.object(campaign, "run_campaign", side_effect=run):
            recovery.recover(output, "1", 1)
    else:

        def measure(*_args: object, **_kwargs: object) -> HistoryMeasurement:
            numerical_work()
            return HistoryMeasurement(
                1.0,
                1.0,
                np.ones(1),
                np.zeros(1),
                np.ones((1, 1, 1)),
                0.0,
                1.0,
                1.0,
                1.0,
                0,
            )

        with patch.object(campaign, "measure_history", side_effect=measure):
            _measure(0, np.ones((5, 12, 4), dtype=np.int64), 1, None, 1)
    assert dtypes == ["float64"]
    return dtypes[0], jax.config.jax_compilation_cache_dir


@pytest.mark.parametrize("boundary", ["cli", "worker"])
def test_recovery_process_entrypoints_select_precision(
    boundary: Literal["cli", "worker"],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parent test precision cannot hide a missing CLI or spawned-worker guard."""
    cache = tmp_path / "compilation-cache"
    monkeypatch.setenv("JAX_ENABLE_X64", "false")
    monkeypatch.setenv("PP_JAX_CACHE_DIR", str(cache))
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        dtype, configured_cache = executor.submit(
            _cold_process_probe, boundary, tmp_path
        ).result(timeout=60)
    assert dtype == "float64"
    if boundary == "worker":
        assert configured_cache is not None
        assert Path(configured_cache) == cache.resolve()
