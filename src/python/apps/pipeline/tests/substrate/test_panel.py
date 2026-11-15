"""Contract tests for shared priced-universe and return-panel loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pipeline.stages.substrate import panel as panel_module
from pipeline.stages.substrate.panel import (
    load_aligned_return_panel,
    load_priced_distance_universe,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_prices(path: Path, dates: list[str], prices: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "close_price": prices,
        }
    )
    frame["return"] = frame["close_price"].pct_change()
    frame.to_parquet(path, index=False)


def test_aligned_panel_recomputes_the_training_evaluation_seam(
    tmp_path: Path,
) -> None:
    """The first evaluation return uses the final training close."""
    train = tmp_path / "train"
    evaluation = tmp_path / "evaluation"
    train.mkdir()
    evaluation.mkdir()
    _write_prices(
        train / "AAA.parquet",
        ["2022-12-29", "2022-12-30"],
        [100.0, 110.0],
    )
    _write_prices(
        evaluation / "AAA.parquet",
        ["2023-01-03", "2023-01-04"],
        [121.0, 133.1],
    )

    dates, returns = load_aligned_return_panel(
        train,
        ["AAA"],
        start="2023-01-03",
        oos_returns_dir=evaluation,
    )

    assert dates.tolist() == [
        np.datetime64("2023-01-03T00:00:00.000000"),
        np.datetime64("2023-01-04T00:00:00.000000"),
    ]
    np.testing.assert_allclose(returns[:, 0], [0.1, 0.1])


def test_aligned_panel_rejects_empty_ticker_roster(tmp_path: Path) -> None:
    """An empty roster cannot define a panel."""
    with pytest.raises(ValueError, match="at least one"):
        load_aligned_return_panel(tmp_path, [])


def test_priced_universe_uses_public_typed_artifact_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rooted W2 remains a distance and is squared only for quadratic work."""
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    (returns_dir / "AAA.parquet").touch()
    values = pd.DataFrame(
        {
            "item_i": ["AAA", "AAA", "BBB", "BBB"],
            "item_j": ["AAA", "BBB", "AAA", "BBB"],
            "value": [0.0, 0.4, 0.4, 0.0],
        }
    )
    monkeypatch.setattr(
        panel_module,
        "read_typed_distance_artifact",
        lambda *_args, **_kwargs: (
            values,
            {
                "item_ids": ["AAA", "BBB"],
                "metadata": {"value_semantics": "statistical_distance"},
            },
        ),
    )

    tickers, distance, squared = load_priced_distance_universe(
        tmp_path / "typed", returns_dir
    )

    assert tickers == ["AAA"]
    np.testing.assert_array_equal(distance, np.zeros((1, 1)))
    np.testing.assert_array_equal(squared, np.zeros((1, 1)))
