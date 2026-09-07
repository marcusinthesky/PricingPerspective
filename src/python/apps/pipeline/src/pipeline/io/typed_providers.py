"""Provider and representation loading for typed distance stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pandas as pd

from pipeline._kernels.typed_geometry import (
    TypedGeometryError,
    apply_representation,
    make_cloud,
)

if TYPE_CHECKING:
    from pipeline._kernels.typed_geometry import EmbeddingCloud
    from pipeline.io.typed_analysis import (
        ProviderConfig,
        RepresentationSpec,
        SampleWindowSpec,
    )

VECTOR_RANK = 2
OBSERVATION_DATE_COLUMN = "created_date"


class ProviderInputError(ValueError):
    """Raised when a provider artifact violates its declared contract."""


@dataclass(frozen=True, slots=True)
class DatedProviderObservations:
    """One provider item with deterministic rows and retained article dates."""

    item_id: str
    values: np.ndarray
    dates: np.ndarray
    url_hashes: tuple[str, ...]
    source_hash: str


def _provider_error(message: str) -> NoReturn:
    raise ProviderInputError(message)


def _order_by_url_hash(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Put a provider frame in its canonical row order, rejecting bad keys.

    The sort is what makes ``sample_size`` truncation and window filtering
    reproducible, so a missing or duplicated key is refused rather than left to
    decide row order by parquet write order.
    """
    if "url_hash" not in frame:
        return frame
    if frame["url_hash"].isna().any():
        _provider_error(f"{path} has missing url_hash values")
    if frame["url_hash"].duplicated().any():
        _provider_error(f"{path} has duplicate url_hash values")
    return frame.sort_values("url_hash", kind="mergesort")


def _apply_window(
    frame: pd.DataFrame, window: SampleWindowSpec, path: Path
) -> pd.DataFrame:
    """Restrict one provider frame to a declared inclusive observation window.

    This is the only place in the typed stack that drops provider rows on an
    observation date.  It rejects rather than repairs, in the sense of the
    doors described in ``jcor/sample/evidence.py``: an undated row cannot be
    placed inside or outside the window, so imputing one would silently decide
    a sample-inclusion question the caller never asked about.
    """
    if OBSERVATION_DATE_COLUMN not in frame:
        _provider_error(
            f"{path} has no {OBSERVATION_DATE_COLUMN} column; window "
            f"{window.window_id!r} cannot be applied"
        )
    dates = pd.to_datetime(
        frame[OBSERVATION_DATE_COLUMN], errors="coerce"
    ).dt.normalize()
    if dates.isna().any():
        _provider_error(
            f"{path} has {int(dates.isna().sum())} rows with an unparsable "
            f"{OBSERVATION_DATE_COLUMN}; window {window.window_id!r} is undefined there"
        )
    keep = pd.Series(data=True, index=frame.index)
    if window.start is not None:
        keep &= dates >= pd.Timestamp(window.start)
    if window.end is not None:
        keep &= dates <= pd.Timestamp(window.end)
    windowed = frame.loc[keep]
    if windowed.empty:
        _provider_error(f"{path} has no rows inside window {window.window_id!r}")
    return windowed


def load_provider_clouds(  # noqa: C901, PLR0912
    provider: ProviderConfig,
    representation: RepresentationSpec,
    *,
    item_ids: list[str] | tuple[str, ...] | None = None,
    sample_size: int | None = None,
    window: SampleWindowSpec | None = None,
) -> tuple[EmbeddingCloud, ...]:
    """Load and validate one provider's deterministic per-item Parquet tree.

    ``window`` restricts each item to the declared inclusive observation-date
    range before the cloud is built. When ``sample_size`` is also declared, the
    loader then takes the first ``m`` rows in canonical URL-hash order inside
    that window. This deterministic window-then-balance rule is required by
    the equal-cardinality assignment solver used for empirical W1/W2.
    """
    if representation.provider_id != provider.provider_id:
        _provider_error("representation/provider identity mismatch")
    root = Path(provider.artifact_root)
    files = {path.stem: path for path in root.glob("*.parquet")}
    if not files:
        _provider_error(
            f"provider {provider.provider_id!r} has no Parquet files at {root}"
        )
    if item_ids is not None and len(set(item_ids)) != len(item_ids):
        _provider_error("requested provider item IDs must be unique")
    selected = sorted(files) if item_ids is None else sorted(item_ids)
    missing = [item_id for item_id in selected if item_id not in files]
    if missing:
        _provider_error(f"provider {provider.provider_id!r} is missing items {missing}")
    clouds: list[EmbeddingCloud] = []
    for item_id in selected:
        path = files[item_id]
        frame = pd.read_parquet(path)
        if "embedding" not in frame:
            _provider_error(f"{path} is missing embedding column")
        frame = _order_by_url_hash(frame, path)
        if window is not None:
            frame = _apply_window(frame, window, path)
        if sample_size is not None:
            if len(frame) < sample_size:
                _provider_error(
                    f"{path} has {len(frame)} rows, expected at least {sample_size}"
                )
            frame = frame.iloc[:sample_size]
        values = np.stack([np.asarray(value) for value in frame["embedding"]])
        if not np.issubdtype(values.dtype, np.floating):
            _provider_error(f"{path} embeddings must have a floating dtype")
        if values.ndim != VECTOR_RANK or values.shape[0] == 0 or values.shape[1] == 0:
            _provider_error(f"{path} embedding column must contain nonempty vectors")
        if values.shape[1] != provider.dimensions:
            _provider_error(
                f"{path} has width {values.shape[1]}, expected {provider.dimensions}"
            )
        if provider.normalization == "unit_rows":
            norms = np.linalg.norm(values, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
                _provider_error(f"{path} violates declared unit_rows normalization")
        if (
            representation.dimension is not None
            and representation.dimension > values.shape[1]
        ):
            _provider_error(
                f"representation {representation.representation_id!r} dimension "
                f"{representation.dimension} exceeds provider native width "
                f"{values.shape[1]}"
            )
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if provider.source_hash is not None and source_hash != provider.source_hash:
            _provider_error(f"{path} disagrees with declared provider source_hash")
        try:
            cloud = make_cloud(
                item_id,
                values,
                provider.provider_id,
                representation,
                source_hash=source_hash,
                native_dimension=provider.dimensions,
                provider_model_id=provider.model_id,
                provider_vintage=provider.vintage,
                provider_status=provider.status,
                provider_caveat=provider.caveat,
            )
        except TypedGeometryError as error:
            _provider_error(f"{path} representation transform failed: {error}")
        clouds.append(cloud)
    return tuple(clouds)


def load_dated_provider_observations(
    provider: ProviderConfig,
    representation: RepresentationSpec,
    *,
    item_ids: list[str] | tuple[str, ...] | None = None,
    sample_size: int | None = None,
    window: SampleWindowSpec | None = None,
) -> tuple[DatedProviderObservations, ...]:
    """Load deterministic provider rows while retaining dates for resampling.

    Selection deliberately matches :func:`load_provider_clouds`: canonical URL-hash
    order, then the declared date window, then optional truncation.  The existing
    cloud loader remains unchanged because dated rows are an inference-only contract.
    """
    if representation.provider_id != provider.provider_id:
        _provider_error("representation/provider identity mismatch")
    root = Path(provider.artifact_root)
    files = {path.stem: path for path in root.glob("*.parquet")}
    selected = sorted(files) if item_ids is None else sorted(item_ids)
    if not files or any(item_id not in files for item_id in selected):
        _provider_error("dated provider request contains unavailable items")
    observations: list[DatedProviderObservations] = []
    for item_id in selected:
        path = files[item_id]
        frame = _order_by_url_hash(pd.read_parquet(path), path)
        if window is not None:
            frame = _apply_window(frame, window, path)
        if sample_size is not None:
            if len(frame) < sample_size:
                _provider_error(
                    f"{path} has {len(frame)} rows, expected at least {sample_size}"
                )
            frame = frame.iloc[:sample_size]
        required = {"embedding", OBSERVATION_DATE_COLUMN, "url_hash"}
        missing = required.difference(frame.columns)
        if missing:
            _provider_error(
                f"{path} is missing dated-observation columns {sorted(missing)}"
            )
        dates = pd.to_datetime(
            frame[OBSERVATION_DATE_COLUMN], errors="coerce"
        ).dt.normalize()
        if dates.isna().any():
            _provider_error(f"{path} contains unparsable observation dates")
        raw_values = np.stack([np.asarray(value) for value in frame["embedding"]])
        try:
            values = apply_representation(raw_values, representation)
        except TypedGeometryError as error:
            _provider_error(f"{path} representation transform failed: {error}")
        observations.append(
            DatedProviderObservations(
                item_id=item_id,
                values=values,
                dates=dates.to_numpy(dtype="datetime64[D]"),
                url_hashes=tuple(frame["url_hash"].astype(str)),
                source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(observations)


__all__ = [
    "DatedProviderObservations",
    "ProviderInputError",
    "load_dated_provider_observations",
    "load_provider_clouds",
]
