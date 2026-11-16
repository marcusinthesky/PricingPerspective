"""Validated immutable Kenneth French daily-factor inputs."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Pydantic resolves runtime annotations.
from typing import Literal, NoReturn, Self, cast

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

type FactorModelId = Literal["ff3", "ff3_mom", "ff5", "ff5_mom"]
type FactorSourceId = Literal["ff3", "ff5", "mom"]

_DATE = re.compile(r"^\d{8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FACTOR_SOURCE_COUNT = 3
_SOURCE_COLUMNS = {
    "ff3": ("MKT_RF", "SMB", "HML", "RF"),
    "ff5": ("MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"),
    "mom": ("MOM",),
}
_MODEL_COLUMNS = {
    "ff3": ("MKT_RF", "SMB", "HML"),
    "ff3_mom": ("MKT_RF", "SMB", "HML", "MOM"),
    "ff5": ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
    "ff5_mom": ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"),
}
_HEADER_ALIASES = {"Mkt-RF": "MKT_RF", "Mom": "MOM"}
_MISSING_SENTINELS = {-99.99, -999.0}


class FrenchFactorError(ValueError):
    """Raised when a frozen French factor input violates its contract."""


def _fail(message: str) -> NoReturn:
    raise FrenchFactorError(message)


class FactorSourceConfig(BaseModel):
    """One immutable Kenneth French daily-factor ZIP snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: FactorSourceId
    path: Path
    source_url: str = Field(min_length=1)
    sha256: str
    source_vintage: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sha256(self) -> Self:
        """Require a canonical immutable source identity."""
        if _SHA256.fullmatch(self.sha256) is None:
            _fail(f"factor source {self.source_id!r} needs a lowercase SHA-256")
        return self


class FactorDataConfig(BaseModel):
    """Shared primary and nested factor-model identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[FactorSourceConfig, ...]
    primary_model: Literal["ff5_mom"]
    diagnostic_models: tuple[Literal["ff3", "ff3_mom", "ff5"], ...]

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require independent FF3, FF5, and momentum source snapshots."""
        source_ids = tuple(source.source_id for source in self.sources)
        if (
            set(source_ids) != {"ff3", "ff5", "mom"}
            or len(source_ids) != _FACTOR_SOURCE_COUNT
        ):
            _fail("factor data requires exactly ff3, ff5, and mom")
        if self.diagnostic_models != ("ff3", "ff3_mom", "ff5"):
            _fail("factor data requires the frozen FF3/FF3+MOM/FF5 diagnostic order")
        return self

    def source(self, source_id: FactorSourceId) -> FactorSourceConfig:
        """Return a source by its stable registry identity."""
        for source in self.sources:
            if source.source_id == source_id:
                return source
        message = f"unknown factor source {source_id!r}"
        raise KeyError(message)


@dataclass(frozen=True, slots=True)
class FrenchFactorPanel:
    """One assembled factor model with merge and source provenance."""

    model_id: FactorModelId
    frame: pd.DataFrame
    factor_columns: tuple[str, ...]
    source_hashes: dict[str, str]
    dropped_dates: dict[str, int]


def load_factor_data_config(params_file: Path, namespace: str) -> FactorDataConfig:
    """Load one paper's shared factor-data contract from ``params.yaml``."""
    try:
        raw_value = yaml.safe_load(params_file.read_text(encoding="utf-8"))
        raw = cast("dict[str, object]", raw_value)
        paper = cast("dict[str, object]", raw[namespace])
        return FactorDataConfig.model_validate(paper["factor_data"])
    except (
        KeyError,
        OSError,
        TypeError,
        ValidationError,
        yaml.YAMLError,
    ) as error:
        message = f"{params_file}: invalid {namespace} factor-data contract: {error}"
        raise FrenchFactorError(message) from error


def _canonical_header(value: str) -> str:
    stripped = value.strip()
    return _HEADER_ALIASES.get(stripped, stripped)


def _read_source(source: FactorSourceConfig) -> tuple[pd.DataFrame, str]:  # noqa: C901, PLR0912
    payload = source.path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        _fail(f"{source.path}: SHA-256 {digest} disagrees with {source.sha256}")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = tuple(name for name in archive.namelist() if not name.endswith("/"))
            if len(names) != 1:
                _fail(f"{source.path}: expected exactly one CSV member")
            text = archive.read(names[0]).decode("utf-8-sig")
    except (UnicodeDecodeError, zipfile.BadZipFile) as error:
        message = f"{source.path}: invalid French factor ZIP: {error}"
        raise FrenchFactorError(message) from error

    rows = list(csv.reader(io.StringIO(text)))
    expected = _SOURCE_COLUMNS[source.source_id]
    header_index = -1
    header: tuple[str, ...] = ()
    for index, row in enumerate(rows):
        candidate = tuple(_canonical_header(value) for value in row)
        if candidate and not candidate[0] and candidate[1:] == expected:
            header_index = index
            header = ("DATE", *candidate[1:])
            break
    if header_index < 0:
        _fail(f"{source.path}: expected columns {expected} were not found")

    parsed: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        if not row or not _DATE.fullmatch(row[0].strip()):
            continue
        if len(row) != len(header):
            _fail(f"{source.path}: malformed daily row {row[0]!r}")
        values = [float(value.strip()) for value in row[1:]]
        if any(value in _MISSING_SENTINELS for value in values):
            _fail(f"{source.path}: missing sentinel in daily row {row[0]!r}")
        parsed.append(
            {
                "Date": pd.to_datetime(row[0].strip(), format="%Y%m%d"),
                **{
                    name: value / 100.0
                    for name, value in zip(header[1:], values, strict=True)
                },
            }
        )
    frame = pd.DataFrame(parsed)
    if frame.empty:
        _fail(f"{source.path}: no daily factor rows")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        _fail(f"{source.path}: dates must be unique and increasing")
    numeric = frame.loc[:, list(expected)].to_numpy(dtype=np.float64)
    if not bool(np.all(np.isfinite(numeric))):
        _fail(f"{source.path}: factor values must be finite")
    return frame, digest


def load_factor_model(
    config: FactorDataConfig,
    model_id: FactorModelId,
) -> FrenchFactorPanel:
    """Load one primary or nested daily factor model from frozen ZIPs."""
    if model_id not in _MODEL_COLUMNS:
        _fail(f"unknown factor model {model_id!r}")
    base_id: FactorSourceId = "ff3" if model_id.startswith("ff3") else "ff5"
    base, base_hash = _read_source(config.source(base_id))
    source_hashes: dict[str, str] = {base_id: base_hash}
    dropped_dates: dict[str, int] = {base_id: 0}
    assembled = base
    if model_id.endswith("_mom"):
        momentum, momentum_hash = _read_source(config.source("mom"))
        source_hashes["mom"] = momentum_hash
        shared_dates = set(base["Date"]).intersection(momentum["Date"])
        dropped_dates = {
            base_id: len(base) - len(shared_dates),
            "mom": len(momentum) - len(shared_dates),
        }
        assembled = base.merge(
            momentum.loc[:, ["Date", "MOM"]],
            on="Date",
            how="inner",
            validate="one_to_one",
            sort=True,
        )
    factor_columns = _MODEL_COLUMNS[model_id]
    columns = ("Date", *factor_columns, "RF")
    assembled = assembled.loc[:, list(columns)].reset_index(drop=True)
    if (
        assembled["Date"].duplicated().any()
        or not assembled["Date"].is_monotonic_increasing
    ):
        _fail(f"{model_id}: merged factor dates must be unique and increasing")
    return FrenchFactorPanel(
        model_id=model_id,
        frame=assembled,
        factor_columns=factor_columns,
        source_hashes=source_hashes,
        dropped_dates=dropped_dates,
    )


__all__ = [
    "FactorDataConfig",
    "FactorModelId",
    "FactorSourceConfig",
    "FactorSourceId",
    "FrenchFactorError",
    "FrenchFactorPanel",
    "load_factor_data_config",
    "load_factor_model",
]
