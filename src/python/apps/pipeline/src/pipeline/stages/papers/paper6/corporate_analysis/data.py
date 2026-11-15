"""Admit only the frozen corporate article support and dated financial inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from hotelling.dynamics.artifacts import file_hash, repository_root

from pipeline._kernels.paper6.corporate_analysis import YEARS
from pipeline.stages.papers.paper6.artifacts import read

if TYPE_CHECKING:
    from pathlib import Path

    from pipeline._kernels.paper6.corporate_analysis import Cloud, CloudKey, FloatArray

SUPPORT = "data/papers/paper6/historical-evidence/t79/article-support.npz"
INVENTORY = "data/papers/paper6/review-artifacts/t81/inventory.json"
BASELINE = "data/papers/paper6/review-artifacts/t81"
INPUTS = "data/papers/paper6/corporate_inputs"
WIDTH = 2560
FIRMS = 100


def _admit(path: Path, expected: str, original: str) -> dict[str, str]:
    if not path.is_file():
        msg = f"missing frozen input: {path}; expected SHA256 {expected} ({original})"
        raise FileNotFoundError(msg)
    actual = file_hash(path)
    if actual != expected:
        msg = f"frozen input mismatch: {path}; expected SHA256 {expected}, got {actual}"
        raise ValueError(msg)
    return {"original_path": original, "sha256": actual}


def inventory() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Check scientific input hashes without interpreting obsolete execution files."""
    root = repository_root()
    saved = read(root / INVENTORY)
    hashes = saved["input_sha256"]
    admitted = {
        name: _admit(root / name, expected, name) for name, expected in hashes.items()
    }
    for name in (INVENTORY, f"{BASELINE}/results.json", f"{BASELINE}/measurement.npz"):
        admitted[name] = {"original_path": name, "sha256": file_hash(root / name)}
    return saved, admitted


def _firm(firm: str) -> tuple[dict[str, Any], bool]:
    """Validate vector identities and the published text-free cohort metadata."""
    root = repository_root()
    table = pq.read_table(
        root / f"data/shared/embeddings/qwen3-embedding-4b/{firm}.parquet"
    )
    vector = table["embedding"].combine_chunks()
    if not np.all(pc.call_function("list_value_length", [vector]).to_numpy() == WIDTH):
        msg = f"wrong embedding width: {firm}"
        raise ValueError(msg)
    points = vector.values.to_numpy(zero_copy_only=False).reshape(len(table), WIDTH)
    norms = np.linalg.norm(points, axis=1)
    if (
        not np.isfinite(points).all()
        or np.any(norms <= 0)
        or np.any(np.linalg.norm(points[:, :64], axis=1) <= 0)
    ):
        msg = f"invalid embedding vector: {firm}"
        raise ValueError(msg)
    meta = table.drop(["embedding"]).to_pydict()
    keys = np.asarray(meta["url_hash"], dtype=str)
    dates = np.asarray(
        [str(value)[:10] for value in meta["created_date"]], dtype="datetime64[D]"
    )
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    if (
        len(set(keys)) != len(keys)
        or any(not key for key in keys)
        or set(meta["symbol"]) != {firm}
        or set(meta["embedding_dim"]) != {WIDTH}
        or np.isnat(dates).any()
        or np.any((years < YEARS[0]) | (years > YEARS[-1]))
        or any(meta.get("_missing_date", []))
    ):
        msg = f"invalid article identity/date: {firm}"
        raise ValueError(msg)
    source = pq.read_table(
        root / f"data/shared/article_metadata/{firm}.parquet"
    ).to_pylist()
    by_id = {row["url_hash"]: row for row in source}
    if len(by_id) != len(source) or set(by_id) != set(keys):
        msg = f"source article IDs differ: {firm}"
        raise ValueError(msg)
    for index, key in enumerate(keys):
        row = by_id[key]
        if str(row["created_date"])[:10] != str(dates[index]):
            message = f"source article date differs: {firm}/{key}"
            raise ValueError(message)
        if meta["text_length"][index] != row["text_length"]:
            message = f"prepared-text length differs: {firm}/{key}"
            raise ValueError(message)
    order = np.argsort(keys, kind="stable")
    return {
        "keys": keys[order],
        "dates": dates[order],
        "years": years[order],
        "points": np.asarray(points[order], dtype=np.float64) / norms[order, None],
    }, True


def load_clouds(saved: dict[str, Any]) -> tuple[dict[CloudKey, Cloud], dict[str, Any]]:
    """Reconcile every admitted article and all 500 firm-year support counts."""
    root = repository_root()
    with np.load(root / SUPPORT, allow_pickle=False) as source:
        ids = source["firm_ids"].astype(str)
        support_keys = source["keys"].astype(str)
        support_dates = source["dates"].astype("datetime64[D]")
        support_firms = source["firm_index"]
        support_core = source["core"].astype(bool)
    if (
        ids.tolist() != saved["firm_ids"]
        or len(ids) != FIRMS
        or len(set(support_keys)) != len(support_keys)
    ):
        msg = "frozen all-firm ordering or article identity changed"
        raise ValueError(msg)
    with np.load(root / INPUTS / "design/arrays.npz", allow_pickle=False) as design:
        np.testing.assert_array_equal(design["firm_ids"].astype(str), ids)
        canonical = {
            str(key): (int(firm), date)
            for key, firm, date in zip(
                design["article_keys"],
                design["firm_index"],
                design["article_dates"].astype("datetime64[D]"),
                strict=True,
            )
        }
    expected = {(row["firm"], row["year"]): row for row in saved["annual_counts"]}
    clouds: dict[CloudKey, Cloud] = {}
    missing_text = []
    article_total = matched_total = 0
    for i, firm in enumerate(ids):
        values, missing = _firm(str(firm))
        if missing:
            missing_text.append(str(firm))
        selected = support_firms == i
        order = np.argsort(support_keys[selected], kind="stable")
        np.testing.assert_array_equal(values["keys"], support_keys[selected][order])
        np.testing.assert_array_equal(values["dates"], support_dates[selected][order])
        flags = np.asarray([str(key) in canonical for key in values["keys"]])
        np.testing.assert_array_equal(flags, support_core[selected][order])
        for key, date in zip(
            values["keys"][flags], values["dates"][flags], strict=True
        ):
            if canonical[str(key)] != (i, date):
                msg = f"canonical article identity differs: {firm}/{key}"
                raise ValueError(msg)
        for year in YEARS:
            mask = values["years"] == year
            core = flags[mask]
            counts = expected[str(firm), year]
            if (
                len(core) != counts["expanded"]
                or int(core.sum()) != counts["matched"]
                or not core.any()
            ):
                msg = f"frozen annual count differs: {firm}/{year}"
                raise ValueError(msg)
            clouds[i, year] = {
                "points": values["points"][mask],
                "dates": values["dates"][mask],
                "core": core,
            }
            article_total += len(core)
            matched_total += int(core.sum())
    if matched_total != len(canonical):
        msg = "canonical articles absent from expanded support"
        raise ValueError(msg)
    return clouds, {
        "firm_year_counts_reconciled": len(expected),
        "expanded_articles": article_total,
        "matched_articles": matched_total,
        "missing_retained_content_firms": missing_text,
        "article_identity_and_dates": "exact frozen support and source IDs",
        "duplicate_family_treatment": (
            "Frozen cohort retained; no deduplication applied. "
            "Complete family mapping unavailable."
        ),
        "market_commentary": (
            "Included according to frozen article membership; "
            "no new content filter applied."
        ),
        "provenance_limits": saved["source_limitations"],
    }


def financial(firms: list[str]) -> dict[int, tuple[np.ndarray, FloatArray]]:
    """Open unchanged daily excess-return panels after support admission."""
    root = repository_root()
    result = {}
    for year in YEARS[1:]:
        with np.load(
            root / INPUTS / f"returns/{year}.npz", allow_pickle=False
        ) as source:
            np.testing.assert_array_equal(source["firm_ids"].astype(str), firms)
            dates = source["dates"].astype("datetime64[D]")
            returns = source["excess_returns"]
        if (
            returns.shape != (len(dates), len(firms))
            or not np.isfinite(returns).all()
            or np.any(np.diff(dates) <= np.timedelta64(0, "D"))
            or np.any(dates.astype("datetime64[Y]").astype(int) + 1970 != year)
        ):
            msg = f"invalid frozen financial panel: {year}"
            raise ValueError(msg)
        result[year] = dates, returns
    return result
