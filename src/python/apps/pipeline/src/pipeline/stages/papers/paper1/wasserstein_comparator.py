"""Paper 1's controlled cross-encoder distributional comparator.

The common-width Wasserstein W1 and W2 matrices are read from the shared
typed-distance backplane. Matched V-energy, signed-U, and sample-size
sensitivity pairs are computed here on deterministic
``url_hash`` prefixes. Rows are L2-normalized and use harmonized Euclidean chord
ground geometry across complete encoder/input pipelines. This is an inference
geometry, not a claim that the four encoders share one native training loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import yaml
from statadic.geometry.discrepancy.transport.exact import chord_cost_matrix

from pipeline._kernels.typed_distances import (
    EmbeddingCloud,
    compute_statistical_distance,
)
from pipeline.io.typed_analysis import PointDistanceSpec, StatisticalDistanceSpec
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

_COMMON_M = 128
# The frozen Qwen3-Embedding-8B pool contains 128 rows per firm. Keep the
# sensitivity within that support.
_SENSITIVITY_M = (32, 64, 128)
# Roster sizes. The source of truth is params.yaml:symbols / market_symbols;
# these mirror it because this module never loads params. They are equal now:
# every firm in the news universe is return-eligible, so the projection to
# market_symbols is the identity and market_symbol_exclusions is empty.
_PRICED_UNIVERSE_SIZE = 100
_EMBEDDING_UNIVERSE_SIZE = 100
#: Mirrors params.yaml:market_symbol_exclusions -- firms carrying an embedding
#: but deliberately no return series. The current governed roster has none.
_EMBEDDING_ONLY_TICKERS: tuple[str, ...] = ()
_ENCODER_COUNT = 4
_FUNCTIONAL_NEGATIVE_TOLERANCE = -1e-10
_CORRELATION_SYMMETRY_TOLERANCE = 1e-12
_MATRIX_NDIM = 2
_MODEL_DIRS = {
    "qwen3-embedding-4b": "qwen3-embedding-4b",
    "qwen3-embedding-8b": "qwen3-embedding-8b",
    "bge-m3": "bge-m3",
    "bge-large-en-v1.5": "bge-large-en-v1.5",
}


class WassersteinInputError(ValueError):
    """Report invalid comparator inputs or incompatible artifacts."""


@dataclass(frozen=True)
class TransportMatrices:
    """The two precomputed transport matrices for one encoder at one ``m``.

    They travel together because they are the same matched sample read at two
    transport orders; splitting them into separate arguments would let a caller
    pass a $W_1$ matrix where a $W_2$ one is expected without any type saying so.
    """

    order_one: np.ndarray
    order_two: np.ndarray


@dataclass(frozen=True)
class WassersteinComparatorConfig:
    """Input paths and the matched encoder pair artifact."""

    embeddings_root: Path
    typed_distance_root: Path
    covariance_matrix: Path
    output_pairs: Path


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != _MATRIX_NDIM or values.shape[0] == 0 or values.shape[1] == 0:
        message = "embedding sample must be a nonempty matrix"
        raise WassersteinInputError(message)
    if not np.all(np.isfinite(values)):
        message = "embedding sample contains nonfinite values"
        raise WassersteinInputError(message)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        message = "embedding sample contains a zero-norm row"
        raise WassersteinInputError(message)
    return values / norms[:, None]


def _load_prefix(path: Path, n_rows: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Load a deterministic lexicographic ``url_hash`` prefix."""
    frame = pd.read_parquet(path, columns=["url_hash", "embedding"])
    if frame["url_hash"].isna().any() or frame["url_hash"].duplicated().any():
        message = f"url_hash must be nonmissing and unique in {path}"
        raise WassersteinInputError(message)
    frame = frame.sort_values("url_hash", kind="mergesort")
    if len(frame) < n_rows:
        message = f"{path} has {len(frame)} rows; {n_rows} required"
        raise WassersteinInputError(message)
    selected = frame.iloc[:n_rows]
    hashes = tuple(str(value) for value in selected["url_hash"])
    rows = _unit_rows(np.stack(list(selected["embedding"])))
    return hashes, rows


def _embedding_files(directory: Path) -> dict[str, Path]:
    files = {path.stem: path for path in directory.glob("*.parquet")}
    if not files:
        message = f"no embedding parquets found in {directory}"
        raise WassersteinInputError(message)
    return files


def _load_model(
    directory: Path, n_rows: int
) -> tuple[dict[str, tuple[str, ...]], dict[str, np.ndarray]]:
    hashes: dict[str, tuple[str, ...]] = {}
    samples: dict[str, np.ndarray] = {}
    for ticker, path in sorted(_embedding_files(directory).items()):
        hashes[ticker], samples[ticker] = _load_prefix(path, n_rows)
    return hashes, samples


def _load_return_distances(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_parquet(path, columns=["ticker_i", "ticker_j", "correlation"])
    tickers = sorted(set(frame["ticker_i"]) | set(frame["ticker_j"]))
    lookup = {ticker: pos for pos, ticker in enumerate(tickers)}
    corr = np.full((len(tickers), len(tickers)), np.nan, dtype=np.float64)
    for row in frame.itertuples(index=False):
        corr[lookup[str(row.ticker_i)], lookup[str(row.ticker_j)]] = float(
            str(row.correlation)
        )
    if not np.all(np.isfinite(corr)):
        message = "return correlation matrix is incomplete or nonfinite"
        raise WassersteinInputError(message)
    if not np.allclose(corr, corr.T, atol=_CORRELATION_SYMMETRY_TOLERANCE):
        message = "return correlation matrix is not symmetric"
        raise WassersteinInputError(message)
    corr = np.clip(corr, -1.0, 1.0)
    return tickers, np.sqrt(np.maximum(2.0 - 2.0 * corr, 0.0))


def _load_typed_wasserstein_matrix(
    root: Path, model: str, tickers: list[str], distance_id: str = "wasserstein_w1"
) -> np.ndarray:
    """Load the canonical 128-row Wasserstein artifact for one encoder."""
    artifact = root / model / f"{model}-unit" / distance_id
    try:
        frame, summary = read_typed_distance_artifact(
            artifact,
            expected_identity={
                "provider_id": model,
                "representation_id": f"{model}-unit",
                "distance_id": distance_id,
                "sample_size": 128,
            },
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        message = f"invalid typed Wasserstein artifact for {model}: {artifact}"
        raise WassersteinInputError(message) from exc
    item_ids = summary.get("item_ids")
    if not isinstance(item_ids, list) or set(item_ids) != set(tickers):
        message = f"typed Wasserstein ticker coverage differs for {model}"
        raise WassersteinInputError(message)
    positions = {str(item_id): index for index, item_id in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    for row in frame.itertuples(index=False):
        left, right = str(row.item_i), str(row.item_j)
        matrix[positions[left], positions[right]] = float(cast("float", row.value))
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        message = f"typed Wasserstein matrix is invalid for {model}"
        raise WassersteinInputError(message)
    return matrix


def _within_terms(sample: np.ndarray) -> tuple[float, float]:
    """Return the V- and U-statistic within-cloud mean chord costs.

    Called once per ticker, so this deliberately stays on the eager float64
    ``chord_cost_matrix`` door: it is where every sample entering
    :func:`_pair_metrics` passes Statadic's unit-row validation, and this per-firm
    loop is not the computational bottleneck.
    """
    costs = chord_cost_matrix(sample, sample)
    m = costs.shape[0]
    v_term = float(costs.mean(dtype=np.float64))
    u_term = float(costs.sum(dtype=np.float64) / (m * (m - 1)))
    return v_term, u_term


_WASSERSTEIN_DISTANCE = StatisticalDistanceSpec(
    distance_id="wasserstein_w1",
    family="wasserstein",
    ground_distance="euclidean_chord",
    exponent=1.0,
    estimator="balanced_wasserstein_1",
    normalization="rooted",
)
_WASSERSTEIN2_DISTANCE = StatisticalDistanceSpec(
    distance_id="wasserstein_w2",
    family="wasserstein",
    ground_distance="euclidean_chord",
    exponent=1.0,
    estimator="balanced_wasserstein_2",
    normalization="rooted",
)
_CHORD_GROUND = PointDistanceSpec(
    distance_id="euclidean_chord",
    metric="euclidean",
    domain="unit_sphere",
)


def _typed_wasserstein_matrix(
    samples: dict[str, np.ndarray],
    tickers: list[str],
    m: int,
    distance: StatisticalDistanceSpec = _WASSERSTEIN_DISTANCE,
) -> np.ndarray:
    """Compute a sample-prefix Wasserstein matrix through the shared typed kernel."""
    clouds = tuple(
        EmbeddingCloud(
            item_id=ticker,
            values=_unit_rows(samples[ticker][:m]),
            provider_id="paper1-comparator",
            representation_id="paper1-comparator-unit",
        )
        for ticker in tickers
    )
    return np.asarray(
        compute_statistical_distance(clouds, distance, _CHORD_GROUND).values,
        dtype=np.float64,
    )


def _sqrt_functional(value: float, label: str) -> float:
    if not np.isfinite(value):
        message = f"{label} energy functional is nonfinite"
        raise WassersteinInputError(message)
    if value < _FUNCTIONAL_NEGATIVE_TOLERANCE:
        message = f"{label} energy functional is negative ({value:.3e})"
        raise WassersteinInputError(message)
    return float(np.sqrt(max(value, 0.0)))


def _pair_metrics(
    samples: dict[str, np.ndarray],
    tickers: list[str],
    return_dist: np.ndarray,
    model: str,
    m: int,
    transport: TransportMatrices | None = None,
) -> pd.DataFrame:
    # This is a pipeline boundary: callers may supply raw model embeddings, but
    # every statistic below is defined on the unit-sphere projection. Keep the
    # model-aware transformation here and pass only Euclidean/chord inputs to
    # Statadic. The operation is idempotent for artifacts normalized by `_load_prefix`.
    normalized = {ticker: _unit_rows(samples[ticker][:m]) for ticker in tickers}
    # Every ticker's sample passes Statadic's eager unit-row validation here, once,
    # before the dyad loop enters the transformable core with it.
    within = {ticker: _within_terms(normalized[ticker]) for ticker in tickers}
    pairs = list(combinations(tickers, 2))
    if transport is None:
        transport = TransportMatrices(
            order_one=_typed_wasserstein_matrix(normalized, tickers, m),
            order_two=_typed_wasserstein_matrix(
                normalized, tickers, m, _WASSERSTEIN2_DISTANCE
            ),
        )
    wasserstein_matrix = transport.order_one
    wasserstein2_matrix = transport.order_two
    expected_shape = (len(tickers), len(tickers))
    if (
        wasserstein_matrix.shape != expected_shape
        or wasserstein2_matrix.shape != expected_shape
    ):
        message = "typed Wasserstein matrix has the wrong shape"
        raise WassersteinInputError(message)
    cross_values = np.asarray(
        [
            float(chord_cost_matrix(normalized[ticker_i], normalized[ticker_j]).mean())
            for ticker_i, ticker_j in pairs
        ],
        dtype=np.float64,
    )
    lookup = {ticker: pos for pos, ticker in enumerate(tickers)}
    w1_values = np.asarray(
        [
            wasserstein_matrix[lookup[ticker_i], lookup[ticker_j]]
            for ticker_i, ticker_j in pairs
        ],
        dtype=np.float64,
    )
    w2_values = np.asarray(
        [
            wasserstein2_matrix[lookup[ticker_i], lookup[ticker_j]]
            for ticker_i, ticker_j in pairs
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    for pair_index, (ticker_i, ticker_j) in enumerate(pairs):
        cross = float(cross_values[pair_index])
        v_functional = 2.0 * cross - within[ticker_i][0] - within[ticker_j][0]
        u_functional = 2.0 * cross - within[ticker_i][1] - within[ticker_j][1]
        rows.append(
            {
                "model": model,
                "m": m,
                "ticker_i": ticker_i,
                "ticker_j": ticker_j,
                "return_chord": float(return_dist[lookup[ticker_i], lookup[ticker_j]]),
                "wasserstein_w1": float(w1_values[pair_index]),
                "wasserstein_w2": float(w2_values[pair_index]),
                "energy_v": _sqrt_functional(v_functional, "V-statistic"),
                "energy_v_functional": v_functional,
                "energy_u_functional": u_functional,
            }
        )
    frame = pd.DataFrame(rows)
    u_values = frame["energy_u_functional"].to_numpy(dtype=np.float64)
    if bool(np.all(u_values >= _FUNCTIONAL_NEGATIVE_TOLERANCE)):
        frame["energy_u_sqrt"] = np.sqrt(np.maximum(u_values, 0.0))
    return frame


def _hash_digest(hashes: dict[str, tuple[str, ...]], m: int) -> str:
    digest = hashlib.sha256()
    for ticker in sorted(hashes):
        digest.update(ticker.encode())
        digest.update(b"\0")
        digest.update("\n".join(hashes[ticker][:m]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_common_hashes(
    common_hashes: dict[str, dict[str, tuple[str, ...]]], m: int
) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Require identical ticker coverage and URL-hash prefixes across models."""
    if len(common_hashes) != _ENCODER_COUNT or set(common_hashes) != set(_MODEL_DIRS):
        message = "common comparison must contain all four locked encoders"
        raise WassersteinInputError(message)
    reference_model = next(iter(_MODEL_DIRS))
    reference_hashes = common_hashes[reference_model]
    for model, hashes in common_hashes.items():
        if set(hashes) != set(reference_hashes):
            message = f"{model} ticker coverage differs at common m={m}"
            raise WassersteinInputError(message)
        for ticker in hashes:
            if hashes[ticker] != reference_hashes[ticker]:
                message = f"url_hash prefix mismatch for {model}/{ticker} at m={m}"
                raise WassersteinInputError(message)
    return _hash_digest(reference_hashes, m), reference_hashes


def _validate_model_tickers(
    model: str, embedding_tickers: list[str], priced_tickers: list[str]
) -> None:
    """Require the governed embedding and priced universes to match the roster."""
    if len(embedding_tickers) != _EMBEDDING_UNIVERSE_SIZE:
        message = (
            f"expected {_EMBEDDING_UNIVERSE_SIZE} embedding tickers for "
            f"{model}; found {len(embedding_tickers)}"
        )
        raise WassersteinInputError(message)
    missing = sorted(set(priced_tickers) - set(embedding_tickers))
    if missing:
        message = f"{model} lacks priced tickers: {missing}"
        raise WassersteinInputError(message)
    excluded = sorted(set(embedding_tickers) - set(priced_tickers))
    if excluded != sorted(_EMBEDDING_ONLY_TICKERS):
        message = (
            f"embedding-only tickers must equal the declared exclusions "
            f"{sorted(_EMBEDDING_ONLY_TICKERS)}; found {excluded}"
        )
        raise WassersteinInputError(message)


def run_wasserstein_comparator(
    config: WassersteinComparatorConfig,
) -> dict[str, object]:
    """Run the controlled four-encoder and Qwen sample-size comparisons."""
    priced_tickers, return_dist = _load_return_distances(config.covariance_matrix)
    if len(priced_tickers) != _PRICED_UNIVERSE_SIZE:
        message = (
            f"expected {_PRICED_UNIVERSE_SIZE} priced tickers; "
            f"found {len(priced_tickers)}"
        )
        raise WassersteinInputError(message)
    all_pair_frames: list[pd.DataFrame] = []
    common_hashes: dict[str, dict[str, tuple[str, ...]]] = {}

    for model, dirname in _MODEL_DIRS.items():
        maximum = max(_SENSITIVITY_M) if model == "qwen3-embedding-8b" else _COMMON_M
        hashes, samples = _load_model(config.embeddings_root / dirname, maximum)
        embedding_tickers = sorted(samples)
        _validate_model_tickers(model, embedding_tickers, priced_tickers)
        common_hashes[model] = {ticker: hashes[ticker][:_COMMON_M] for ticker in hashes}
        typed_wasserstein = _load_typed_wasserstein_matrix(
            config.typed_distance_root, model, priced_tickers
        )
        typed_wasserstein2 = _load_typed_wasserstein_matrix(
            config.typed_distance_root, model, priced_tickers, "wasserstein_w2"
        )

        m_values = _SENSITIVITY_M if model == "qwen3-embedding-8b" else (_COMMON_M,)
        for m in m_values:
            frame = _pair_metrics(
                samples,
                priced_tickers,
                return_dist,
                model,
                m,
                transport=(
                    TransportMatrices(
                        order_one=typed_wasserstein, order_two=typed_wasserstein2
                    )
                    if m == _COMMON_M
                    else None
                ),
            )
            all_pair_frames.append(frame)

    _validate_common_hashes(common_hashes, _COMMON_M)

    combined = pd.concat(all_pair_frames, ignore_index=True)
    config.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(config.output_pairs, index=False)
    return {
        "common_m": _COMMON_M,
        "priced_universe": _PRICED_UNIVERSE_SIZE,
        "n_priced_dyads": _PRICED_UNIVERSE_SIZE * (_PRICED_UNIVERSE_SIZE - 1) // 2,
        "pair_artifact": str(config.output_pairs),
    }
