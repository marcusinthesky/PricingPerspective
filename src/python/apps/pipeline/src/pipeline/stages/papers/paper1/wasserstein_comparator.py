"""Paper 1's controlled cross-encoder distributional comparator.

The common-width Wasserstein W1 and W2 matrices are read from the shared
typed-distance backplane. Matched V-energy, signed-U, bootstrap, sensitivity,
and same-firm noise-floor diagnostics are computed here on deterministic
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
from scipy.stats import pearsonr, spearmanr

from pipeline._kernels.typed_distances import (
    EmbeddingCloud,
    compute_statistical_distance,
)
from pipeline._kernels.wasserstein import (
    chord_cost_matrix,
)
from pipeline.io.typed_analysis import PointDistanceSpec, StatisticalDistanceSpec
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from pathlib import Path

_COMMON_M = 128
# The frozen Qwen3-Embedding-8B pool contains 128 rows per firm. Keep the
# sensitivity within that support, and require two nonempty halves for the
# same-firm split floor.
_SENSITIVITY_M = (32, 64, 128)
_FLOOR_M = (32, 64)
_DEFAULT_BOOTSTRAP_ITERS = 999
_DEFAULT_BOOTSTRAP_SEED = 20260727
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
_MIN_CORRELATION_PAIRS = 2
_MODEL_DIRS = {
    "qwen3-embedding-4b": "qwen3-embedding-4b",
    "qwen3-embedding-8b": "qwen3-embedding-8b",
    "bge-m3": "bge-m3",
    "bge-large-en-v1.5": "bge-large-en-v1.5",
}


class WassersteinInputError(ValueError):
    """Report invalid comparator inputs or incompatible artifacts."""


class WassersteinComputationError(RuntimeError):
    """Report a failed comparator bootstrap or orchestration invariant."""


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
class RunSummaryOptions:
    """Collect metadata and bootstrap controls for one comparator run."""

    native_dim: int
    n_embedding_tickers: int
    n_boot: int
    seed: int


@dataclass(frozen=True)
class WassersteinComparatorConfig:
    """Paths and bootstrap controls for the encoder comparator."""

    embeddings_root: Path
    typed_distance_root: Path
    covariance_matrix: Path
    output_summary: Path
    output_pairs: Path
    bootstrap_iters: int = _DEFAULT_BOOTSTRAP_ITERS
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED


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
    :func:`_pair_metrics` passes JCOR's unit-row validation, and this per-firm
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
    # JCOR. The operation is idempotent for artifacts normalized by `_load_prefix`.
    normalized = {ticker: _unit_rows(samples[ticker][:m]) for ticker in tickers}
    # Every ticker's sample passes JCOR's eager unit-row validation here, once,
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


def _matrix_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    target = frame["return_chord"].to_numpy(dtype=np.float64)
    result: dict[str, dict[str, float]] = {}
    metrics = [
        "wasserstein_w1",
        "wasserstein_w2",
        "energy_v",
        "energy_u_functional",
    ]
    if "energy_u_sqrt" in frame:
        metrics.append("energy_u_sqrt")
    for metric in metrics:
        values = frame[metric].to_numpy(dtype=np.float64)
        result[metric] = {
            "pearson": float(pearsonr(values, target).statistic),
            "spearman": float(spearmanr(values, target).statistic),
        }
    return result


def _vector_to_matrix(
    frame: pd.DataFrame, tickers: list[str], column: str
) -> np.ndarray:
    lookup = {ticker: pos for pos, ticker in enumerate(tickers)}
    matrix = np.zeros((len(tickers), len(tickers)), dtype=np.float64)
    for row in frame[["ticker_i", "ticker_j", column]].itertuples(index=False):
        i, j = lookup[str(row.ticker_i)], lookup[str(row.ticker_j)]
        value = float(str(row[2]))
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def _distinct_original_pairs(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return resampled endpoint indices, excluding duplicate-original dyads."""
    tri_i, tri_j = np.triu_indices(len(idx), k=1)
    distinct = idx[tri_i] != idx[tri_j]
    return idx[tri_i][distinct], idx[tri_j][distinct]


def _paired_node_bootstrap(
    frame: pd.DataFrame,
    tickers: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Paired node bootstrap gaps versus V-energy matrix correlation."""
    columns = (
        "wasserstein_w1",
        "wasserstein_w2",
        "energy_u_functional",
        "energy_v",
        "return_chord",
    )
    matrix_columns = list(columns)
    if "energy_u_sqrt" in frame:
        matrix_columns.append("energy_u_sqrt")
    matrices = {
        name: _vector_to_matrix(frame, tickers, name) for name in matrix_columns
    }
    n = len(tickers)
    rng = np.random.default_rng(seed)
    comparators = [
        "wasserstein_w1",
        "wasserstein_w2",
        "energy_u_functional",
    ]
    if "energy_u_sqrt" in frame:
        comparators.append("energy_u_sqrt")
    draws: dict[str, dict[str, list[float]]] = {
        comparator: {"pearson": [], "spearman": []} for comparator in comparators
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows, cols = _distinct_original_pairs(idx)
        target = matrices["return_chord"][rows, cols]
        reference = matrices["energy_v"][rows, cols]
        if (
            target.size < _MIN_CORRELATION_PAIRS
            or np.std(target) <= 0.0
            or np.std(reference) <= 0.0
        ):
            continue
        reference_corr = {
            "pearson": float(pearsonr(reference, target).statistic),
            "spearman": float(spearmanr(reference, target).statistic),
        }
        for comparator in comparators:
            values = matrices[comparator][rows, cols]
            if np.std(values) <= 0.0:
                continue
            draws[comparator]["pearson"].append(
                float(pearsonr(values, target).statistic) - reference_corr["pearson"]
            )
            draws[comparator]["spearman"].append(
                float(spearmanr(values, target).statistic) - reference_corr["spearman"]
            )

    observed = _matrix_statistics(frame)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for comparator in comparators:
        output[comparator] = {}
        for method in ("pearson", "spearman"):
            values = np.asarray(draws[comparator][method], dtype=np.float64)
            if values.size == 0:
                message = "paired node bootstrap produced no valid draws"
                raise WassersteinComputationError(message)
            output[comparator][method] = {
                "observed_gap_vs_energy_v": (
                    observed[comparator][method] - observed["energy_v"][method]
                ),
                "bootstrap_mean_gap": float(values.mean()),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "probability_le_zero": float(np.mean(values <= 0.0)),
                "n_boot": int(values.size),
            }
    return output


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


def _same_firm_split_values(
    samples: dict[str, np.ndarray], m: int, distance: StatisticalDistanceSpec
) -> np.ndarray:
    """Return each firm's own-sample split distance under one transport order."""
    return np.asarray(
        [
            _typed_wasserstein_matrix(
                {"left": sample[:m], "right": sample[m : 2 * m]},
                ["left", "right"],
                m,
                distance,
            )[0, 1]
            for sample in samples.values()
        ],
        dtype=np.float64,
    )


def _same_firm_floor(
    samples: dict[str, np.ndarray],
    m: int,
    cross_firm_w1_median: float,
    cross_firm_w2_median: float,
) -> dict[str, object]:
    """Report the finite-sample floor for both transport orders.

    The floor bounds what a matched-sample Wasserstein comparison can resolve at
    this ``m``, and that bound is order-specific: W2 charges the same transport
    quadratically, so its floor cannot be inferred from the W1 one.
    """
    values = _same_firm_split_values(samples, m, _WASSERSTEIN_DISTANCE)
    values_w2 = _same_firm_split_values(samples, m, _WASSERSTEIN2_DISTANCE)
    return {
        "m_per_half": m,
        "n_embedding_firms": int(values.size),
        "split": "first m versus next m after lexicographic url_hash sort",
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "cross_firm_w1_median": cross_firm_w1_median,
        "w2_mean": float(values_w2.mean()),
        "w2_median": float(np.median(values_w2)),
        "w2_p95": float(np.percentile(values_w2, 95.0)),
        "cross_firm_w2_median": cross_firm_w2_median,
        "use": "diagnostic floor only; not subtracted and not used for debiasing",
    }


def _run_summary(
    frame: pd.DataFrame,
    tickers: list[str],
    hashes: dict[str, tuple[str, ...]],
    options: RunSummaryOptions,
) -> dict[str, object]:
    model = str(frame["model"].iloc[0])
    m = int(frame["m"].iloc[0])
    u_values = frame["energy_u_functional"].to_numpy(dtype=np.float64)
    u_negative = int(np.sum(u_values < _FUNCTIONAL_NEGATIVE_TOLERANCE))
    return {
        "model": model,
        "native_dim": options.native_dim,
        "m": m,
        "n_embedding_tickers": options.n_embedding_tickers,
        "n_priced_tickers": len(tickers),
        "n_dyads": len(frame),
        "url_hash_prefix_sha256": _hash_digest(hashes, m),
        "u_functional_diagnostics": {
            "minimum": float(u_values.min()),
            "negative_count": u_negative,
            "sqrt_transform_available": u_negative == 0,
            "rule": (
                "sqrt(U) emitted only when every dyad is nonnegative within "
                "1e-10; signed U is never clipped or described as a metric"
            ),
        },
        "matrix_correlations": _matrix_statistics(frame),
        "paired_gaps": _paired_node_bootstrap(
            frame, tickers, options.n_boot, options.seed
        ),
    }


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
    common_runs: list[dict[str, object]] = []
    sensitivity_runs: list[dict[str, object]] = []
    common_hashes: dict[str, dict[str, tuple[str, ...]]] = {}
    qwen_samples: dict[str, np.ndarray] | None = None

    for model_index, (model, dirname) in enumerate(_MODEL_DIRS.items()):
        maximum = max(_SENSITIVITY_M) if model == "qwen3-embedding-8b" else _COMMON_M
        hashes, samples = _load_model(config.embeddings_root / dirname, maximum)
        embedding_tickers = sorted(samples)
        _validate_model_tickers(model, embedding_tickers, priced_tickers)
        common_hashes[model] = {ticker: hashes[ticker][:_COMMON_M] for ticker in hashes}
        native_dim = int(next(iter(samples.values())).shape[1])
        typed_wasserstein = _load_typed_wasserstein_matrix(
            config.typed_distance_root, model, priced_tickers
        )
        typed_wasserstein2 = _load_typed_wasserstein_matrix(
            config.typed_distance_root, model, priced_tickers, "wasserstein_w2"
        )

        m_values = _SENSITIVITY_M if model == "qwen3-embedding-8b" else (_COMMON_M,)
        for run_index, m in enumerate(m_values):
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
            run = _run_summary(
                frame,
                priced_tickers,
                hashes,
                RunSummaryOptions(
                    native_dim=native_dim,
                    n_embedding_tickers=len(embedding_tickers),
                    n_boot=config.bootstrap_iters,
                    seed=config.bootstrap_seed + 100 * model_index + run_index,
                ),
            )
            if m == _COMMON_M:
                common_runs.append(run)
            if model == "qwen3-embedding-8b":
                sensitivity_runs.append(run)
        if model == "qwen3-embedding-8b":
            qwen_samples = samples

    common_digest, _reference_hashes = _validate_common_hashes(common_hashes, _COMMON_M)

    if qwen_samples is None:
        message = "Qwen3-Embedding-8B samples were not loaded"
        raise WassersteinComputationError(message)
    qwen_frames = {
        int(frame["m"].iloc[0]): frame
        for frame in all_pair_frames
        if str(frame["model"].iloc[0]) == "qwen3-embedding-8b"
    }
    floors = [
        _same_firm_floor(
            qwen_samples,
            m,
            float(qwen_frames[m]["wasserstein_w1"].median()),
            float(qwen_frames[m]["wasserstein_w2"].median()),
        )
        for m in _FLOOR_M
    ]

    combined = pd.concat(all_pair_frames, ignore_index=True)
    config.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(config.output_pairs, index=False)
    result: dict[str, object] = {
        "estimator": (
            "exact empirical balanced Wasserstein-1 and Wasserstein-2; uniform "
            "equal cardinality. The two solve different assignment problems: W2 "
            "matches on the squared chord cost, so its matrix is not a transform "
            "of the W1 one"
        ),
        "ground_geometry": {
            "name": "Euclidean chord distance on row-L2-normalized embeddings",
            "role": (
                "harmonized inference geometry across complete encoder/input pipelines"
            ),
            "native_loss_claim": "none; the encoders need not share a training loss",
            "relation": "chord=sqrt(2-2*cos(theta))=2*sin(theta/2); angular=theta",
            "cosine_dissimilarity": "1-cos(theta)=chord^2/2 and is not a metric",
            "theorem_selection_claim": "none",
        },
        "selection": "deterministic lexicographic url_hash prefix",
        # Derived, not restated: these drifted from the constants above when the
        # roster changed, so the manifest reported a universe the run had not
        # used.
        "embedding_universe": _EMBEDDING_UNIVERSE_SIZE,
        "priced_universe": _PRICED_UNIVERSE_SIZE,
        "n_priced_dyads": _PRICED_UNIVERSE_SIZE * (_PRICED_UNIVERSE_SIZE - 1) // 2,
        "return_alignment_exclusion": None,
        "common_m": _COMMON_M,
        "common_url_hashes_equal": True,
        "common_prefix_sha256": common_digest,
        "common_native_width_runs": common_runs,
        "qwen8b_sample_size_sensitivity": sensitivity_runs,
        "qwen8b_same_firm_split_floor": floors,
        "bootstrap": {
            "unit": "firm/node",
            "requested_draws": config.bootstrap_iters,
            "seed": config.bootstrap_seed,
            "duplicate_original_self_dyads": "excluded",
            "pairing": "all metric correlations recomputed on the same resample",
        },
        "pair_artifact": str(config.output_pairs),
    }
    config.output_summary.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary.write_text(
        yaml.safe_dump(result, sort_keys=False), encoding="utf-8"
    )
    return result
