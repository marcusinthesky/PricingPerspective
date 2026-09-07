"""Energy test computation using JAX (jcor).

Provides:
- ``compute_energy_tests`` — compute pairwise energy test statistics across
  ticker embedding distributions using
  ``jcor.discrepancy.energy.energy_distance_matrix``
  (batched, jit+vmap) for tickers sharing a sample count, falling back to
  the scalar ``jcor.energy_distance`` for the rare cross-group pair.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, NoReturn, TypedDict

import duckdb
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from jcor.discrepancy.energy import energy_distance, energy_distance_matrix
from tqdm import tqdm

from pipeline.jax_cache import configure_persistent_cache

if TYPE_CHECKING:
    from pathlib import Path

configure_persistent_cache()

logger = logging.getLogger(__name__)
MIN_TICKERS = 2
SYMBOL_PREVIEW_LIMIT = 10
# Named rather than written inline because PLR2004 (magic-value-comparison) is
# live under `select = ["ALL"]` (src/python/pyproject.toml:34). Until t80 this
# was a deliberate second definition of a constant the Paper 1 neighbour stage
# also carried, justified by apps/pipeline/ARCHITECTURE.md:106-107 pointing
# imports strictly downward (`_kernels/` <- `io/` <- `stages/`), "never a
# sibling or ancestor". That stage now reads the shared geometry component and
# loads no cloud of its own, so the duplicate is gone and this is the only
# definition left.
EMBEDDING_NDIM = 2


class MatryoshkaDimensionError(ValueError):
    """Raised when a requested truncation exceeds an embedding's width."""

    def __init__(self, requested: int, width: int, path: Path) -> None:
        """Describe the invalid requested dimension and source artifact."""
        super().__init__(
            f"matryoshka_dim={requested} exceeds embedding width {width} in {path}"
        )


class EmbeddingShapeError(ValueError):
    """Raised when an embedding block is not a nonempty 2-D matrix."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Describe the rejected block shape.

        Args:
            shape: The offending array shape.

        """
        super().__init__(f"embeddings must be a nonempty 2-D block, got shape {shape}")


class NonFiniteEmbeddingError(ValueError):
    """Raised when an embedding block holds NaN or infinite entries."""

    def __init__(self, offending: int, total: int) -> None:
        """Describe how many rows carry a non-finite entry.

        Args:
            offending: Rows holding at least one NaN or infinite entry.
            total: Rows inspected.

        """
        super().__init__(
            f"{offending} of {total} embedding rows contain NaN or infinity; "
            "normalization is undefined there"
        )


class ZeroNormRowError(ValueError):
    """Raised when a row's L2 norm is zero or non-finite.

    Named for a zero *row*, not a zero pooled mean: jcor keeps those two facts
    apart on purpose (``jcor/sample/evidence.py`` lines 33-42 — the antipodal
    cloud ``{x, -x}`` has unit rows and pools to exactly zero), so this must not
    read as a twin of ``jcor.discrepancy._mmd.contracts.ZeroNormMeanEmbeddingError``.
    """

    def __init__(self, offending: int, total: int) -> None:
        """Describe how many rows cannot be scaled to unit length.

        Args:
            offending: Rows whose L2 norm is zero or non-finite.
            total: Rows inspected.

        """
        super().__init__(
            f"{offending} of {total} embedding rows have zero or non-finite "
            "norm; repair or drop them upstream, this stage does not "
            "substitute a norm of one"
        )


class EnergyResult(TypedDict):
    """One pairwise energy-distance test between two symbols' article embeddings.

    The row shape written to ``energy_tests.parquet``: ``test_statistic`` is the
    alpha-scaled energy statistic and ``energy_distance`` the raw functional.
    """

    symbol1: str
    symbol2: str
    test_statistic: float
    energy_distance: float
    n_samples1: int
    n_samples2: int


@partial(jax.jit, static_argnames=("alpha",))
def _batched_energy_distance(
    left: jnp.ndarray, right: jnp.ndarray, alpha: float
) -> jnp.ndarray:
    """Evaluate independent, same-shaped scalar energy-distance pairs together."""
    return jax.vmap(
        lambda embeddings1, embeddings2: energy_distance(
            embeddings1, embeddings2, exponent=alpha, metric="euclidean"
        )
    )(left, right)


def _load_embeddings(
    parquet_path: Path,
    matryoshka_dim: int | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> jnp.ndarray:
    """Load embeddings from a parquet file.

    When ``matryoshka_dim`` is set, the leading ``matryoshka_dim`` coordinates
    are kept (Matryoshka truncation) BEFORE any downstream renormalization.
    ``None`` returns the full-width embedding unchanged.

    ``con`` lets callers reuse one DuckDB connection across many files
    (opening a fresh in-memory connection per file is measurably slower).
    When omitted, a throwaway connection is opened for this call only.

    The DuckDB result is routed through ``numpy`` before ``jnp.asarray``:
    calling ``jnp.array`` directly on the nested Python list from
    ``fetchall()`` is ~20x slower than letting numpy build the contiguous
    buffer first (numpy's C-level list conversion vs. JAX's per-element
    tracing path for Python objects). Embeddings don't need float64
    precision, so the numpy buffer is built as float32 directly.
    """
    query = "SELECT embedding FROM read_parquet(?)"
    parameters = [str(parquet_path)]
    if con is not None:
        embeddings_list = con.execute(query, parameters).fetchall()
    else:
        with duckdb.connect(":memory:") as owned_con:
            embeddings_list = owned_con.execute(query, parameters).fetchall()
    embeddings = jnp.asarray(
        np.array([emb[0] for emb in embeddings_list], dtype=np.float32)
    )
    if matryoshka_dim is not None:
        if matryoshka_dim > embeddings.shape[1]:
            raise MatryoshkaDimensionError(
                matryoshka_dim, embeddings.shape[1], parquet_path
            )
        embeddings = embeddings[:, :matryoshka_dim]
    return embeddings


def _normalize_embeddings(embeddings: jnp.ndarray) -> jnp.ndarray:
    """Scale every row to unit L2 norm, rejecting rows where that is undefined.

    An eager host door in the sense of ``jcor/sample/evidence.py`` lines 44-51
    ("Doors reject; they never repair"), which names *this* function by path as
    its counter-example. It used to write ``jnp.where(norms == 0, 1, norms)``, so
    a zero row was divided by one and came back unchanged behind a docstring
    promising unit output. Two further leaks rode along with that guard:
    ``norms == 0`` is False for ``NaN``, so a NaN row propagated into every
    energy distance it appeared in, and ``x / inf`` silently mapped an
    overflowing row to zeros. All three now raise. Picking a substitute row or a
    drop policy is the caller's decision to own and record, exactly as
    ``_kernels/typed_geometry.py::apply_representation`` leaves it; the
    rejection set below mirrors that door on the jnp/float32 path. (The
    counter-example this paragraph used to cite, a hand-rolled normalizer in
    the Paper 1 neighbour stage, was retired in t80 along with that stage's
    private energy implementation.)

    The zero row is a live mechanism, not a hypothetical: the Matryoshka
    truncation directly above (lines 168-173) keeps only the leading
    ``matryoshka_dim`` coordinates, so a row that was nonzero at full width can
    be exactly zero afterwards, and nothing between that slice and this call
    inspects it. ``substrate/embeddings.py:707,714`` does gate on
    ``zero_count == 0``, but it is a DAG sibling of ``energy_tests`` (both hang
    off ``${shared}/embeddings/${model}``, nothing orders it first) and it reads
    full-width float64, i.e. before any truncation.

    Both call sites are eager -- :func:`_load_valid_embeddings` below and
    legacy embedding consumers are plain Python loops
    over parquet files -- so the ``bool()`` calls here are safe. The only
    compiled boundary, :func:`_batched_energy_distance`, is strictly downstream
    and receives the already-normalized concrete array; under ``jit``/``vmap``
    these reductions would raise ``TracerBoolConversionError`` instead.

    On a block with no violating row the result is bit-identical to the previous
    implementation: with no zero norm, ``jnp.where`` returned ``norms``
    elementwise, and the division is untouched.

    Args:
        embeddings: A nonempty ``(n, d)`` block of raw embedding rows.

    Returns:
        The same block with every row scaled to unit L2 norm.

    Raises:
        EmbeddingShapeError: If the block is not a nonempty 2-D matrix.
        NonFiniteEmbeddingError: If any entry is NaN or infinite.
        ZeroNormRowError: If any row's L2 norm is zero or non-finite.

    """
    if embeddings.ndim != EMBEDDING_NDIM or 0 in embeddings.shape:
        raise EmbeddingShapeError(embeddings.shape)
    total = int(embeddings.shape[0])
    finite_rows = jnp.all(jnp.isfinite(embeddings), axis=1)
    if not bool(jnp.all(finite_rows)):
        raise NonFiniteEmbeddingError(int(jnp.count_nonzero(~finite_rows)), total)
    norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    # Finite entries do not imply a finite norm: `jnp.linalg.norm` computes
    # `sqrt(sum(x * conj(x)))` with no internal rescaling, so a float32 row of
    # 3e38 overflows to `inf` and `x / inf` is exactly the zero row this door
    # exists to reject. Witnessed by
    # `tests/substrate/test_energy_normalization.py::
    # test_row_whose_squared_norm_overflows_float32_is_rejected`, not reasoned.
    # `> 0` is the load-bearing test for the ordinary zero row and also retires
    # the old `== 0` float equality that RUF069 (extend-selected at
    # src/python/pyproject.toml:39) flags.
    usable = jnp.isfinite(norms) & (norms > 0)
    if not bool(jnp.all(usable)):
        raise ZeroNormRowError(int(jnp.count_nonzero(~usable)), total)
    return embeddings / norms


def _energy_result(
    symbol1: str, symbol2: str, n1: int, n2: int, distance: float
) -> EnergyResult:
    """Build one consistently scaled pairwise result record."""
    scale = (n1 * n2) / (n1 + n2)
    return {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "test_statistic": scale * distance,
        "energy_distance": distance,
        "n_samples1": n1,
        "n_samples2": n2,
    }


def _score_equal_sample_groups(
    groups: dict[int, list[str]],
    embeddings: dict[str, jnp.ndarray],
    alpha: float,
    progress: tqdm[NoReturn],
) -> dict[tuple[str, str], EnergyResult]:
    """Score equal-size ticker groups with the matrix kernel where possible."""
    results: dict[tuple[str, str], EnergyResult] = {}
    for sample_count, group_symbols in groups.items():
        ordered_symbols = sorted(group_symbols)
        if len(ordered_symbols) < MIN_TICKERS:
            continue
        stacked = [embeddings[symbol] for symbol in ordered_symbols]
        try:
            distances = np.asarray(
                energy_distance_matrix(
                    stacked, exponent=alpha, metric="euclidean"
                ).values
            )
        except Exception:
            logger.exception(
                "Batched energy computation failed for sample-count group %d (%s); "
                "falling back to per-pair",
                sample_count,
                ordered_symbols,
            )
            for symbol1, symbol2 in combinations(ordered_symbols, 2):
                _try_scalar_pair(symbol1, symbol2, embeddings, alpha, results)
                progress.update(1)
            continue
        for left_index, symbol1 in enumerate(ordered_symbols):
            for right_index in range(left_index + 1, len(ordered_symbols)):
                symbol2 = ordered_symbols[right_index]
                distance = float(distances[left_index, right_index])
                results[(symbol1, symbol2)] = _energy_result(
                    symbol1, symbol2, sample_count, sample_count, distance
                )
                progress.update(1)
    return results


def _score_remaining_pairs(
    valid_symbols: list[str],
    embeddings: dict[str, jnp.ndarray],
    alpha: float,
    progress: tqdm[NoReturn],
    results: dict[tuple[str, str], EnergyResult],
) -> None:
    """Score pairs absent from equal-size matrix batches."""
    cross_groups: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    scalar_pairs: list[tuple[str, str]] = []
    for symbol1, symbol2 in combinations(valid_symbols, 2):
        if (symbol1, symbol2) in results:
            continue
        shape = (embeddings[symbol1].shape[0], embeddings[symbol2].shape[0])
        target = scalar_pairs if shape[0] == shape[1] else cross_groups[shape]
        target.append((symbol1, symbol2))

    for symbol1, symbol2 in scalar_pairs:
        _try_scalar_pair(symbol1, symbol2, embeddings, alpha, results)
        progress.update(1)

    for shape, pairs in cross_groups.items():
        try:
            left = jnp.stack([embeddings[symbol1] for symbol1, _ in pairs])
            right = jnp.stack([embeddings[symbol2] for _, symbol2 in pairs])
            distances = np.asarray(_batched_energy_distance(left, right, alpha))
        except Exception:
            logger.exception(
                "Batched cross-group energy computation failed for sample-count "
                "shape %s (%s); falling back to per-pair",
                shape,
                pairs,
            )
            for symbol1, symbol2 in pairs:
                _try_scalar_pair(symbol1, symbol2, embeddings, alpha, results)
                progress.update(1)
            continue
        n1, n2 = shape
        for (symbol1, symbol2), distance in zip(pairs, distances, strict=True):
            results[(symbol1, symbol2)] = _energy_result(
                symbol1, symbol2, n1, n2, float(distance)
            )
            progress.update(1)


def _pairwise_energy_results(
    valid_symbols: list[str],
    embeddings_dict: dict[str, jnp.ndarray],
    alpha: float,
    pbar: tqdm[NoReturn],
) -> list[EnergyResult]:
    """Compute pairwise energy-test results for every ordered ticker pair.

    Tickers whose embeddings share an identical sample count are stacked and
    scored in one compiled
    :func:`jcor.discrepancy.energy.energy_distance_matrix` call (jit +
    vmap + ``lax.map`` over the whole group) instead of one JAX dispatch per
    pair. Remaining pairs are grouped by their ordered ``(n1, n2)`` sample
    shape and vmapped through the same scalar kernel. A failed group retains
    the scalar per-pair retry so one bad pair does not suppress valid output.

    This replaces the previous per-pair call to
    ``jcor.discrepancy.balancing.energy_test_kernel``, which solves a full simplex QP to
    pick optimal mixture *weights* over candidate distributions — overkill
    here, since every call only ever passed one candidate, for which the QP's
    answer is trivially weight=1.0. For a single candidate,
    ``energy_test_kernel`` is numerically equivalent (to QP tolerance) to
    ``scale * energy_distance(x, y)`` with ``scale = (n·m)/(n+m)``.

    The test statistic ``(n·m)/(n+m) · energy_distance`` is derived from the
    batched/scalar energy distance rather than requested directly, since both
    ``energy_distance_matrix`` and ``energy_distance`` only return the
    distance term.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for symbol in valid_symbols:
        groups[embeddings_dict[symbol].shape[0]].append(symbol)

    results = _score_equal_sample_groups(groups, embeddings_dict, alpha, pbar)
    _score_remaining_pairs(valid_symbols, embeddings_dict, alpha, pbar, results)

    return [results[pair] for pair in combinations(valid_symbols, 2) if pair in results]


def _try_scalar_pair(
    symbol1: str,
    symbol2: str,
    embeddings_dict: dict[str, jnp.ndarray],
    alpha: float,
    results: dict[tuple[str, str], EnergyResult],
) -> None:
    """Score one cross-group pair with the scalar energy_distance path."""
    embeddings1 = embeddings_dict[symbol1]
    embeddings2 = embeddings_dict[symbol2]
    n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
    try:
        dist = float(
            energy_distance(
                embeddings1, embeddings2, exponent=alpha, metric="euclidean"
            )
        )
    except Exception:
        logger.exception("Scalar energy computation failed for %s-%s", symbol1, symbol2)
        return

    results[(symbol1, symbol2)] = _energy_result(symbol1, symbol2, n1, n2, dist)


def _load_valid_embeddings(
    embeddings_dir: Path,
    symbols: list[str],
    min_samples: int,
    matryoshka_dim: int | None,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    """Load, normalize, and filter discovered ticker embedding artifacts."""
    embeddings_by_symbol: dict[str, jnp.ndarray] = {}
    skipped_symbols: list[str] = []
    with duckdb.connect(":memory:") as connection:
        for symbol in tqdm(symbols, desc="Loading embeddings"):
            parquet_path = embeddings_dir / f"{symbol}.parquet"
            try:
                embeddings = _load_embeddings(
                    parquet_path, matryoshka_dim=matryoshka_dim, con=connection
                )
            except Exception:
                logger.exception(
                    "Failed to load embeddings for %s from %s", symbol, parquet_path
                )
                skipped_symbols.append(symbol)
                continue
            # Deliberately OUTSIDE the `try`, and the message above no longer
            # says "or normalize". An unreadable or missing parquet is a
            # per-symbol load failure the run survives by skipping the ticker; a
            # zero or non-finite row is a corpus-integrity violation. Catching it
            # here would drop that ticker -- and every pair it appears in --
            # from energy_tests.parquet under a zero exit code, a coarser silent
            # repair than the one the door itself just stopped doing.
            embeddings = _normalize_embeddings(embeddings)
            if len(embeddings) < min_samples:
                typer.echo(
                    f"Warning: {symbol} has only {len(embeddings)} samples "
                    f"(min: {min_samples})",
                    err=True,
                )
                skipped_symbols.append(symbol)
                continue
            embeddings_by_symbol[symbol] = embeddings
            typer.echo(f"Loaded {symbol}: {embeddings.shape}")
    return embeddings_by_symbol, skipped_symbols


def _write_energy_results(output_file: Path, results: list[EnergyResult]) -> None:
    """Persist pairwise results with a stable Arrow schema."""
    schema = pa.schema(
        [
            ("symbol1", pa.string()),
            ("symbol2", pa.string()),
            ("test_statistic", pa.float64()),
            ("energy_distance", pa.float64()),
            ("n_samples1", pa.int64()),
            ("n_samples2", pa.int64()),
        ]
    )
    data = {
        field: [row[field] for row in results]
        for field in (
            "symbol1",
            "symbol2",
            "test_statistic",
            "energy_distance",
            "n_samples1",
            "n_samples2",
        )
    }
    pq.write_table(pa.table(data, schema=schema), output_file)


def compute_energy_tests(
    embeddings_dir: Path,
    output_file: Path,
    min_samples: int = 10,
    alpha: float = 1.0,
    matryoshka_dim: int | None = None,
) -> None:
    """Compute energy test for all combinations of ticker embeddings.

    Uses ``jcor.discrepancy.energy.energy_distance_matrix`` (batched jit+vmap)
    for tickers sharing a sample count and a shape-grouped ``jax.vmap`` path
    for irregular pairs. A failed irregular group falls back to scalar, jit-compiled
    ``jcor.energy_distance`` calls. For each pair of tickers, computes the
    energy test statistic and energy distance.

    Auto-discovers tickers from the embeddings directory (all ``*.parquet``).

    When ``matryoshka_dim`` is set, embeddings are Matryoshka-truncated to the
    leading ``matryoshka_dim`` coordinates and THEN L2-renormalized (truncate
    -then-renormalize). ``None`` uses the full embedding width and reproduces
    the canonical (untruncated) output.
    """
    typer.echo(f"Auto-discovering tickers from {embeddings_dir}")

    if not embeddings_dir.exists():
        typer.echo(f"Error: Embeddings directory not found: {embeddings_dir}", err=True)
        raise typer.Exit(code=1)

    parquet_files = list(embeddings_dir.glob("*.parquet"))
    if not parquet_files:
        typer.echo(f"Error: No parquet files found in {embeddings_dir}", err=True)
        raise typer.Exit(code=1)

    symbols = sorted(file.stem for file in parquet_files)
    preview = symbols[:SYMBOL_PREVIEW_LIMIT]
    suffix = "..." if len(symbols) > SYMBOL_PREVIEW_LIMIT else ""
    typer.echo(f"Found {len(symbols)} tickers: {preview}{suffix}")
    typer.echo(f"Using distance exponent α = {alpha}")
    if matryoshka_dim is not None:
        typer.echo(
            f"Matryoshka truncation to dim = {matryoshka_dim} (then renormalize)"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading embeddings...")
    embeddings_dict, skipped_symbols = _load_valid_embeddings(
        embeddings_dir, symbols, min_samples, matryoshka_dim
    )

    if len(embeddings_dict) < MIN_TICKERS:
        typer.echo(
            f"Error: Need at least {MIN_TICKERS} tickers with valid embeddings",
            err=True,
        )
        raise typer.Exit(code=1)

    valid_symbols = sorted(embeddings_dict.keys())
    typer.echo(f"\nComputing energy test for {len(valid_symbols)} tickers")
    typer.echo(f"Skipped tickers: {skipped_symbols}")

    total_pairs = len(list(combinations(valid_symbols, 2)))

    typer.echo("Note: First computation may be slow due to JIT compilation...")

    with tqdm(total=total_pairs, desc="Computing energy tests") as pbar:
        results = _pairwise_energy_results(valid_symbols, embeddings_dict, alpha, pbar)

    if not results:
        typer.echo("No results generated", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nSaving {len(results)} results to {output_file}")

    _write_energy_results(output_file, results)

    typer.echo(f"\nComputed {len(results)} energy tests (GPU-accelerated with JAX)")
    typer.echo(
        f"Test statistic range: [{min(r['test_statistic'] for r in results):.4f}, "
        f"{max(r['test_statistic'] for r in results):.4f}]"
    )
    typer.echo(
        f"Energy distance range: [{min(r['energy_distance'] for r in results):.4f}, "
        f"{max(r['energy_distance'] for r in results):.4f}]"
    )
    typer.echo("Test statistic = (n·m)/(n+m) · energy_distance")
