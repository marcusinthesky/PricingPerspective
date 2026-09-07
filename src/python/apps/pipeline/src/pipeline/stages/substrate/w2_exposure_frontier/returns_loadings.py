"""Returns-based factor loadings for the H1 identification frontier (T1).

Replaces the F1 tautology (numerator and denominator both derived from the
same text embeddings, see ``pipeline.h1_pilot``) with a genuinely independent
numerator: whitened PCA loadings estimated from the **daily return panel**
(``data/shared/returns/<TICKER>.parquet``). See
``.context/plan/2026-07-11_h1-returns-frontier/README.md`` Requirements 1-2.

Whitened PCA construction
--------------------------
Given the sample return covariance ``Sigma = V @ diag(lambda) @ V.T`` (top-``K``
eigenpairs, descending eigenvalues), the per-ticker loading is

    beta_hat[i] = (sqrt(lambda_1) * V[i, 1], ..., sqrt(lambda_K) * V[i, K])

so that ``<beta_hat_i, beta_hat_j> = Sigma^(K)_ij`` (the rank-``K`` truncated
/ systematic covariance) and ``||beta_hat_i||^2 = Sigma^(K)_ii`` exactly, by
construction of the eigendecomposition -- this is the algebraic identity the
T1 unit test checks to 1e-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import duckdb
import numpy as np
import pandas as pd

from pipeline.stages.substrate.panel import read_seam_return

if TYPE_CHECKING:
    from pathlib import Path


def load_return_panel(
    returns_dir: Path,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    oos_returns_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the aligned daily-return panel for ``tickers`` over ``[start, end]``.

    Each ``<TICKER>.parquet`` under ``returns_dir`` has columns ``Date``,
    ``ticker``, ``return``. The in-sample panel's first row has a ``NaN``
    return (no prior close); the OOS panel's first row instead carries the
    seam return against the last in-sample close. Dates are inner-joined
    across all requested
    tickers (a handful of per-ticker gaps exist, e.g. late listings) so the
    returned panel is a dense ``(T, N)`` matrix with no missing values, and
    ``N == len(tickers)`` in the given order.

    Args:
        returns_dir: Directory of per-ticker return parquets (in-sample).
        tickers: Ticker order for the returned matrix's columns. Every
            ticker must have a ``<ticker>.parquet`` file in ``returns_dir``
            (raises ``FileNotFoundError`` otherwise -- callers should
            pre-filter to the priced intersection before calling this).
        start: Optional inclusive ISO date lower bound (e.g. ``"2018-01-01"``).
        end: Optional inclusive ISO date upper bound (e.g. ``"2021-12-31"``).
        oos_returns_dir: Optional second directory of per-ticker return
            parquets (e.g. ``data/shared/oos/returns``) to union with
            ``returns_dir`` per ticker, keyed on the ``Date`` column, before
            applying ``start``/``end``. ``None`` (default) preserves the
            single-directory behavior exactly. When given, every ticker must
            also have a ``<ticker>.parquet`` file under ``oos_returns_dir``.

    Returns:
        Tuple ``(dates, returns)`` where ``dates`` is a ``(T,)`` array of
        ``datetime64[us]`` values (ascending, common to all tickers) and
        ``returns`` is the ``(T, N)`` float64 return matrix in ``tickers``
        order, with the leading all-``NaN`` row dropped.

    """
    con = duckdb.connect(":memory:")
    frames = []
    for ticker in tickers:
        path = returns_dir / f"{ticker}.parquet"
        if not path.exists():
            message = (
                f"No return panel for ticker {ticker!r} at {path}; "
                "pre-filter to the priced intersection before calling "
                "load_return_panel."
            )
            raise FileNotFoundError(message)
        if oos_returns_dir is None:
            df = con.execute(
                """
                    SELECT Date, return
                    FROM read_parquet(?)
                    ORDER BY Date
                """,
                [str(path)],
            ).df()
        else:
            oos_path = oos_returns_dir / f"{ticker}.parquet"
            if not oos_path.exists():
                message = (
                    f"No OOS return panel for ticker {ticker!r} at {oos_path}; "
                    "every in-sample ticker must also have an OOS parquet "
                    "when oos_returns_dir is given."
                )
                raise FileNotFoundError(message)
            # The seam is continuous without a close price: the OOS panel's
            # leading row stores the return measured against the last
            # in-sample close, rather than the null a per-file pct_change
            # would leave there. The vendor close series is withheld and no
            # longer carried by these panels -- see
            # `scripts/strip_close_price`.
            df = con.execute(
                """
                    SELECT Date, return
                    FROM read_parquet(?)
                    ORDER BY Date
                """,
                [str(path)],
            ).df()
            oos_df = con.execute(
                """
                    SELECT Date, return
                    FROM read_parquet(?)
                    ORDER BY Date
                """,
                [str(oos_path)],
            ).df()
            seam = read_seam_return(oos_path)
            if seam is not None and not oos_df.empty:
                oos_df.loc[oos_df.index[0], "return"] = seam
            # De-duplicate on the Date key -- the in-sample frame wins any
            # seam-date overlap (there should be none in practice).
            df = (
                pd.concat([df, oos_df], ignore_index=True)
                .drop_duplicates(subset="Date", keep="first")
                .sort_values("Date")
                .reset_index(drop=True)
            )

        if start:
            df = df[df["Date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["Date"] <= pd.Timestamp(end)]

        df = df.rename(columns={"return": ticker}).set_index("Date")
        frames.append(df)
    con.close()

    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.join(frame, how="inner")
    panel = panel.dropna(how="any")
    panel = panel[tickers]

    return panel.index.to_numpy(), panel.to_numpy(dtype=np.float64)


class LoadingsResult(NamedTuple):
    """Whitened PCA loadings and diagnostics for one ``K``.

    Attributes:
        betas: ``(N, K)`` array, row ``i`` is ``beta_hat_i`` (§module doc).
        eigenvalues: ``(K,)`` descending eigenvalues of the sample covariance
            (the retained systematic variance per factor).
        explained_variance_ratio: Fraction of the total covariance trace
            explained by the top ``K`` factors (``sum(eigenvalues[:K]) /
            trace(Sigma)``).
        sigma: ``(N, N)`` full sample covariance (``np.cov(returns, ddof=1)``),
            kept for the algebra unit test (``betas @ betas.T ==
            Sigma^(K)``, the rank-``K`` truncation of ``sigma``, not
            ``sigma`` itself).
        k: Number of retained factors.

    """

    betas: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: float
    sigma: np.ndarray
    k: int


@dataclass(frozen=True)
class _PcaSpectrum:
    """Immutable covariance eigensystem shared by a K sweep."""

    sigma: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    total_variance: float


def _compute_spectrum(returns: np.ndarray) -> _PcaSpectrum:
    """Compute covariance and its descending eigensystem exactly once."""
    sigma = np.cov(returns, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(eigvals)[::-1]
    ordered_values = eigvals[order]
    ordered_vectors = eigvecs[:, order]
    return _PcaSpectrum(
        sigma=sigma,
        eigenvalues=ordered_values,
        eigenvectors=ordered_vectors,
        total_variance=float(np.trace(sigma)),
    )


def _loadings_from_spectrum(spectrum: _PcaSpectrum, k: int) -> LoadingsResult:
    """Materialize one public loading result from a shared spectrum."""
    n = spectrum.sigma.shape[0]
    if not (1 <= k <= n):
        message = f"k={k} must be in [1, {n}]"
        raise ValueError(message)
    top_vals = np.clip(spectrum.eigenvalues[:k], 0.0, None)
    top_vecs = spectrum.eigenvectors[:, :k]
    betas = top_vecs * np.sqrt(top_vals)[None, :]
    explained = (
        float(top_vals.sum() / spectrum.total_variance)
        if spectrum.total_variance > 0.0
        else float("nan")
    )
    return LoadingsResult(
        betas=betas,
        eigenvalues=top_vals,
        explained_variance_ratio=explained,
        sigma=spectrum.sigma,
        k=k,
    )


def estimate_loadings(returns: np.ndarray, k: int) -> LoadingsResult:
    """Whitened PCA loadings from a ``(T, N)`` return panel.

    Args:
        returns: ``(T, N)`` array of daily returns (no NaNs).
        k: Number of principal factors to retain (``k <= N``).

    Returns:
        A :class:`LoadingsResult`.

    """
    return _loadings_from_spectrum(_compute_spectrum(returns), k)


def sweep_k(
    returns: np.ndarray, ks: tuple[int, ...] = (3, 5, 10)
) -> dict[int, LoadingsResult]:
    """Estimate loadings for each ``K`` in ``ks``; report explained variance.

    Args:
        returns: ``(T, N)`` array of daily returns (no NaNs).
        ks: Candidate factor counts to sweep.

    Returns:
        Mapping ``{k: LoadingsResult}``.

    """
    spectrum = _compute_spectrum(returns)
    return {k: _loadings_from_spectrum(spectrum, k) for k in ks}


def pick_k(results: dict[int, LoadingsResult], target: float = 0.80) -> int:
    """Pick the smallest swept ``K`` reaching ``target`` explained variance.

    Falls back to the largest swept ``K`` if none reaches ``target`` (the
    caller should disclose this as a degeneracy/coverage note, per the plan's
    kill-condition confounds section).

    Args:
        results: Output of :func:`sweep_k`.
        target: Explained-variance-ratio threshold (default 0.80).

    Returns:
        The chosen ``K``.

    """
    for k in sorted(results):
        if results[k].explained_variance_ratio >= target:
            return k
    return max(results)
