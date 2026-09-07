r"""Paper 5 energy-radius calibration, misspecification, and frontier slack.

Builds the entrywise covariance brackets ``[Sigma_lo, Sigma_hi]`` that make
``Sigma_dist`` (Paper 3's point estimate) the *center* of a statistically
calibrated ambiguity set used by Paper 5's H3 arm.

DISCIPLINE (Lean-pinned, T1+T2, ``RobustEnergy.lean``): radii add on the
energy **metric** :math:`\\mathcal{E} = \\sqrt{\\mathcal{E}^2}`, never on the
squared distance directly -- squared brackets are asymmetric,
``[(E - eps)_+^2, (E + eps)^2]`` (T1) -- and the bi-Lipschitz envelope
``[ell, L]`` enters as the bracket on the distance-to-covariance MAP itself
(T2, ``covariance_bracket_of_embedding_radius``), never as a multiplier on
the sampling radius.

Three radius sources (plan Sec. "Radius calibration"):

1. ``bootstrap_stat_radius`` -- per-firm sampling error of the mean text
   embedding, from article-level bootstrap on the CALIBRATION epoch only.
   Composes via the triangle inequality: ``eps_ij = eps_i + eps_j`` (T1).
2. ``pair_gmm_radius`` -- per-pair misspecification radius from Paper 3's
   GMM moment-violation machinery (``pipeline.paper3_empirical``), estimated
   on the calibration epoch. Lives naturally on the squared (``D^2``, i.e.
   Sigma-unit) scale, not the metric scale -- composed as an ADDITIVE
   correction on ``D^2`` after the metric-level triangle-inequality
   composition (CALIBRATED mode only; the CERTIFIED mode's envelope already
   brackets every per-pair proportionality, so adding it there would double
   count -- see ``assemble_brackets``).
3. ``frontier_envelope`` -- shared W2-to-return-exposure frontier
   ``[ell_lo, L_hat]``, entering T2-faithfully as the interval of map
   constants ``[kappa_lo, kappa_hi]`` in ``assemble_brackets``: the lower
   covariance bound evaluates the polarization at ``L_hat*(E + eps)``, the
   upper at ``ell_lo*max(0, E - eps)``. It is NOT a radius multiplier.

``assemble_brackets`` composes these into ``[Sigma_lo, Sigma_hi]`` (called
once per mode: certified ``[ell_lo, L_hat]`` / calibrated ``kappa_hat``).
``bootstrap_ridge_radius`` calibrates the Frobenius-ball radius ``r(T)`` for
the ridge-robust intersection term. ``implied_kappa_envelope_check`` reports
how often the calibration data's own per-pair proportionality falls inside
the certified envelope (the no-double-counting diagnostic).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jcor.core.random import cell_key
from jcor.model.covariance import sample_covariance, sample_covariance_kernel
from jcor.operators.covariance import sigma_dist
from numpy.typing import NDArray
from simulation.radii import (
    bootstrap_stat_radius as _bootstrap_stat_radius_kernel,
)
from simulation.radii import stat_radius_matrix as _stat_radius_matrix_kernel

from pipeline._kernels.arrays import l2_normalize_rows
from pipeline.stages.papers.paper3.empirical import (
    _pair_moment_process,
    calibrate_kappa,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

Float = NDArray[np.float64]


def bootstrap_stat_radius(
    embeddings: Mapping[str, Float],
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Adapt heterogeneous labelled article panels to the array-only kernel."""
    radii: dict[str, float] = {}
    for ticker, observations in embeddings.items():
        key = cell_key(
            seed, "paper5_energy_robust_bootstrap_stat_radius", "ticker", ticker
        )
        value = _bootstrap_stat_radius_kernel(
            key,
            jnp.asarray(observations)[None, :, :],
            n_boot=n_boot,
        )[0]
        radii[ticker] = float(value)
    return radii


def stat_radius_matrix(radii: Mapping[Any, float], tickers: list[Any]) -> Float:
    """Convert labelled host radii to the array-only pairwise composition."""
    values = jnp.asarray([radii[ticker] for ticker in tickers], dtype=jnp.float64)
    return np.asarray(_stat_radius_matrix_kernel(values), dtype=np.float64)


@dataclass(frozen=True)
class CovarianceBracketInputs:
    """Point estimate, ambiguity radii, and map bounds for one bracket."""

    squared_distances: Float
    eps_stat_matrix: Float
    eps_gmm_d2: Float
    sigma_sq: Float
    kappa_lo: float
    kappa_hi: float


@dataclass(frozen=True)
class RidgeRadiusInputs:
    """Calibration panel, covariance center, and target window grid."""

    calibration_returns: Float
    sigma_center: Float
    windows: tuple[int, ...]


@dataclass(frozen=True)
class RidgeRadiusConfig:
    """Bootstrap controls for the ridge-radius calibration."""

    quantile: float = 0.9
    n_boot: int = 200
    seed: int = 0


# ---------------------------------------------------------------------------
# Source 1: sampling-error radius (article-level bootstrap on mean embedding)
# ---------------------------------------------------------------------------


def load_calibration_embeddings(
    embeddings_dir: Path,
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, Float]:
    """Load per-ticker article embeddings restricted to ``[start, end]``.

    Args:
        embeddings_dir: Directory of ``<TICKER>.parquet`` files with columns
            ``embedding`` (object array-like, fixed dim) and ``created_date``
            (ISO string).
        tickers: Tickers to load (raises if a file is missing).
        start: Inclusive ISO calibration-epoch start date.
        end: Inclusive ISO calibration-epoch end date.

    Returns:
        Dict ``{ticker: (n_articles_t, dim)}`` of L2-normalized embeddings
        (angular-metric convention, matching ``jcor``'s default).

    """
    out: dict[str, Float] = {}
    for t in tickers:
        df = pd.read_parquet(
            embeddings_dir / f"{t}.parquet", columns=["created_date", "embedding"]
        )
        mask = (df["created_date"] >= start) & (df["created_date"] <= end)
        sub = df.loc[mask, "embedding"]
        if len(sub) == 0:
            message = f"No calibration-epoch articles for {t} in {start}..{end}"
            raise ValueError(message)
        arr = np.stack([np.asarray(e, dtype=np.float64) for e in sub])
        out[t] = l2_normalize_rows(arr)
    return out


# ---------------------------------------------------------------------------
# Source 2: misspecification radius (GMM moment violation, calibration epoch)
# ---------------------------------------------------------------------------


def pair_gmm_radius(
    calibration_returns: Float, squared_distances: Float, kappa_hat: float
) -> Float:
    """Per-pair misspecification radius on the squared (``D^2``) scale.

    Uses the SAME pair-moment process as Paper 3's identity tests
    (``pipeline.paper3_empirical._pair_moment_process``): under the identity,
    ``E[g_ij] = 0``; the calibration-epoch sample mean ``mean_t g_ij`` is the
    empirical moment-violation magnitude. Converting to a ``D^2``-scale
    radius via the identity's own gradient (``g approx -0.5*kappa^2*Delta(D^2)``
    near the fitted kappa) gives ``eps_ij^GMM = 2*|mean_t g_ij| / kappa_hat^2``.

    DISCLOSURE: this is deliberately NOT folded into the metric-level
    ``eps^stat`` composition (T1 triangle inequality); it is applied as a
    separate additive correction on ``D^2`` in :func:`assemble_brackets`,
    and ONLY in the calibrated (single-``kappa_hat``) mode. The Lean T2
    theorem covers the metric-radius + envelope composition; this D^2-scale
    add-on remains a documented engineering choice for the calibrated set,
    not a machine-checked bound. In the certified mode it is omitted: the
    ``[ell_lo, L_hat]`` envelope already brackets every per-pair
    proportionality, so adding a misfit-of-the-single-``kappa`` correction
    on top would double count the same uncertainty (see
    :func:`implied_kappa_envelope_check` for the empirical justification).

    Args:
        calibration_returns: Calibration-epoch return panel, shape ``(T_cal, n)``.
        squared_distances: Squared energy-distance matrix (``mathcal{E}^2``), shape
            ``(n, n)``.
        kappa_hat: Calibration-epoch kappa (see
            :func:`pipeline.paper3_empirical.calibrate_kappa`).

    Returns:
        ``(n, n)`` symmetric, nonnegative matrix, zero diagonal.

    """
    n = squared_distances.shape[0]
    g, _dg = _pair_moment_process(calibration_returns, squared_distances, kappa_hat)
    mean_g = np.abs(g.mean(axis=0))  # (P,)
    kappa_floor = max(abs(kappa_hat), 1e-3)
    eps_pair = 2.0 * mean_g / (kappa_floor**2)
    iu = np.triu_indices(n, k=1)
    out = np.zeros((n, n))
    out[iu] = eps_pair
    out[(iu[1], iu[0])] = eps_pair
    return out


# ---------------------------------------------------------------------------
# Source 3: identification-frontier envelope (shared W2 frontier)
# ---------------------------------------------------------------------------


def frontier_envelope(frontier_metrics_path: Path) -> dict[str, float]:
    """Read shared W2-to-return-exposure frontier ``[ell_lo, L_hat]``.

    The envelope enters :func:`assemble_brackets` T2-faithfully as the
    interval of distance-to-covariance map constants (``kappa_lo = ell_lo``,
    ``kappa_hi = L_hat``), per ``covariance_bracket_of_embedding_radius``
    (``RobustEnergy.lean``). ``slack_factor = L_hat/ell_lo - 1`` is retained
    as a REPORTED diagnostic of envelope width only; it is never used in the
    bracket composition (the earlier proportional-multiplier composition
    ``eps_metric = eps_stat*(1+slack)`` inflated the metric radius ~7.8x and
    is superseded).

    Args:
        frontier_metrics_path: Path to the shared W2 frontier metrics JSON.

    Returns:
        Dict with ``ell_lo``, ``ell_hat``, ``L_hat`` and the diagnostic
        ``slack_factor``.

    """
    with frontier_metrics_path.open(encoding="utf-8") as fh:
        d = json.load(fh)
    ell_lo = float(d["ell_lo"])
    ell_hat = float(d["ell_hat"])
    l_hat = float(d["L_hat"])
    slack = (l_hat / ell_lo - 1.0) if ell_lo > 0 else float("nan")
    return {"ell_lo": ell_lo, "ell_hat": ell_hat, "L_hat": l_hat, "slack_factor": slack}


def implied_kappa_envelope_check(
    calibration_returns: Float, squared_distances: Float, ell: float, l_hat: float
) -> dict[str, float]:
    """Fraction of pairs whose calibration-implied ``kappa_ij`` lies in ``[ell, L]``.

    Inverting the Sigma_dist identity pairwise on the calibration epoch,
    ``kappa_ij^2 = (sigma_i^2 + sigma_j^2 - 2*S_ij) / D2_ij`` (defined where
    the numerator is positive and ``D2_ij > 0``). If the realized per-pair
    proportionality is (mostly) inside the certified envelope, the envelope
    subsumes the misfit-of-a-single-``kappa`` that the GMM radius measures --
    the empirical justification for omitting ``eps^GMM`` from the CERTIFIED
    bracket (no double counting).

    Args:
        calibration_returns: Calibration-epoch return panel, shape ``(T_cal, n)``.
        squared_distances: Squared energy-distance matrix, shape ``(n, n)``.
        ell: Envelope lower constant (``ell_lo``).
        l_hat: Envelope upper constant (``L_hat``).

    Returns:
        Dict with ``frac_in_envelope`` (of defined pairs), ``frac_defined``
        (pairs with a real implied kappa), ``median_implied_kappa``.

    """
    n = squared_distances.shape[0]
    s = sample_covariance(calibration_returns, ddof=1)
    sigma_sq = np.diag(s)
    iu = np.triu_indices(n, k=1)
    num = sigma_sq[iu[0]] + sigma_sq[iu[1]] - 2.0 * s[iu]
    d2 = squared_distances[iu]
    defined = (num > 0) & (d2 > 0)
    kappa_ij = np.sqrt(num[defined] / d2[defined])
    in_env = (kappa_ij >= ell) & (kappa_ij <= l_hat)
    return {
        "frac_in_envelope": float(np.mean(in_env)) if kappa_ij.size else float("nan"),
        "frac_defined": float(np.mean(defined)),
        "median_implied_kappa": (
            float(np.median(kappa_ij)) if kappa_ij.size else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Bracket assembly
# ---------------------------------------------------------------------------


def assemble_brackets(inputs: CovarianceBracketInputs) -> dict[str, object]:
    """Assemble entrywise covariance brackets ``[Sigma_lo, Sigma_hi]`` (T2).

    Implements the machine-checked Lean T2 composition
    (``covariance_bracket_of_embedding_radius``, ``RobustEnergy.lean``): the
    distance-to-covariance map constant is an INTERVAL ``[kappa_lo,
    kappa_hi]``, and the metric radii enter only through the measured
    distance -- never as a multiplier on each other.

    Composition (in order):

    1. Metric-level triangle-inequality radii (T1):
       ``e_lo = (E - eps_stat)_+``, ``e_hi = E + eps_stat`` where
       ``E = sqrt(D2)`` (asymmetric squared bracket after squaring).
    2. Additive ``D^2``-scale misspecification correction (Source 2,
       calibrated mode only -- pass zeros in certified mode):
       ``D2_hi = e_hi^2 + eps_gmm_d2``, ``D2_lo = max(0, e_lo^2 - eps_gmm_d2)``.
    3. Covariance brackets via ``Sigma = 0.5*(sigma_i^2+sigma_j^2-kappa^2*D2)``
       with the map-constant interval: since larger ``kappa^2*D2`` maps to
       SMALLER Sigma, ``Sigma_lo = sigma_dist(D2_hi, kappa=kappa_hi)`` and
       ``Sigma_hi = sigma_dist(D2_lo, kappa=kappa_lo)`` -- exactly T2's
       ``L*(dist + eps)`` lower / ``ell*max(0, dist - eps)`` upper structure.
       Diagonal fixed at ``sigma_sq`` (Paper 3 convention; no bracket on the
       variances themselves).

    The two modes of the paper are two calls:

    * CERTIFIED (Lean-T2-faithful): ``kappa_lo = ell_lo``, ``kappa_hi =
      L_hat`` (shared W2 envelope), ``eps_gmm_d2 = 0`` (the envelope already
      brackets every per-pair proportionality; adding the GMM misfit of a
      single kappa would double count -- see
      :func:`implied_kappa_envelope_check`).
    * CALIBRATED (statistical): ``kappa_lo = kappa_hi = kappa_hat`` with the
      GMM ``D^2`` correction (disclosed engineering composition).

    Args:
        inputs: Squared distances, radii, variances, and map-constant bounds.

    Returns:
        Dict with ``Sigma_lo``, ``Sigma_hi``, ``D2_lo``, ``D2_hi``,
        metric-level and Sigma-level width summary stats (for the
        materiality diagnostic).

    """
    if not 0.0 <= inputs.kappa_lo <= inputs.kappa_hi:
        message = (
            "Need 0 <= kappa_lo <= kappa_hi, got "
            f"[{inputs.kappa_lo}, {inputs.kappa_hi}]"
        )
        raise ValueError(message)
    n = inputs.squared_distances.shape[0]
    e = np.sqrt(np.clip(inputs.squared_distances, 0.0, None))
    e_lo = np.clip(e - inputs.eps_stat_matrix, 0.0, None)
    e_hi = e + inputs.eps_stat_matrix
    d2_hi = e_hi**2 + inputs.eps_gmm_d2
    d2_lo = np.clip(e_lo**2 - inputs.eps_gmm_d2, 0.0, None)
    np.fill_diagonal(d2_lo, 0.0)
    np.fill_diagonal(d2_hi, 0.0)

    # Materialize writable host copies before fixing the bracket diagonals.
    with jax.enable_x64(new_val=True):
        sigma_hi = np.array(
            sigma_dist(d2_lo, inputs.sigma_sq, kappa=inputs.kappa_lo),
            copy=True,
        )
        sigma_lo = np.array(
            sigma_dist(d2_hi, inputs.sigma_sq, kappa=inputs.kappa_hi),
            copy=True,
        )
    np.fill_diagonal(sigma_hi, inputs.sigma_sq)
    np.fill_diagonal(sigma_lo, inputs.sigma_sq)

    iu = np.triu_indices(n, k=1)
    width_d = (e_hi - e_lo)[iu]
    width_sigma = (sigma_hi - sigma_lo)[iu]
    return {
        "Sigma_lo": sigma_lo,
        "Sigma_hi": sigma_hi,
        "D2_lo": d2_lo,
        "D2_hi": d2_hi,
        "bracket_width_D_median": float(np.median(width_d)),
        "bracket_width_D_mean": float(np.mean(width_d)),
        "bracket_width_Sigma_median": float(np.median(width_sigma)),
        "bracket_width_Sigma_mean": float(np.mean(width_sigma)),
    }


# ---------------------------------------------------------------------------
# Ridge-ball radius r(T) ~ c / sqrt(T)
# ---------------------------------------------------------------------------


#: Cap on the ``chunk x win x n`` block panel one scan step may materialise.
_BLOCK_CHUNK_ELEMENTS = 1 << 22


def _block_covariance_deviations(
    calibration_returns: Float,
    sigma_center: Float,
    block_index: NDArray[np.intp],
) -> Float:
    """Frobenius deviations of each resampled block covariance from the center.

    One traced scan replaces a host loop that called the eager
    ``sample_covariance`` door per draw — each of those validated the panel,
    dispatched a kernel and synced a matrix back, measured at 0.17-0.59 ms of
    round trip. The traced kernel is the same computation without the trip.

    The eager door's finiteness check moves here and runs once on the whole
    panel rather than once per block. Every block is a slice of that panel, so
    the guarantee is unchanged; only the message names the calibration panel
    instead of a nameless ``returns`` argument.
    """
    panel = np.asarray(calibration_returns, dtype=np.float64)
    if not np.all(np.isfinite(panel)):
        message = "calibration_returns must contain only finite values."
        raise ValueError(message)
    center = np.asarray(sigma_center, dtype=np.float64)
    span = max(int(block_index.shape[1]) * panel.shape[1], 1)
    chunk = max(1, min(int(block_index.shape[0]), _BLOCK_CHUNK_ELEMENTS // span))
    with jax.enable_x64(new_val=True):
        blocks = jnp.asarray(panel)
        target = jnp.asarray(center)

        def one_block(index: jax.Array) -> jax.Array:
            covariance = sample_covariance_kernel(blocks[index], ddof=1)
            return jnp.linalg.norm(covariance - target)

        schedule = jnp.asarray(block_index)
        # Prefer a single vmap. `lax.map(batch_size=len)` is not equivalent to
        # it — measured on the robustness grid, the scan wrapper it emits even for
        # one chunk ran 4-15x slower than the plain vmap it should reduce to.
        # The scan is only worth its overhead when the batch would not fit.
        deviations = (
            jax.vmap(one_block)(schedule)
            if block_index.shape[0] <= chunk
            else jax.lax.map(one_block, schedule, batch_size=chunk)
        )
    return np.asarray(deviations, dtype=np.float64)


def bootstrap_ridge_radius(
    inputs: RidgeRadiusInputs,
    config: RidgeRadiusConfig | None = None,
) -> dict[str, object]:
    """Calibrate the Frobenius-ball radius ``r(T) = c / sqrt(T)`` by bootstrap.

    For each window length ``T`` in ``windows``, draws ``n_boot`` circular
    (wrap-around) contiguous blocks of length ``T`` from the calibration
    panel, computes the sample covariance of each block, and takes the
    ``quantile`` (default 90th) of ``||S_T - Sigma_center||_F`` as ``r(T)``.
    Fits the ``r(T) = c/sqrt(T)`` functional form by ``c = median_T(r(T) *
    sqrt(T))`` (closed-form; the plan-mandated rate, not a free fit).

    Args:
        inputs: Calibration panel, covariance center, and target windows.
        config: Bootstrap quantile, resample count, and seed.

    Returns:
        Dict with ``r_by_window`` ``{T: r(T)}``, ``c_hat`` (fitted constant),
        and ``r_fn`` values recomputed from ``c_hat/sqrt(T)`` for
        cross-checking against the raw bootstrap quantiles.

    """
    config = RidgeRadiusConfig() if config is None else config
    t_cal = inputs.calibration_returns.shape[0]
    r_by_window: dict[int, float] = {}
    for win in inputs.windows:
        if win >= t_cal:
            logger.warning(
                "bootstrap_ridge_radius: window %d >= calibration length %d; "
                "using full-panel single draw (no resampling)",
                win,
                t_cal,
            )
            r_by_window[win] = 0.0
            continue
        key = cell_key(
            config.seed,
            "paper5_energy_robust_bootstrap_ridge_radius",
            "window",
            win,
        )
        starts = np.asarray(
            jax.random.randint(key, (config.n_boot,), 0, t_cal),
            dtype=np.intp,
        )
        block_index = (starts[:, None] + np.arange(win)[None, :]) % t_cal
        devs = _block_covariance_deviations(
            inputs.calibration_returns,
            inputs.sigma_center,
            block_index,
        )
        r_by_window[win] = float(np.quantile(devs, config.quantile))
    valid = [v * np.sqrt(w) for w, v in r_by_window.items() if v > 0]
    c_hat = float(np.median(valid)) if valid else 0.0
    r_fn = {w: (c_hat / np.sqrt(w) if w > 0 else 0.0) for w in inputs.windows}
    return {
        "r_by_window_raw": {int(w): float(v) for w, v in r_by_window.items()},
        "c_hat": c_hat,
        "r_by_window_fitted": {int(w): float(v) for w, v in r_fn.items()},
    }


__all__ = [
    "CovarianceBracketInputs",
    "RidgeRadiusConfig",
    "RidgeRadiusInputs",
    "assemble_brackets",
    "bootstrap_ridge_radius",
    "bootstrap_stat_radius",
    "calibrate_kappa",
    "frontier_envelope",
    "implied_kappa_envelope_check",
    "load_calibration_embeddings",
    "pair_gmm_radius",
    "stat_radius_matrix",
]
