r"""DISCO (energy-distance ANOVA) global sector-separation test for Paper 1 (t19).

Implements the *DISCO* decomposition of Rizzo & Szekely (2010), "DISCO
analysis: a nonparametric extension of analysis of variance" — the
energy-statistics-native analogue of PERMANOVA — directly on Paper 1's
square-root energy metric (:math:`\\alpha = 1`) projected from the immutable
functional computed by the substrate energy pipeline (the same
``energy_tests.parquet`` t18's :mod:`sector_alignment` module loads; no
recomputation of distances).

For groups :math:`g = 1..K` with sizes :math:`n_g`, :math:`N = \\sum_g n_g`,
and pairwise energy distances :math:`d_{ij}` (:math:`\\alpha = 1`), let
:math:`\\bar a_{XY}` be the mean of :math:`d_{ij}` over :math:`i \\in X, j \\in
Y` (including the zero diagonal when :math:`X = Y`, as required by the
V-statistic definition). Within-group dispersion is
:math:`S_W = \\sum_g (n_g/2) \\bar a_{gg}`; total dispersion is :math:`T =
(N/2) \\bar a`, with :math:`\\bar a` the mean pairwise distance over the whole
sample; between-group dispersion is :math:`S_B = T - S_W`. This module
reports:

- the DISCO index :math:`R^2_E = S_B / T`;
- the DISCO F-ratio :math:`F = [S_B/(K-1)] / [S_W/(N-K)]`;
- an add-one label-permutation :math:`p` for :math:`F`
  (:math:`p = (1 + \\#\\{F_{perm} \\ge F_{obs}\\})/(B+1)`), fixed seed, :math:`B
  \\ge 9{,}999`;
- a **dispersion companion**: distance from each firm to its sector's spatial
  (geometric) median in PCoA coordinates, summarised by a one-way-ANOVA-style
  F-ratio across sectors (a PERMDISP-style homogeneity-of-dispersion test),
  with its own permutation :math:`p`.

PCoA (classical/Torgerson scaling) is built from the double-centred
*squared* distance matrix — the standard Gower construction — which is
legitimate because energy distance (:math:`\\alpha = 1`) is of **negative
type**: the double-centred Gram matrix is guaranteed PSD, so no
Lingoes/Cailliez correction is applied; non-negative eigenvalues (up to
numerical tolerance) are asserted rather than corrected for. Squaring here is
purely the PCoA embedding step and is unrelated to the DISCO statistic
itself, which never squares the energy distances.

No third-party statistics dependency (no ``scikit-bio``/``vegan``): both the
DISCO test and the PERMDISP-style dispersion companion are implemented
natively in JAX over the existing distance matrix.

Where the method now lives (t46.6)
----------------------------------
Only the **artifact glue** is Paper 1's: label encoding, the parquet/CSV read,
the provenance strings, the YAML write. The three method cores moved into
``jcor``, each to the stage that owns the object it consumes:

* the DISCO decomposition and its null → :mod:`jcor.association.dispersion`
  (a distance matrix plus a grouping in, an ANOVA table out);
* classical scaling → :mod:`jcor.geometry.embedding`, which now takes a
  ``DMat`` whose law declares conditional negative definiteness of ``D²`` and
  **raises** on a materially negative Gram spectrum instead of clipping it away;
* the spatial medians and the PERMDISP F-ratio → :mod:`jcor.geometry.median`.

The two ``jcor`` halves cannot import each other — ``association`` and
``geometry`` are rank-tied at 4 in the import fence — so this module is the
composition point. That is legal in this direction: ``pipeline`` sits above all
of ``jcor``. The stage enters a caller-owned ``jax.enable_x64`` context around
the branded PCoA call. Argument canonicalization happens before a jitted body
can change precision, so the JCOR door does not mutate process-global policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jcor.association.decomposition import group_design
from jcor.association.dispersion import disco_decomposition as _disco_core
from jcor.core.matrices import unsafe_assume_dmat
from jcor.geometry.embedding import pcoa as _pcoa
from jcor.geometry.median import dispersion_f_ratio as _dispersion_core
from jcor.geometry.median import dispersion_permutation_null
from jcor.inference.calibration import add_one_p as _add_one_p
from jcor.inference.disco import disco_permutation_test

from pipeline.stages.substrate.energy_metric import _load_energy_metric

if TYPE_CHECKING:
    from pathlib import Path

    from jcor.core.domains import EmpiricalDistribution
    from jcor.core.matrices import (
        DMat,
        HilbertianMetricLaw,
        RootEnergyMetric,
        VStatisticScheme,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "SIGNIFICANCE_LEVEL",
    "_add_one_p",
    "_compute_disco",
    "_disco_core",
    "_encode_groups",
    "_pcoa",
    "run_disco",
]

_DEFAULT_N_PERMUTATIONS = 9_999
_DEFAULT_SEED = 42
SIGNIFICANCE_LEVEL = 0.05


def _encode_groups(sector_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Map sector labels to contiguous deterministic integer codes."""
    uniques = sorted(set(sector_ids.tolist()))
    lookup = {s: i for i, s in enumerate(uniques)}
    codes = np.array([lookup[s] for s in sector_ids], dtype=np.int32)
    return codes, uniques


def _compute_disco(
    dist: np.ndarray,
    sector_ids: np.ndarray,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    random_state: int = _DEFAULT_SEED,
) -> dict[str, object]:
    """Compose the DISCO test with its dispersion companion.

    Kept separate from I/O so tests can exercise
    the definitions and permutation determinism without touching disk. Since
    t46.6 the two halves come from ``jcor``: the decomposition from
    :mod:`jcor.association.dispersion`, the embedding and the PERMDISP F-ratio
    from :mod:`jcor.geometry`. This function is the only place they meet — see
    the module docstring for why ``jcor`` cannot join them itself.

    Args:
        dist: Square distance matrix over the retained firms.
        sector_ids: Sector label per firm, aligned with ``dist``.
        n_permutations: Number of label permutations for both nulls.
        random_state: Seed shared by both permutation nulls.

    Returns:
        The flat summary dict written to ``summary.yaml``.

    """
    group_codes, uniques = _encode_groups(sector_ids)
    n_groups = len(uniques)
    n = dist.shape[0]

    with jax.enable_x64(new_val=True):
        matrix: DMat[
            EmpiricalDistribution,
            HilbertianMetricLaw,
            RootEnergyMetric[VStatisticScheme],
        ] = unsafe_assume_dmat(jnp.asarray(dist))
        design = group_design(group_codes, n_groups=n_groups)
        calibrated = disco_permutation_test(
            matrix,
            design,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        coords_jax, eigvals_jax = _pcoa(matrix)
    decomposition = calibrated.decomposition
    disco_p = float(calibrated.test.pvalue)
    disco_null = np.asarray(calibrated.null.draws, dtype=np.float64)
    coords = np.asarray(coords_jax, dtype=np.float64)
    eigvals = np.asarray(eigvals_jax, dtype=np.float64)
    disp_f_obs = float(
        _dispersion_core(jnp.asarray(coords), jnp.asarray(group_codes), n_groups)
    )
    # jcor returns the null as a JAX array; the host conversion is this stage's
    # job, matching how `calibrated.null.draws` is handled above.
    disp_f_perm = np.asarray(
        dispersion_permutation_null(
            coords, group_codes, n_groups, n_permutations, random_state
        ),
        dtype=np.float64,
    )
    disp_p = _add_one_p(disp_f_obs, disp_f_perm)

    location_driven = bool(np.isfinite(disco_p) and disco_p < SIGNIFICANCE_LEVEL)
    dispersion_driven = bool(np.isfinite(disp_p) and disp_p < SIGNIFICANCE_LEVEL)
    if location_driven and not dispersion_driven:
        reading = "location-driven"
    elif dispersion_driven and not location_driven:
        reading = "dispersion-driven"
    elif location_driven and dispersion_driven:
        reading = "both location- and dispersion-driven"
    else:
        reading = "neither location- nor dispersion-driven"

    return {
        "r2_e": float(decomposition.index.value),
        "s_within": float(decomposition.components.within.value),
        "s_between": float(decomposition.components.between.value),
        "total_dispersion": float(decomposition.components.total.value),
        "f_statistic": float(decomposition.f_ratio.value),
        "disco_perm_p": disco_p,
        "disco_perm_mean": float(disco_null.mean()),
        "disco_perm_std": float(disco_null.std(ddof=1)),
        "dispersion_f_statistic": disp_f_obs,
        "dispersion_perm_p": disp_p,
        "dispersion_perm_mean": float(disp_f_perm.mean()),
        "dispersion_perm_std": float(disp_f_perm.std(ddof=1)),
        "pcoa_min_eigenvalue": float(eigvals.min()),
        "pcoa_n_positive_axes": int(coords.shape[1]),
        "reading": reading,
        "n_groups": int(n_groups),
        "n_firms": int(n),
        "n_permutations": int(n_permutations),
        "random_state": int(random_state),
    }


def run_disco(
    distance_artifact_dir: Path,
    universe_csv: Path,
    output_file: Path,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    random_state: int = _DEFAULT_SEED,
    provider_id: str = "qwen3-embedding-8b",
    representation_id: str = "qwen3-embedding-8b-unit",
    distance_id: str = "energy_v",
) -> dict[str, object]:
    """Compute DISCO and dispersion statistics and write YAML output.

    Reuses the immutable shared functional and explicitly projects it to the
    square-root metric used by Paper 1 and t18's :mod:`sector_alignment`
    (:func:`_load_energy_distances`) — no recomputation of embeddings.
    """
    tickers, energy_dist = _load_energy_metric(
        distance_artifact_dir,
        provider_id=provider_id,
        representation_id=representation_id,
        distance_id=distance_id,
    )

    universe = pd.read_csv(universe_csv)
    sectors = dict(zip(universe["Symbol"], universe["Sector"], strict=False))

    common = sorted(t for t in tickers if t in sectors)
    idx = [tickers.index(t) for t in common]
    dist = energy_dist[np.ix_(idx, idx)]
    sector_ids = np.array([sectors[t] for t in common])

    result = _compute_disco(
        dist, sector_ids, n_permutations=n_permutations, random_state=random_state
    )
    result["sector_sizes"] = {
        str(s): int(c) for s, c in pd.Series(sector_ids).value_counts().items()
    }
    result["energy_input_scale"] = "energy_functional_S=2A-B_i-B_j"
    result["energy_analysis_scale"] = "sqrt_energy_functional"
    result["disco_normalization"] = "V-statistic n_g^2 and N^2 denominators"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return result
