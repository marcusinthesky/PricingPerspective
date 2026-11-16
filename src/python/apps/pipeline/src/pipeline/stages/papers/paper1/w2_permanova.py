"""Sector PERMANOVA on the canonical, original rooted-W2 firm geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statadic.hypothesis.core import Hypothesis
from statadic.hypothesis.homogeneity import (
    PERMDISP,
    EqualGroupDispersion,
    PermutationPlan,
)
from statadic.observation.data import Observations
from statadic.observation.design import Categorical, Intercept
from statadic.observation.geometry import ObservedDissimilarity
from statadic.ordination.fit import Ordination
from statadic.regression import (
    GeometricResponse,
    Reduced,
    Regression,
    ResponsePermutation,
)

from pipeline.io.w2_permanova_artifacts import write_w2_permanova_results
from pipeline.precision import own_float64
from pipeline.stages.shared.typed_artifacts import read_typed_distance_artifact

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from statadic.observation.coordinates import OrdinationSpectrum

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

_EPS = np.finfo(np.float64).eps


def _validate_distances(distances: object) -> tuple[FloatArray, float]:
    """Admit finite nonnegative symmetric distances with a zero diagonal."""
    original = np.asarray(distances)
    if original.dtype.kind not in "fiu" or original.ndim != 2:  # noqa: PLR2004
        message = "distances must be a numeric square matrix"
        raise ValueError(message)
    matrix = np.asarray(original, dtype=np.float64)
    if matrix.shape[0] < 3 or matrix.shape[0] != matrix.shape[1]:  # noqa: PLR2004
        message = "distances must be square over at least three firms"
        raise ValueError(message)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        message = "distances must be finite and nonnegative"
        raise ValueError(message)
    precision = (
        float(np.finfo(original.dtype).eps) if original.dtype.kind == "f" else _EPS
    )
    scale = float(matrix.max())
    tolerance = 32 * precision * scale
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
        message = "distances must be symmetric"
        raise ValueError(message)
    if not np.all(np.abs(np.diag(matrix)) <= tolerance):
        message = "distance diagonal must be zero"
        raise ValueError(message)
    # Average only admitted representation roundoff; avoid overflow in d+d.T.
    matrix = matrix / 2 + matrix.T / 2
    np.fill_diagonal(matrix, 0)
    return matrix, precision


def _encode_groups(labels: Sequence[str], n: int) -> tuple[IntArray, list[str]]:
    if len(labels) != n or any(not isinstance(s, str) or not s.strip() for s in labels):
        message = "each firm requires one nonmissing, nonempty sector label"
        raise ValueError(message)
    names = sorted(set(labels))
    if not 1 < len(names) < n:
        message = (
            "PERMANOVA requires at least two groups and positive residual "
            "degrees of freedom"
        )
        raise ValueError(message)
    lookup = {name: index for index, name in enumerate(names)}
    return np.array([lookup[label] for label in labels], dtype=np.int64), names


def _schedule(
    supplied: object | None, n: int, count: int, rng: np.random.Generator
) -> IntArray:
    if supplied is None:
        return np.array([rng.permutation(n) for _ in range(count)], dtype=np.int64)
    schedule = np.asarray(supplied)
    if schedule.dtype.kind not in "iu" or schedule.shape != (count, n):
        message = (
            "permutation indices must have integer shape (n_permutations, n_firms)"
        )
        raise ValueError(message)
    if not np.all(np.sort(schedule, axis=1) == np.arange(n)):
        message = "each permutation row must contain each firm index exactly once"
        raise ValueError(message)
    return np.asarray(schedule, dtype=np.int64).copy()


def _upper_tail(observed: float, null: FloatArray) -> tuple[float, int]:
    """Count positive-infinity ties; never silently discard a reference draw."""
    if np.isnan(observed) or np.isnan(null).any():
        message = "undefined statistics cannot be permutation calibrated"
        raise ValueError(message)
    if np.isposinf(observed):
        exceedances = int(np.isposinf(null).sum())
    else:
        tolerance = 64 * _EPS * max(1, abs(observed))
        exceedances = int(np.count_nonzero(null >= observed - tolerance))
    return (1 + exceedances) / (len(null) + 1), exceedances


def _json_statistic(value: float) -> tuple[float | None, str]:
    if np.isposinf(value):
        return None, "positive_infinity"
    if not np.isfinite(value):
        return None, "undefined"
    return value, "finite"


def _original_units(value: float, scale: float) -> float | None:
    """Restore squared-distance units without exporting inf or false zero."""
    restored = np.longdouble(value) * np.longdouble(scale) ** 2
    if abs(restored) > np.finfo(np.float64).max:
        return None
    result = float(restored)
    return None if value != 0 and result == 0 else result


def _spectrum_summary(
    spectrum: OrdinationSpectrum,
) -> tuple[FloatArray, dict[str, object]]:
    """Report the signed Gram spectrum ascending, as the archived audit stores it."""
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=np.float64)[::-1]
    absolute_inertia = float(np.abs(eigenvalues).sum())
    return eigenvalues, {
        "minimum_eigenvalue": float(eigenvalues.min()),
        "eigenvalue_tolerance": spectrum.threshold,
        "n_positive_axes": spectrum.n_positive,
        "n_negative_axes": spectrum.n_negative,
        "negative_inertia_fraction": (
            float(-eigenvalues[eigenvalues < 0].sum() / absolute_inertia)
            if absolute_inertia > 0
            else 0.0
        ),
        "material_negative_inertia": spectrum.material_negative_inertia,
        "eigenvalue_units": "distance_max_squared",
    }


def _dispersion(
    distances: FloatArray,
    codes: IntArray,
    names: list[str],
    *,
    precision: float,
    count: int,
    seed: int,
    supplied: object | None,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Bias-adjusted PERMDISP on the replicated sectors via Statadic.

    Singleton sectors contribute no spread and are excluded here only; the
    companion is withheld when its retained subset has material negative inertia.
    """
    sizes = np.bincount(codes)
    kept = np.flatnonzero(sizes[codes] > 1)
    retained_groups = np.flatnonzero(sizes > 1)
    base: dict[str, object] = {
        "status": "not_computed",
        "reason": None,
        "n_firms": len(kept),
        "n_groups": len(retained_groups),
        "excluded_singleton_sectors": [names[i] for i in np.flatnonzero(sizes == 1)],
        "statistic": None,
        "statistic_status": "undefined",
        "pvalue": None,
        "exceedances": None,
        "n_permutations": count,
        "random_state": seed,
        "center": "spatial_median",
        "bias_adjustment": "sqrt(n_g/(n_g-1))",
        "resampling": "full_model_residual_permutation",
        "radii_units": "distance_max",
    }
    audit: dict[str, np.ndarray] = {"dispersion_firm_indices": kept}
    if len(retained_groups) < 2:  # noqa: PLR2004
        base["reason"] = "fewer_than_two_replicated_sectors"
        return base, audit
    _, subset_codes = np.unique(codes[kept], return_inverse=True)
    ids = tuple(range(len(kept)))
    fitted = PERMDISP(bias_adjust=True).from_distances(
        distances[np.ix_(kept, kept)],
        groups=subset_codes,
        row_ids=ids,
        group_ids=ids,
        tolerance=32 * precision * len(kept),
    )
    assert fitted.spectrum is not None  # noqa: S101
    eigenvalues, spectrum = _spectrum_summary(fitted.spectrum)
    base["spectrum"] = spectrum
    audit["dispersion_gram_eigenvalues"] = eigenvalues
    if fitted.reason is not None:
        base["reason"] = fitted.reason
        return base, audit
    schedule = _schedule(supplied, len(kept), count, rng)
    audit.update(
        {
            "dispersion_radii": np.asarray(fitted.dispersion_radii, dtype=np.float64),
            "dispersion_group_codes": subset_codes,
            "dispersion_permutation_indices": schedule,
        }
    )
    statistic = float(fitted.statistic)
    base["statistic"], base["statistic_status"] = _json_statistic(statistic)
    outcome = fitted.infer(
        resampling=PermutationPlan(n=count, indices=schedule, row_ids=ids)
    ).test(Hypothesis(EqualGroupDispersion()))
    assert outcome.evidence is not None  # noqa: S101
    null = np.asarray(outcome.evidence.null, dtype=np.float64)
    if np.isnan(null).all():
        # statadic marks every draw undefined when full-model residuals vanish.
        base["reason"] = "zero_within_group_radius_variation"
        return base, audit
    audit["dispersion_null_f"] = null
    pvalue, exceedances = _upper_tail(statistic, null)
    base.update(status="computed", pvalue=pvalue, exceedances=exceedances)
    return base, audit


def _analyze(
    distances: object,
    labels: Sequence[str],
    *,
    n_permutations: int = 9_999,
    random_state: int = 42,
    permutation_indices: object | None = None,
    dispersion_permutation_indices: object | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Return JSON-safe summaries and the exact permutation audit arrays.

    Rows of each supplied schedule index *labels* (main) or full-model scalar
    residuals (companion).

    The statistic is the one-way PERMANOVA of the squared supplied rooted-W2
    dissimilarities (no Hilbertian claim); draws use the stage's NumPy schedules.
    All draws, including identity and infinity ties, enter the add-one
    denominator through the stage's tolerance-tied upper tail.
    """
    own_float64()
    if (
        not isinstance(n_permutations, int)
        or isinstance(n_permutations, bool)
        or n_permutations < 1
    ):
        message = "n_permutations must be a positive integer"
        raise ValueError(message)
    matrix, precision = _validate_distances(distances)
    n = len(matrix)
    codes, names = _encode_groups(labels, n)
    sizes = np.bincount(codes)
    k = len(names)
    seeds = np.random.SeedSequence(random_state).spawn(2)
    schedule = _schedule(
        permutation_indices, n, n_permutations, np.random.default_rng(seeds[0])
    )
    scale = float(matrix.max())
    normalized = matrix / scale if scale > 0 else matrix
    ids = tuple(range(n))
    full = Ordination.pcoa(
        dim=None,
        negative_inertia="positive-subspace",
        tolerance=32 * precision * n,
    ).estimate(ObservedDissimilarity.from_array(normalized, ids))
    eigenvalues, spectrum = _spectrum_summary(full.spectrum)
    spectrum["normalization_distance_max"] = scale
    audit: dict[str, np.ndarray] = {
        "permanova_permutation_indices": schedule,
        "gram_eigenvalues": eigenvalues,
    }
    main: dict[str, object] = {
        "status": "not_computed" if scale == 0 else "computed",
        "reason": "all_distances_zero" if scale == 0 else None,
        "ss_total": 0.0,
        "ss_within": 0.0,
        "ss_between": 0.0,
        "r2": None,
        "pseudo_f": None,
        "pseudo_f_status": "undefined",
        "pvalue": None,
        "exceedances": None,
        "n_permutations": n_permutations,
        "random_state": random_state,
        "null_r2_expectation": (k - 1) / (n - 1),
        "null_r2_mean": None,
        "excess_r2": None,
        "null_r2_reference_quantiles": None,
        "tie_tolerance": (
            "64*float64_eps*max(1,abs(observed_F)); positive infinities tie"
        ),
    }
    if scale > 0:
        observed = Regression(
            response=GeometricResponse.from_distances(
                normalized, row_ids=ids, distance_power=2
            ),
            terms=Intercept() + Categorical("sector"),
        ).estimate(Observations({}, row_ids=ids, labels={"sector": labels}))
        assert observed.decomposition is not None  # noqa: S101
        within, between, total, r2, f_value = (
            float(value) for value in observed.decomposition
        )
        outcome = observed.infer(
            resampling=ResponsePermutation(
                PermutationPlan(n=n_permutations, indices=schedule, row_ids=ids)
            )
        ).test(Hypothesis(Reduced(("intercept",))))
        assert outcome.evidence is not None  # noqa: S101
        null_f = np.asarray(outcome.evidence.null, dtype=np.float64)
        # R2 = (k-1)F / ((k-1)F + n-k) is exact; F = inf is R2 = 1.
        weighted = (k - 1) * null_f
        null_r2 = np.divide(
            weighted,
            weighted + n - k,
            out=np.ones_like(weighted),
            where=np.isfinite(weighted),
        )
        pvalue, exceedances = _upper_tail(f_value, null_f)
        pseudo_f, statistic_status = _json_statistic(f_value)
        main.update(
            {
                "ss_total": _original_units(total, scale),
                "ss_within": _original_units(within, scale),
                "ss_between": _original_units(between, scale),
                "r2": r2,
                "pseudo_f": pseudo_f,
                "pseudo_f_status": statistic_status,
                "pvalue": pvalue,
                "exceedances": exceedances,
                "null_r2_mean": float(null_r2.mean()),
                "excess_r2": r2 - float(null_r2.mean()),
                "null_r2_reference_quantiles": {
                    str(q): float(np.quantile(null_r2, q)) for q in (0.025, 0.5, 0.975)
                },
            }
        )
        main["sum_of_squares_units_status"] = (
            "finite"
            if all(
                main[key] is not None for key in ("ss_total", "ss_within", "ss_between")
            )
            else "outside_float64_range"
        )
        audit.update(permanova_null_pseudo_f=null_f, permanova_null_r2=null_r2)
    dispersion, dispersion_audit = _dispersion(
        normalized,
        codes,
        names,
        precision=precision,
        count=n_permutations,
        seed=random_state,
        supplied=dispersion_permutation_indices,
        rng=np.random.default_rng(seeds[1]),
    )
    audit.update(dispersion_audit)
    return {
        "schema_version": 1,
        "n_firms": n,
        "n_groups": k,
        "sector_sizes": dict(zip(names, (int(value) for value in sizes), strict=True)),
        "permanova": main,
        "dispersion": dispersion,
        "spectrum": spectrum,
    }, audit


@dataclass(frozen=True, slots=True)
class Paper1W2PermanovaPaths:
    """The typed geometry, explicitly priced cohort, classifications and outputs."""

    distance_artifact_dir: Path
    covariance_matrix: Path
    universe_csv: Path
    output_dir: Path


def run_w2_permanova(
    paths: Paper1W2PermanovaPaths,
    *,
    provider_id: str,
    representation_id: str,
    sample_size: int = 128,
    n_permutations: int = 9_999,
    random_state: int = 42,
) -> dict[str, object]:
    """Admit the full priced cohort, analyze its W2 geometry and save the draws.

    Covariance values never enter this calculation; the covariance artifact
    supplies identities only. Missing sectors or geometry fail rather than
    changing the cohort through an implicit intersection.
    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = paths.output_dir / "results.json"
    result_path.unlink(missing_ok=True)
    frame, summary = read_typed_distance_artifact(
        paths.distance_artifact_dir,
        expected_identity={
            "provider_id": provider_id,
            "representation_id": representation_id,
            "distance_id": "wasserstein_w2",
            "value_semantics": "statistical_distance",
            "sample_size": sample_size,
        },
    )
    metadata = summary["metadata"]
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value
        for key, value in {
            "estimator": "balanced_wasserstein_2",
            "normalization": "rooted",
        }.items()
    ):
        message = "sector PERMANOVA requires rooted balanced-W2 distances"
        raise ValueError(message)
    item_ids = summary["item_ids"]
    if not isinstance(item_ids, list) or not all(
        isinstance(item, str) for item in item_ids
    ):
        message = "typed distance item IDs must be strings"
        raise ValueError(message)
    cohort = pd.read_parquet(paths.covariance_matrix, columns=["ticker_i", "ticker_j"])
    if cohort.empty or cohort.isna().any().any():
        message = "priced cohort identities must be nonmissing"
        raise ValueError(message)
    tickers = sorted(set(cohort["ticker_i"]) | set(cohort["ticker_j"]))
    if not set(tickers) <= set(item_ids):
        message = "W2 artifact does not cover the full priced cohort"
        raise ValueError(message)
    universe = pd.read_csv(paths.universe_csv, dtype=str)
    if universe["Symbol"].isna().any() or universe["Symbol"].duplicated().any():
        message = "universe requires unique, nonmissing firm symbols"
        raise ValueError(message)
    sectors = universe.set_index("Symbol").reindex(tickers)["Sector"]
    if sectors.isna().any() or sectors.str.strip().eq("").any():
        message = "every priced firm must have a nonmissing sector"
        raise ValueError(message)
    positions = [item_ids.index(ticker) for ticker in tickers]
    distances = (
        frame["value"].to_numpy(dtype=np.float64).reshape(len(item_ids), len(item_ids))
    )
    result, audit = _analyze(
        distances[np.ix_(positions, positions)],
        sectors.tolist(),
        n_permutations=n_permutations,
        random_state=random_state,
    )
    result["tickers"] = tickers
    result["distance_identity"] = {
        "provider_id": provider_id,
        "representation_id": representation_id,
        "distance_id": "wasserstein_w2",
        "sample_size": sample_size,
        "estimator": metadata["estimator"],
        "normalization": metadata["normalization"],
        "code_identity": metadata.get("code_identity"),
    }
    result["interpretation"] = (
        "Post-hoc sector association conditional on the observed firms and "
        "distances. Unrestricted whole-firm relabelling requires exchangeability; "
        "sector assignment is not randomized. A nonsignificant dispersion "
        "companion does not establish equal dispersions or a location-only effect."
    )
    audit["tickers"] = np.asarray(tickers, dtype=str)
    audit["sector_labels"] = sectors.to_numpy(dtype=str)
    return write_w2_permanova_results(paths.output_dir, result, audit)
