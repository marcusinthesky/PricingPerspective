"""Paper 5 Monte-Carlo and two-field diagnostic figures.

Drawn here rather than in ``stages/papers/paper5/mc.py``, which produced them until
t39. Living outside ``pipeline.figures`` meant no ``_style()`` (so they rendered in
matplotlib's default sans face, not the manuscript serif), hand-hidden spines
instead of the shared rcParams, inlined ``#0072B2``/``#666666``/``#D55E00``, and no
coverage from ``tests/figures/test_palette_discipline.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from pipeline.figures._common import (
    _C,
    _save,
    _style,
    categorical_axis,
    figsize,
    fit_to_width,
)
from pipeline.io.barycentre_artifacts import read_target_projection_artifact

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.contour import ContourSet
    from numpy.typing import NDArray

    # Renderer inputs arrive either as plain sequences from a simulation or as
    # arrays read back from the stage's parquet outputs.
    Samples = Sequence[float] | NDArray[np.float64]

logger = logging.getLogger(__name__)

# Deviance floors for the two views. The zoom panel is cut below the descriptive
# quasi-log-likelihood contour so the ridge fills it; the context panel needs a
# far deeper cut because a likelihood over tens of thousands of residuals falls
# away from its maximum within a few hundredths of the coefficient.
_ZOOM_FLOOR = -20.0
_CONTEXT_FLOOR = -4000.0
# Descriptive quasi-log-likelihood drop retained for comparison across the surface.
_QUASI_LOGLIK_CONTOUR_DROP = -2.9957
_SURFACE_LEVELS = 11
_ZOOM_MARGIN_FRACTION = 0.25
_ZOOM_MIN_MARGIN = 0.01
_BARYCENTRIC_TOP_COUNT = 5
_MATRIX_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class MisspecificationSeries:
    """Bias in the SAR autocorrelation against W-misspecification magnitude."""

    magnitude: Sequence[float]
    bias_mean: Sequence[float]
    bias_se: Sequence[float]


@dataclass(frozen=True, slots=True)
class SizePowerSeries:
    """Rejection rate against the true mean alpha, with the empirical size."""

    alpha_true: Sequence[float]
    power: Sequence[float]
    empirical_size: float
    nominal_alpha: float
    #: Cross-section size the study was run at. Carried rather than written into
    #: the title, which previously stated a fixed n=52 and would have mislabelled
    #: the figure the moment the panel changed.
    n_assets: int

    def __post_init__(self) -> None:
        """Reject malformed size-power metadata before publication rendering."""
        if (
            isinstance(self.n_assets, bool)
            or not isinstance(self.n_assets, int)
            or self.n_assets < 1
        ):
            message = (
                "size-power artifact metadata 'n' must be a positive integer; "
                f"got {self.n_assets!r}"
            )
            raise ValueError(message)
        if len(self.alpha_true) != len(self.power):
            message = (
                "size-power artifact is inconsistent: alpha_true and power "
                f"contain {len(self.alpha_true)} and {len(self.power)} values"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class BarycentricMixturePaths:
    """Governed inputs and output for the empirical W-flat mixture map."""

    barycentre_dir: Path
    universe_csv: Path
    output_file: Path


def render_w_misspecification(series: MisspecificationSeries, out: Path) -> None:
    r"""Plot bias in :math:`\hat{\rho}` as the weight matrix is perturbed."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.8, 3.2 / 4.5))
    ax.errorbar(
        series.magnitude,
        series.bias_mean,
        yerr=series.bias_se,
        marker="o",
        color=_C["blue"],
        capsize=3,
    )
    ax.axhline(0.0, color=_C["gray"], linewidth=0.7, linestyle="--")
    ax.set_xlabel("W misspecification magnitude")
    ax.set_ylabel(r"Bias in $\hat{\rho}$")
    ax.set_title(r"SAR $\rho$ recovery under W misspecification")
    fig.tight_layout()
    fit_to_width(fig, 0.8)
    _save(fig, out)


def render_size_power(series: SizePowerSeries, out: Path) -> None:
    r"""Plot the power curve of the :math:`\bar{\alpha}=0` test against its size."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.8, 3.2 / 4.5))
    ax.plot(
        series.alpha_true, series.power, marker="o", color=_C["blue"], label="power"
    )
    ax.axhline(
        series.empirical_size,
        color=_C["vermilion"],
        linestyle="--",
        label=f"size={series.empirical_size:.3f}",
    )
    ax.axhline(
        series.nominal_alpha,
        color=_C["gray"],
        linewidth=0.7,
        linestyle=":",
        label=f"nominal {series.nominal_alpha:g}",
    )
    ax.set_xlabel(r"True $\bar{\alpha}$")
    ax.set_ylabel("Rejection rate")
    ax.set_title(rf"Size/power of the $\bar{{\alpha}}=0$ test, $n={series.n_assets}$")
    ax.legend()
    fig.tight_layout()
    fit_to_width(fig, 0.8)
    _save(fig, out)


@dataclass(frozen=True, slots=True)
class TwoFieldRegion:
    """Joint two-field profile surface, bootstrap cloud, and nested boundaries.

    The surface is regular in the mixture weight and the total feedback, so it
    is scattered in channel coordinates and must be contoured as such.
    """

    surface_rho_b: Samples
    surface_rho_n: Samples
    surface_loglik: Samples
    bootstrap_rho_b: Samples
    bootstrap_rho_n: Samples
    joint_rho_b: float
    joint_rho_n: float
    field_b_boundary: float
    field_n_boundary: float


def _draw_two_field_panel(
    ax: plt.Axes,
    region: TwoFieldRegion,
    *,
    floor: float,
    zoomed: bool,
) -> ContourSet:
    """Draw one view of the joint surface at the requested deviance floor."""
    rho_b = np.asarray(region.surface_rho_b, dtype=np.float64)
    rho_n = np.asarray(region.surface_rho_n, dtype=np.float64)
    deviance = np.asarray(region.surface_loglik, dtype=np.float64)
    deviance = np.maximum(deviance - deviance.max(), floor)

    filled = ax.tricontourf(
        rho_b,
        rho_n,
        deviance,
        levels=np.linspace(floor, 0.0, _SURFACE_LEVELS),
        cmap="Blues",
    )
    if zoomed:
        # Both single-field fits lie far outside this window, and the stability
        # boundary is off-panel; labelling them here would key markers to nothing.
        ax.tricontour(
            rho_b,
            rho_n,
            deviance,
            levels=[_QUASI_LOGLIK_CONTOUR_DROP],
            colors=_C["black"],
            linewidths=0.8,
        )
        ax.scatter(
            region.bootstrap_rho_b,
            region.bootstrap_rho_n,
            s=2,
            alpha=0.2,
            color=_C["gray"],
            linewidths=0.0,
            label="bootstrap refits",
        )
        ax.plot(
            [],
            [],
            color=_C["black"],
            linewidth=0.8,
            label="quasi-log-likelihood contour (drop 2.9957)",
        )
        ax.plot(
            [region.joint_rho_b],
            [region.joint_rho_n],
            marker="o",
            markersize=5,
            linestyle="none",
            color=_C["vermilion"],
        )
        ax.set_xlabel(r"Distributional channel $\rho_B$")
        return filled

    ax.plot(
        [0.0, 1.0],
        [1.0, 0.0],
        color=_C["gray"],
        linewidth=0.7,
        linestyle="--",
        label="stability boundary",
    )
    ax.plot(
        [region.field_b_boundary, 0.0],
        [0.0, region.field_n_boundary],
        marker="s",
        markersize=4,
        linestyle="none",
        color=_C["orange"],
        label="single-field fits",
    )
    ax.plot(
        [region.joint_rho_b],
        [region.joint_rho_n],
        marker="o",
        markersize=5,
        linestyle="none",
        color=_C["vermilion"],
        label="joint estimate",
    )
    ax.set_xlabel(r"Distributional channel $\rho_B$")
    return filled


def _zoom_limits(values: Samples, centre: float) -> tuple[float, float]:
    """Bracket a bootstrap coordinate with a margin, clipped below at zero."""
    array = np.asarray(values, dtype=np.float64)
    lower = min(float(array.min()), centre)
    upper = max(float(array.max()), centre)
    margin = max(_ZOOM_MIN_MARGIN, _ZOOM_MARGIN_FRACTION * (upper - lower))
    return max(0.0, lower - margin), upper + margin


def render_two_field_region(region: TwoFieldRegion, out: Path) -> None:
    r"""Plot the joint quasi-log-likelihood surface over the two coefficients.

    The exhibit is meant to read the same way whichever answer the data give. A
    surface that peaks strictly inside the triangle says the two fields carry
    separable information; a long diagonal ridge running between the two
    single-field fits says they do not, and a bootstrap cloud stretched along
    that ridge is what weak identification looks like.

    Two panels are needed because the two facts live at different scales. The
    left panel places the joint estimate against both single-field fits and the
    stability boundary; at that scale a likelihood built on tens of thousands of
    residuals is a point. The right panel is the same surface zoomed to the
    resample cloud, where the ridge and the descriptive contour are legible.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(1.0, 3.2 / 7.0))
    _draw_two_field_panel(axes[0], region, floor=_CONTEXT_FLOOR, zoomed=False)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel(r"News-link channel $\rho_N$")
    axes[0].set_title("Admissible region")
    axes[0].legend(loc="upper right", fontsize="small")

    filled = _draw_two_field_panel(axes[1], region, floor=_ZOOM_FLOOR, zoomed=True)
    axes[1].set_xlim(*_zoom_limits(region.bootstrap_rho_b, region.joint_rho_b))
    axes[1].set_ylim(*_zoom_limits(region.bootstrap_rho_n, region.joint_rho_n))
    axes[1].set_title("Joint estimate and resample cloud")
    axes[1].legend(loc="lower left", fontsize="small")

    bar = fig.colorbar(filled, ax=axes[1])
    bar.set_label("Quasi-log-likelihood less its maximum")
    fig.tight_layout()
    fit_to_width(fig, 1.0)
    _save(fig, out)


def _barycentric_weight_matrix(
    weights: pd.DataFrame, diagnostics: pd.DataFrame, universe_csv: Path
) -> tuple[np.ndarray, list[str]]:
    """Pivot one validated leave-one-out artifact into a ticker-square matrix."""
    if not diagnostics["converged"].all():
        message = "barycentric-mixture exhibit requires converged target projections"
        raise ValueError(message)

    targets = (
        weights.drop_duplicates("target_index")
        .sort_values("target_index")["target"]
        .tolist()
    )
    target_set = set(targets)
    candidate_set = set(weights["candidate"])
    if target_set != candidate_set:
        message = "barycentric-mixture artifact has inconsistent target coverage"
        raise ValueError(message)

    universe = pd.read_csv(universe_csv)
    if "Symbol" not in universe:
        message = "barycentric-mixture universe metadata requires a Symbol column"
        raise ValueError(message)
    tickers = sorted(target_set)
    if not target_set.issubset(set(universe["Symbol"])):
        message = "barycentric-mixture artifact contains tickers missing from universe"
        raise ValueError(message)

    if weights.duplicated(["target", "candidate"]).any():
        message = "barycentric-mixture artifact repeats a target-candidate pair"
        raise ValueError(message)
    square = weights.pivot_table(
        index="target",
        columns="candidate",
        values="weight",
        aggfunc="sum",
        sort=False,
    )
    matrix = (
        square.reindex(index=tickers, columns=tickers, fill_value=0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0):
        message = "barycentric-mixture rows must be nonnegative and unit-sum"
        raise ValueError(message)
    if not np.allclose(np.diag(matrix), 0.0):
        message = "barycentric-mixture rows must have zero self-weight"
        raise ValueError(message)
    return matrix, tickers


def _five_largest_weights(weights: np.ndarray) -> np.ndarray:
    """Keep each row's five largest actual barycentric coefficients for display."""
    if weights.ndim != _MATRIX_DIMENSIONS or weights.shape[0] != weights.shape[1]:
        message = "barycentric-mixture display requires a square weight matrix"
        raise ValueError(message)
    if weights.shape[0] <= 1:
        message = "barycentric-mixture display requires at least two firms"
        raise ValueError(message)
    count = min(_BARYCENTRIC_TOP_COUNT, weights.shape[1] - 1)
    indices = np.argsort(-weights, axis=1, kind="stable")[:, :count]
    display = np.zeros_like(weights)
    display[np.arange(weights.shape[0])[:, None], indices] = weights[
        np.arange(weights.shape[0])[:, None], indices
    ]
    return display


def _render_barycentric_mixture_map(
    output_file: Path, weights: np.ndarray, tickers: list[str]
) -> None:
    """Render actual target-anchored weights in a proposal-style matrix display."""
    _style()
    displayed = _five_largest_weights(weights)
    effective_count = 1.0 / np.sum(np.square(weights), axis=1)
    top_five_mass = displayed.sum(axis=1)
    count = len(tickers)
    positions = np.arange(count)
    label_positions = positions[::2]

    fig = plt.figure(figsize=figsize(1.0, 0.94))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 5.4),
        width_ratios=(6.8, 1.5, 1.5),
        hspace=0.24,
        wspace=0.34,
    )
    ax_incoming = fig.add_subplot(grid[0, 0])
    ax_weights = fig.add_subplot(grid[1, 0])
    ax_effective = fig.add_subplot(grid[1, 1])
    ax_mass = fig.add_subplot(grid[1, 2])
    colorbar_axis = fig.add_subplot(grid[0, 1:])

    # This is a sparse nonnegative map: white means an unshown coefficient,
    # while the five retained coefficients should become darker as their mass
    # rises. Reversed Cividis is bundled with Matplotlib, monotone in lightness,
    # legible for common colour-vision deficiencies, and remains ordered in
    # grayscale print.
    heatmap_cmap = mpl.colormaps["cividis_r"].copy()
    heatmap_cmap.set_bad("white")
    image = ax_weights.imshow(
        np.ma.masked_equal(displayed, 0.0),
        aspect="auto",
        cmap=heatmap_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=float(weights.max()),
    )
    ax_weights.set_title(r"Five largest actual $W^\flat$ weights per target")
    ax_weights.set_xlabel("Source firm (every second ticker labelled)")
    ax_weights.set_ylabel("Target firm (every second ticker labelled)")
    ax_weights.set_xticks(
        label_positions, [tickers[index] for index in label_positions]
    )
    ax_weights.set_yticks(
        label_positions, [tickers[index] for index in label_positions]
    )
    ax_weights.tick_params(axis="x", labelrotation=90, labelsize=5.8)
    ax_weights.tick_params(axis="y", labelsize=5.8)
    categorical_axis(ax_weights, "both")

    ax_incoming.bar(positions, weights.sum(axis=0), color=_C["blue"], width=0.82)
    ax_incoming.set_title(r"Incoming $W^\flat$ mass", fontsize=8.8)
    ax_incoming.set_ylabel("sum")
    ax_incoming.set_xlim(-0.5, count - 0.5)
    ax_incoming.set_xticks([])
    categorical_axis(ax_incoming, "x")

    for axis, values, title, xlabel in (
        (
            ax_effective,
            effective_count,
            "Effective\nsource count",
            r"$1/\sum_j (W^\flat_{ij})^2$",
        ),
        (
            ax_mass,
            top_five_mass,
            "Top-five\nmass shown",
            r"$\sum_{j\in T_5(i)} W^\flat_{ij}$",
        ),
    ):
        axis.barh(positions, values, height=0.78, color=_C["blue"], zorder=2)
        axis.axvline(float(np.median(values)), color=_C["gray"], linestyle=":", lw=0.7)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlabel(xlabel)
        axis.set_ylim(count - 0.5, -0.5)
        axis.set_yticks([])
        axis.grid(axis="x", linestyle=":", alpha=0.4)

    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"Actual barycentric weight $W^\flat_{ij}$")
    colorbar_axis.xaxis.set_ticks_position("top")
    colorbar_axis.xaxis.set_label_position("top")
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.12, top=0.90)
    fit_to_width(fig, 1.0)
    _save(fig, output_file)


def render_barycentric_mixture_map(paths: BarycentricMixturePaths) -> None:
    """Render the Paper 5 empirical target-anchored W2 mixture operator."""
    weights, diagnostics, summary = read_target_projection_artifact(
        paths.barycentre_dir,
        expected_identity={
            "arm_id": "wasserstein_w2_loo",
            "provider_id": "qwen3-embedding-8b",
            "representation_id": "qwen3-embedding-8b-unit",
            "geometry": "wasserstein_w2",
            "feasible_set": "simplex_nonnegative",
        },
    )
    if summary.get("solver") != "wasserstein_w2_target":
        message = "barycentric-mixture exhibit requires the W2 target solver"
        raise ValueError(message)
    matrix, tickers = _barycentric_weight_matrix(
        weights, diagnostics, paths.universe_csv
    )
    paths.output_file.parent.mkdir(parents=True, exist_ok=True)
    _render_barycentric_mixture_map(paths.output_file, matrix, tickers)
    logger.info("Rendered Paper 5 barycentric-mixture map over %d firms", len(tickers))
