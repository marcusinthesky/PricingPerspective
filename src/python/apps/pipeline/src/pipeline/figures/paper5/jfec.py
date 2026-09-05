"""Native-width JFEC adapters for Paper 5's cached empirical figures.

The venue exception is feature-local and consumes only governed artifacts that
already exist.  The small amount of drawing duplication is deliberate: importing
private helpers from the shared-measure module would couple a 124 mm presentation
change back into the Monte-Carlo and two-field estimation fingerprints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import numpy as np
import pandas as pd

from pipeline.figures._common import _C, Grid, categorical_axis, figsize, figure
from pipeline.figures.paper5._sizing import JFEC_WIDTH_FRACTION
from pipeline.figures.paper5.figures import TwoFieldRegion
from pipeline.io.barycentre_artifacts import read_target_projection_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.contour import ContourSet

logger = logging.getLogger(__name__)

_ZOOM_FLOOR = -20.0
_CONTEXT_FLOOR = -4000.0
_QUASI_LOGLIK_CONTOUR_DROP = -2.9957
_SURFACE_LEVELS = 11
_ZOOM_MARGIN_FRACTION = 0.25
_ZOOM_MIN_MARGIN = 0.01
_BARYCENTRIC_TOP_COUNT = 5
_MATRIX_DIMENSIONS = 2
_JFEC_RASTER_DPI = 300


@dataclass(frozen=True, slots=True)
class JfecBarycentricFigurePaths:
    """Cached W2 projection inputs and one native-width JFEC output."""

    barycentre_dir: Path
    universe_csv: Path
    output_file: Path


@dataclass(frozen=True, slots=True)
class JfecTwoFieldFigurePaths:
    """Cached two-field inputs and one native-width JFEC output."""

    two_field_dir: Path
    output_file: Path


def _record(value: object, label: str) -> dict[str, object]:
    """Validate one JSON object used by the two-field figure adapter."""
    if not isinstance(value, dict):
        message = f"two-field figure requires a {label} object"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _number(record: dict[str, object], key: str) -> float:
    """Read one finite numeric field from a two-field results record."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"two-field figure requires numeric field {key!r}"
        raise TypeError(message)
    number = float(value)
    if not np.isfinite(number):
        message = f"two-field figure requires finite field {key!r}"
        raise ValueError(message)
    return number


def _two_field_region(two_field_dir: Path) -> TwoFieldRegion:
    """Rehydrate the published two-field surface without refitting the model."""
    payload = _record(
        json.loads((two_field_dir / "results.json").read_text(encoding="utf-8")),
        "results",
    )
    primary_period = payload.get("primary_period")
    if not isinstance(primary_period, str):
        message = "two-field figure requires a primary_period string"
        raise TypeError(message)
    records = _record(payload.get("results"), "results mapping")
    primary = _record(records.get(primary_period), "primary-period result")
    joint = _record(primary.get("joint"), "joint result")
    boundaries = _record(primary.get("boundaries"), "boundaries mapping")
    w_flat = _record(boundaries.get("w_flat"), "W-flat boundary")
    w_news = _record(boundaries.get("w_co_mentions"), "news-link boundary")

    surface = pd.read_parquet(two_field_dir / "surface.parquet")
    bootstrap = pd.read_parquet(two_field_dir / "bootstrap.parquet")
    surface_columns = {"rho_b", "rho_n", "loglik"}
    bootstrap_columns = {"rho_b", "rho_n"}
    if not surface_columns.issubset(surface.columns):
        message = "two-field surface artifact lacks rho_b, rho_n, or loglik"
        raise ValueError(message)
    if not bootstrap_columns.issubset(bootstrap.columns):
        message = "two-field bootstrap artifact lacks rho_b or rho_n"
        raise ValueError(message)

    return TwoFieldRegion(
        surface_rho_b=surface["rho_b"].to_numpy(dtype=np.float64),
        surface_rho_n=surface["rho_n"].to_numpy(dtype=np.float64),
        surface_loglik=surface["loglik"].to_numpy(dtype=np.float64),
        bootstrap_rho_b=bootstrap["rho_b"].to_numpy(dtype=np.float64),
        bootstrap_rho_n=bootstrap["rho_n"].to_numpy(dtype=np.float64),
        joint_rho_b=_number(joint, "rho_b"),
        joint_rho_n=_number(joint, "rho_n"),
        field_b_boundary=_number(w_flat, "rho_b"),
        field_n_boundary=_number(w_news, "rho_n"),
    )


def _draw_two_field_panel(
    ax: Axes,
    region: TwoFieldRegion,
    *,
    floor: float,
    zoomed: bool,
) -> ContourSet:
    """Draw one JFEC view without depending on private shared-measure helpers."""
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
            label="quasi-log-likelihood\ncontour (drop 2.9957)",
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


def _zoom_limits(values: object, centre: float) -> tuple[float, float]:
    """Bracket a bootstrap coordinate with a margin, clipped below at zero."""
    array = np.asarray(values, dtype=np.float64)
    lower = min(float(array.min()), centre)
    upper = max(float(array.max()), centre)
    margin = max(_ZOOM_MIN_MARGIN, _ZOOM_MARGIN_FRACTION * (upper - lower))
    return max(0.0, lower - margin), upper + margin


def render_jfec_two_field_region(region: TwoFieldRegion, output_file: Path) -> None:
    """Render the two-field exhibit at JFEC's native 124 mm measure."""
    with figure(
        output_file,
        figsize(JFEC_WIDTH_FRACTION, 3.2 / 7.0),
        Grid(1, 2),
        frac=JFEC_WIDTH_FRACTION,
    ) as (fig, axes):
        _draw_two_field_panel(axes[0], region, floor=_CONTEXT_FLOOR, zoomed=False)
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel(r"News-link channel $\rho_N$")
        axes[0].set_title("Admissible region", fontsize=9)
        axes[0].legend(loc="upper right", fontsize="small")

        filled = _draw_two_field_panel(axes[1], region, floor=_ZOOM_FLOOR, zoomed=True)
        axes[1].set_xlim(*_zoom_limits(region.bootstrap_rho_b, region.joint_rho_b))
        axes[1].set_ylim(*_zoom_limits(region.bootstrap_rho_n, region.joint_rho_n))
        axes[1].set_title("Joint estimate and resample cloud", fontsize=9)
        axes[1].legend(
            loc="lower left",
            fontsize=6.5,
            handlelength=2.4,
            labelspacing=0.3,
        )

        bar = fig.colorbar(filled, ax=axes[1], fraction=0.06, pad=0.04)
        bar.set_label("Quasi-log-likelihood drop", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        fig.tight_layout()


def _barycentric_weight_matrix(
    weights: pd.DataFrame, diagnostics: pd.DataFrame, universe_csv: Path
) -> tuple[np.ndarray, list[str]]:
    """Validate and pivot one leave-one-out artifact into a ticker-square matrix."""
    if not diagnostics["converged"].all():
        message = "barycentric-mixture exhibit requires converged target projections"
        raise ValueError(message)
    targets = (
        weights.drop_duplicates("target_index")
        .sort_values("target_index")["target"]
        .tolist()
    )
    target_set = set(targets)
    if target_set != set(weights["candidate"]):
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
    """Keep each row's five largest actual coefficients for display."""
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


def render_jfec_barycentric_mixture_map(
    output_file: Path, weights: np.ndarray, tickers: list[str]
) -> None:
    """Render the actual target-anchored weights at the 124 mm measure."""
    with (
        mpl.rc_context(),
        figure(
            output_file,
            figsize(JFEC_WIDTH_FRACTION, 0.94),
            frac=JFEC_WIDTH_FRACTION,
        ) as (fig, placeholder),
    ):
        # OUP requires raster components at no less than 300 dpi at final size.
        # Scope the override to this raster-bearing JFEC leaf; the shared figure
        # contract deliberately remains at 200 dpi for unrelated outputs.
        mpl.rcParams["savefig.dpi"] = _JFEC_RASTER_DPI
        placeholder.remove()
        displayed = _five_largest_weights(weights)
        effective_count = 1.0 / np.sum(np.square(weights), axis=1)
        top_five_mass = displayed.sum(axis=1)
        count = len(tickers)
        positions = np.arange(count)
        label_positions = positions[::2]
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
            axis.axvline(
                float(np.median(values)),
                color=_C["gray"],
                linestyle=":",
                lw=0.7,
            )
            axis.set_title(title, fontsize=8.5)
            axis.set_xlabel(xlabel)
            axis.set_ylim(count - 0.5, -0.5)
            axis.set_yticks([])
            axis.grid(axis="x", linestyle=":", alpha=0.4)

        colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label(r"Barycentric weight $W^\flat_{ij}$", fontsize=8)
        colorbar_axis.xaxis.set_ticks_position("top")
        colorbar_axis.xaxis.set_label_position("top")
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.12, top=0.90)


def render_paper5_jfec_barycentric_figure(
    paths: JfecBarycentricFigurePaths,
) -> None:
    """Render the native-width mixture map from its cached W2 artifact."""
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
    render_jfec_barycentric_mixture_map(paths.output_file, matrix, tickers)
    logger.info("Rendered Paper 5 JFEC barycentric map over %d firms", len(tickers))


def render_paper5_jfec_two_field_figure(paths: JfecTwoFieldFigurePaths) -> None:
    """Render the native-width two-field exhibit from its cached artifacts."""
    render_jfec_two_field_region(
        _two_field_region(paths.two_field_dir), paths.output_file
    )
