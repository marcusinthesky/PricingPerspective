"""Conceptual illustrations for Paper 5's exposure and operator arguments.

The figures are deliberately schematic or simulation-based. They make the
economic sequence visible without presenting synthetic geometry as evidence
from the 100-firm panel. The retained rank-one illustration also keeps the
numerical warning that the strong-interaction limit approaches a singular
resolvent.

Outputs:

``rank_one_collapse.pgf``
    Heatmaps of ``(1-rho)^2 Sigma_SAR`` and a condition-number curve.
``spatial_exposure_arc.pgf``
    Observed, latent, theorem, and estimated objects in the manuscript arc.
``w2_operator_construction.pgf``
    One restricted target-anchored assignment and simplex row construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment, minimize

mpl.use("Agg")
from matplotlib import pyplot as plt

from pipeline.figures._common import (
    _C,
    _TINT,
    Grid,
    _save,
    _style,
    figsize,
    figure,
    fit_to_width,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Float = NDArray[np.float64]

_DEFAULT_RHOS = np.array([0.0, 0.5, 0.9, 0.99, 0.9999], dtype=np.float64)
_MIN_FIRMS = 3

# Exposure-arc schematic. Boxes are sized in data units from a point-valued label, so
# the padding is data units and the arrow floor is points -- the two are not
# interchangeable on an axes that is 5.9in across and 3.0in down.
_BOX_FONTSIZE = 8.0
_EDGE_LABEL_FONTSIZE = 6.5
_LANE_FONTSIZE = 8.5
_BOX_PAD_X = 0.020
_BOX_PAD_Y = 0.030
_AXES_MARGIN = 0.005
# Below this a direction component is axis-parallel and constrains no border.
_DIRECTION_EPS = 1e-12
# An `-|>` head at `mutation_scale=8` is ~8pt; below this a connector is all head.
_MIN_ARROW_POINTS = 12.0
# Fixed clearance between horizontally adjacent boxes, independent of label metrics.
_CHAIN_GAP = 0.050
# Vertical bands. Boxes on bands closer together than one box height must not overlap
# in x; `_assert_arc_fits` checks exactly that.
_ARC_ROW_Y = {
    "operator": 0.83,
    "cloud": 0.72,
    "qmle": 0.72,
    "rho": 0.72,
    "returns": 0.60,
    "lambda": 0.44,
    "primitive": 0.16,
    "realized": 0.16,
}

# W2 operator construction. One scale for both coordinates, so the assignment panel
# cannot squash the cloud it is matching.
_ASSIGNMENT_SCALE = 0.36
_ASSIGNMENT_TARGET_X = -0.70
_ASSIGNMENT_PEER_X = 0.58
_ASSIGNMENT_LABEL_GAP = 0.10
_ROW_BAR_WIDTH = 0.62
_ROW_AXIS_X = -0.92


class IllustrationInputError(ValueError):
    """Report invalid inputs to Paper 5 conceptual illustrations."""

    @classmethod
    def too_few_firms(cls) -> IllustrationInputError:
        """Construct the minimum-universe error."""
        return cls("n must be at least 3")

    @classmethod
    def nonpositive_bandwidth(cls) -> IllustrationInputError:
        """Construct the invalid-bandwidth error."""
        return cls("bandwidth must be positive")

    @classmethod
    def invalid_rho(cls) -> IllustrationInputError:
        """Construct the invalid-interaction-strength error."""
        return cls("the illustration requires 0 <= rho < 1")

    @classmethod
    def invalid_rho_grid(cls) -> IllustrationInputError:
        """Construct the invalid-rho-grid error."""
        return cls("rhos must be a non-empty one-dimensional array")

    @classmethod
    def rho_grid_out_of_bounds(cls) -> IllustrationInputError:
        """Construct the rho-grid bounds error."""
        return cls("every rho must satisfy 0 <= rho < 1")


class IllustrationComputationError(RuntimeError):
    """Report a failed numerical illustration computation."""

    @classmethod
    def optimizer_failed(cls, detail: object) -> IllustrationComputationError:
        """Construct an optimizer-failure error with its native detail."""
        return cls(f"simplex QP failed: {detail}")

    @classmethod
    def nonfinite_weights(cls) -> IllustrationComputationError:
        """Construct the nonfinite-weights error."""
        return cls("simplex QP returned non-finite weights")

    @classmethod
    def missing_heatmap(cls) -> IllustrationComputationError:
        """Construct the missing-heatmap error."""
        return cls("rank-one illustration did not produce a heatmap")

    @classmethod
    def box_overflows_axes(cls, key: str) -> IllustrationComputationError:
        """Construct the schematic-overflow error."""
        return cls(f"schematic box {key!r} does not fit inside its axes")

    @classmethod
    def boxes_overlap(cls, left: str, right: str) -> IllustrationComputationError:
        """Construct the colliding-boxes error."""
        return cls(f"schematic boxes {left!r} and {right!r} overlap")

    @classmethod
    def arrow_too_short(cls, points: float) -> IllustrationComputationError:
        """Construct the degenerate-connector error."""
        return cls(
            f"schematic arrow is {points:.1f}pt long, shorter than its own head; "
            f"the two boxes it joins are touching"
        )


@dataclass(frozen=True)
class RankOneCollapseData:
    """Inputs and outputs used by the rank-one-collapse illustration."""

    w: Float
    v: Float
    rhos: Float
    covariances: Float
    condition_rhos: Float
    condition_numbers: Float


@dataclass(frozen=True)
class W2OperatorConstructionData:
    """Synthetic inputs and outputs for one target-anchored operator row."""

    target: Float
    candidates: Float
    aligned_candidates: Float
    weights: Float
    reconstruction: Float
    row: Float
    self_index: int


def make_symmetric_row_stochastic_w(n: int = 24, bandwidth: float = 2.5) -> Float:
    """Construct a primitive, symmetric row-stochastic interaction matrix.

    The circular distance makes the matrix doubly stochastic, so its stationary
    distribution is uniform.  The diagonal is set to zero to keep the toy
    interaction operator close to the Paper 5 construction; all off-diagonal
    entries remain positive, making the chain primitive.
    """
    if n < _MIN_FIRMS:
        raise IllustrationInputError.too_few_firms()
    if bandwidth <= 0.0:
        raise IllustrationInputError.nonpositive_bandwidth()

    indices = np.arange(n)
    distance = np.abs(indices[:, None] - indices[None, :])
    circular_distance = np.minimum(distance, n - distance)
    weights = np.exp(-((circular_distance / bandwidth) ** 2))
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


def rescaled_covariance(rho: float, w: Float, v: Float) -> tuple[Float, float]:
    """Return ``(1-rho)^2 Sigma_SAR`` and ``cond(I-rho W)``.

    ``np.linalg.solve`` avoids explicitly forming an inverse, but it does not
    make the problem well-conditioned.  The condition number is therefore
    returned as a first-class output and plotted by
    :func:`render_rank_one_collapse`.
    """
    if not 0.0 <= rho < 1.0:
        raise IllustrationInputError.invalid_rho()
    n = w.shape[0]
    identity = np.eye(n)
    resolvent = identity - rho * w
    condition_number = float(np.linalg.cond(resolvent))
    multiplier = (1.0 - rho) * np.linalg.solve(resolvent, identity)
    covariance = np.asarray(multiplier @ v @ multiplier.T, dtype=np.float64)
    return covariance, condition_number


def simulate_rank_one_collapse(
    n: int = 24, rhos: Float | None = None
) -> RankOneCollapseData:
    """Simulate the strong-interaction limit with a uniform stationary law."""
    selected_rhos = (
        np.asarray(rhos, dtype=np.float64) if rhos is not None else _DEFAULT_RHOS.copy()
    )
    if selected_rhos.ndim != 1 or selected_rhos.size == 0:
        raise IllustrationInputError.invalid_rho_grid()
    if np.any((selected_rhos < 0.0) | (selected_rhos >= 1.0)):
        raise IllustrationInputError.rho_grid_out_of_bounds()

    w = make_symmetric_row_stochastic_w(n=n)
    # Heterogeneous idiosyncratic variances make the path to the uniform limit
    # visible; the uniform stationary distribution still makes the limiting
    # rank-one matrix spatially constant.
    v = np.diag(np.linspace(0.5, 1.5, n, dtype=np.float64))
    covariances = []
    for rho in selected_rhos:
        covariance, _ = rescaled_covariance(float(rho), w, v)
        covariances.append(covariance)

    condition_rhos = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.99, 120),
                1.0 - np.logspace(-2.0, -5.0, 80),
            ]
        )
    )
    condition_numbers = np.array(
        [rescaled_covariance(float(rho), w, v)[1] for rho in condition_rhos],
        dtype=np.float64,
    )
    return RankOneCollapseData(
        w=w,
        v=v,
        rhos=selected_rhos,
        covariances=np.stack(covariances),
        condition_rhos=condition_rhos,
        condition_numbers=condition_numbers,
    )


def simulate_w2_operator_construction() -> W2OperatorConstructionData:
    """Solve a transparent target-anchored assignment and simplex example.

    The example mirrors the empirical operation order but not its data. Each
    candidate cloud is first aligned separately to the target with a balanced
    assignment. The simplex QP is solved only after those maps are fixed.
    """
    # Roughly isotropic on purpose. The previous support spanned 1.82 in x against
    # 0.66 in y, and no panel can show an assignment between two clouds that flat:
    # every matching line comes out near-horizontal and near-parallel, which is the
    # one thing panel B exists to make visible.
    target = np.array(
        [
            [-0.72, -0.46],
            [-0.58, 0.55],
            [0.02, 0.12],
            [-0.06, -0.68],
            [0.70, 0.58],
            [0.64, -0.30],
        ],
        dtype=np.float64,
    )
    permutations = np.array(
        [
            [2, 0, 5, 3, 1, 4],
            [5, 4, 0, 1, 3, 2],
            [1, 5, 3, 0, 4, 2],
        ]
    )
    # Chosen so the origin sits *inside* the triangle these three offsets span. For
    # clouds that are near-translates of the target the simplex QP reduces to
    # projecting the origin onto that hull, so offsets whose hull excludes it force a
    # vertex solution and a peer weighted exactly zero -- which is what the previous
    # values did, and it makes a poor illustration of a peer *mixture*.
    offsets = np.array(
        [[-0.126, -0.056], [0.070, -0.098], [0.149, 0.246]],
        dtype=np.float64,
    )
    candidates = np.stack(
        [
            target[permutation] + offset
            for permutation, offset in zip(permutations, offsets, strict=True)
        ]
    )
    candidates[0, :, 1] += np.array(
        [0.06, -0.04, 0.05, -0.05, 0.04, -0.06], dtype=np.float64
    )
    aligned = np.empty_like(candidates)
    for candidate_index, candidate in enumerate(candidates):
        squared_cost = np.sum(
            (target[:, None, :] - candidate[None, :, :]) ** 2,
            axis=2,
        )
        target_indices, candidate_indices = linear_sum_assignment(squared_cost)
        aligned[candidate_index, target_indices] = candidate[candidate_indices]

    def objective(weights: Float) -> float:
        reconstruction = np.tensordot(weights, aligned, axes=(0, 0))
        residual = target - reconstruction
        return float(0.5 * np.mean(np.sum(residual**2, axis=1)))

    candidate_count = candidates.shape[0]
    result = minimize(
        objective,
        x0=np.full(candidate_count, 1.0 / candidate_count, dtype=np.float64),
        bounds=[(0.0, 1.0)] * candidate_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 5_000, "disp": False},
    )
    if not result.success:
        raise IllustrationComputationError.optimizer_failed(result.message)
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    if not np.all(np.isfinite(weights)):
        raise IllustrationComputationError.nonfinite_weights()
    weights /= weights.sum()
    reconstruction = np.asarray(
        np.tensordot(weights, aligned, axes=(0, 0)), dtype=np.float64
    )
    self_index = 1
    row = np.insert(weights, self_index, 0.0)
    return W2OperatorConstructionData(
        target=target,
        candidates=candidates,
        aligned_candidates=aligned,
        weights=weights,
        reconstruction=reconstruction,
        row=row,
        self_index=self_index,
    )


def _format_rho(rho: float) -> str:
    """Format heatmap rho values without implying that rho=1 was evaluated."""
    return f"$\\rho={rho:.4g}$"


def render_rank_one_collapse(
    output_path: Path, data: RankOneCollapseData | None = None
) -> None:
    """Render the covariance heatmaps and the condition-number diagnostic."""
    _style()
    illustration = data or simulate_rank_one_collapse()
    n_heatmaps = illustration.rhos.size
    fig = plt.figure(figsize=figsize(0.8, 4.9 / 10.2))
    grid = fig.add_gridspec(
        2,
        n_heatmaps,
        height_ratios=(3.0, 1.8),
        hspace=0.52,
        wspace=0.12,
        left=0.13,
        right=0.87,
        top=0.90,
        bottom=0.13,
    )
    heat_axes = [fig.add_subplot(grid[0, i]) for i in range(n_heatmaps)]
    condition_ax = fig.add_subplot(grid[1, :])

    vmin = float(np.min(illustration.covariances))
    vmax = float(np.max(illustration.covariances))
    image = None
    for i, (ax, rho, covariance) in enumerate(
        zip(heat_axes, illustration.rhos, illustration.covariances, strict=True)
    ):
        image = ax.imshow(
            covariance,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(_format_rho(float(rho)), pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("firm $i$")
        ax.set_xlabel("firm $j$", labelpad=2)

    if image is None:
        raise IllustrationComputationError.missing_heatmap()
    colorbar = fig.colorbar(image, ax=heat_axes, fraction=0.018, pad=0.02)
    colorbar.set_label(r"$(1-\rho)^2\Sigma_{SAR,ij}$", rotation=270, labelpad=13)
    colorbar.ax.tick_params(labelsize=7)

    condition_ax.semilogy(
        illustration.condition_rhos,
        illustration.condition_numbers,
        color=_C["vermilion"],
        linewidth=1.4,
        label=r"$\kappa_2(I-\rho W)$",
    )
    heatmap_conditions = np.array(
        [
            np.linalg.cond(np.eye(illustration.w.shape[0]) - rho * illustration.w)
            for rho in illustration.rhos
        ],
        dtype=np.float64,
    )
    condition_ax.semilogy(
        illustration.rhos,
        heatmap_conditions,
        linestyle="none",
        marker="o",
        markersize=3.5,
        color=_C["black"],
        label="heatmap values",
    )
    condition_ax.axvline(1.0, color=_C["gray"], linestyle="--", linewidth=0.8)
    condition_ax.text(
        0.995,
        0.96,
        r"$\rho=1$: singular",
        transform=condition_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=_C["gray"],
    )
    condition_ax.set_xlim(0.0, 1.01)
    condition_ax.set_xlabel(r"Interaction strength $\rho$ (endpoint $1$ not evaluated)")
    # Short label: the full expression is already carried by the legend, and the long
    # rotated form was what tight-bbox cropped once the canvas shrank to print size.
    condition_ax.set_ylabel("Condition number", fontsize=8)
    condition_ax.set_title(
        "The rank-one limit is accompanied by a resolvent singularity"
    )
    condition_ax.legend(loc="upper left", ncol=2)
    condition_ax.grid(axis="y", alpha=0.2)
    # At `axes.titlesize` rather than the default `large`: this suptitle is the
    # widest artist in the figure, so at 10.8pt it, not the axes, set the saved
    # bounding box — and being point-valued it does not respond to `fit_to_width`.
    fig.suptitle(
        r"Strong-interaction limit: $(1-\rho)^2\Sigma_{SAR}$ becomes spatially uniform",
        fontsize=10,
        y=0.99,
    )
    fit_to_width(fig, 0.8)
    _save(fig, output_path)


@dataclass(frozen=True)
class _BoxSpec:
    """How one exposure-arc box is labelled and coloured. Position is solved."""

    label: str
    fill: str
    edge: str


@dataclass(frozen=True)
class _DiagramBox:
    """One *placed* exposure-arc box, in axes data units."""

    centre: tuple[float, float]
    width: float
    height: float

    def edge_toward(self, target: tuple[float, float]) -> tuple[float, float]:
        """Return where the ray from this centre to ``target`` leaves the border."""
        x, y = self.centre
        dx, dy = target[0] - x, target[1] - y
        reach = min(
            half / abs(delta) if abs(delta) > _DIRECTION_EPS else np.inf
            for half, delta in ((self.width / 2.0, dx), (self.height / 2.0, dy))
        )
        if not np.isfinite(reach):
            return self.centre
        return (x + dx * reach, y + dy * reach)


def _measure_label(ax: plt.Axes, label: str, fontsize: float) -> tuple[float, float]:
    """Return ``label``'s rendered ``(width, height)`` in axes data units.

    Measured with the Agg renderer rather than assumed. Agg resolves
    ``font.family: serif`` to its own serif face while the PGF backend hands the
    string to the document's font, so the measurement runs a few per cent wide --
    the safe direction for something that has to *contain* the text.
    """
    probe = ax.text(0.0, 0.0, label, ha="center", va="center", fontsize=fontsize)
    extent = probe.get_window_extent()
    probe.remove()
    corners = ax.transData.inverted().transform(
        [(extent.x0, extent.y0), (extent.x1, extent.y1)]
    )
    return (
        abs(float(corners[1][0] - corners[0][0])),
        abs(float(corners[1][1] - corners[0][1])),
    )


def _solve_exposure_arc(
    ax: plt.Axes, specs: dict[str, _BoxSpec]
) -> dict[str, _DiagramBox]:
    r"""Solve box geometry from measured label widths and *structural* gaps.

    Boxes are sized in data units while their labels are sized in points, so a
    hard-coded width truncates silently as soon as a label is edited: ``Frozen peer
    geometry`` spilled out of both sides of its rounded rectangle, and
    ``Model-implied`` reached data ``x=1.017`` against an ``xlim`` of ``1.0``, so
    matplotlib clipped 0.10in of that box -- right border included -- out of the
    saved PGF.

    Measuring fixes that, but a measured width cannot be allowed to set a *position*.
    Under the PGF backend ``get_window_extent`` routes through LaTeX, and importing
    ``libpysal``/``spreg`` elsewhere in the same interpreter moves the numbers it
    returns by ~14 per cent (``tests/papers/paper5/test_paper5_validate_estimators``
    is enough to do it). Centres are therefore derived by walking outward from the
    two axes edges with a fixed ``_CHAIN_GAP``, so every gap survives that swing and
    only the boxes themselves breathe. ``_assert_no_overlap`` is the backstop for the
    one thing that can still go wrong -- the chain no longer fitting across.
    """
    measured = {
        key: _measure_label(ax, spec.label, _BOX_FONTSIZE)
        for key, spec in specs.items()
    }
    width = {key: size[0] + 2.0 * _BOX_PAD_X for key, size in measured.items()}
    height = max(size[1] for size in measured.values()) + 2.0 * _BOX_PAD_Y
    left = _AXES_MARGIN
    right = 1.0 - _AXES_MARGIN

    # Construction chain grows rightward from the left edge; the estimation column is
    # right-aligned and the QMLE chain grows leftward off it, so both margins close
    # on the axes rather than on whatever the metrics happened to be.
    cloud_x = left + width["cloud"] / 2.0
    operator_x = left + width["cloud"] + _CHAIN_GAP + width["operator"] / 2.0
    estimated_x = right - max(width["rho"], width["lambda"]) / 2.0
    qmle_x = (
        estimated_x
        - max(width["rho"], width["lambda"]) / 2.0
        - _CHAIN_GAP
        - width["qmle"] / 2.0
    )
    returns_x = qmle_x - width["qmle"] / 2.0 - _CHAIN_GAP - width["returns"] / 2.0
    # The latent pair hangs directly under the operator, which keeps the peer-
    # adjustment arrow vertical and clear of the returns box to its right.
    realized_x = operator_x
    primitive_x = (
        realized_x - width["realized"] / 2.0 - _CHAIN_GAP - width["primitive"] / 2.0
    )
    centre_x = {
        "cloud": cloud_x,
        "operator": operator_x,
        "returns": returns_x,
        "qmle": qmle_x,
        "rho": estimated_x,
        "lambda": estimated_x,
        "primitive": primitive_x,
        "realized": realized_x,
    }
    boxes = {
        key: _DiagramBox((centre_x[key], _ARC_ROW_Y[key]), width[key], height)
        for key in specs
    }
    _assert_arc_fits(boxes)
    return boxes


def _assert_arc_fits(boxes: dict[str, _DiagramBox]) -> None:
    """Fail the render rather than ship a schematic that overlaps or is cropped."""
    for key, box in boxes.items():
        if (
            box.centre[0] - box.width / 2.0 < _AXES_MARGIN - _DIRECTION_EPS
            or box.centre[0] + box.width / 2.0 > 1.0 - _AXES_MARGIN + _DIRECTION_EPS
        ):
            raise IllustrationComputationError.box_overflows_axes(key)
    for (left_key, left), (right_key, right) in combinations(boxes.items(), 2):
        shares_band = abs(left.centre[1] - right.centre[1]) < 0.5 * (
            left.height + right.height
        )
        overlaps_x = abs(left.centre[0] - right.centre[0]) < 0.5 * (
            left.width + right.width
        )
        if shares_band and overlaps_x:
            raise IllustrationComputationError.boxes_overlap(left_key, right_key)


def _draw_boxes(
    ax: plt.Axes, specs: dict[str, _BoxSpec], boxes: dict[str, _DiagramBox]
) -> None:
    """Draw the solved boxes.

    ``boxstyle`` carries no padding of its own: the drawn rectangle has to be exactly
    the ``_DiagramBox`` the connectors aim at, or every arrow lands short of, or
    inside, the border it was solved for.
    """
    for key, spec in specs.items():
        box = boxes[key]
        x, y = box.centre
        ax.add_patch(
            FancyBboxPatch(
                (x - box.width / 2.0, y - box.height / 2.0),
                box.width,
                box.height,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                facecolor=spec.fill,
                edgecolor=spec.edge,
                linewidth=1.0,
                zorder=3,
            )
        )
        ax.text(
            x, y, spec.label, ha="center", va="center", fontsize=_BOX_FONTSIZE, zorder=4
        )


def _diagram_arrow(
    ax: plt.Axes,
    start: _DiagramBox,
    end: _DiagramBox,
    label: str = "",
    *,
    theorem: bool,
    label_shift: float = 0.0,
) -> None:
    r"""Connect two boxes border to border, keeping the label clear of the shaft.

    Endpoints are solved on the two borders instead of being written down. The
    previous hand-written pairs had drifted out of step with the boxes they joined:
    the ``\hat\rho``-to-``\hat\lambda`` connector was 0.005 data units long against an
    8pt head, so it rendered as a blob on top of the ``\hat\rho`` glyph, and the
    return-bridge head stopped inside the returns box.

    Lengths and label offsets are computed in display space, because a schematic
    whose axes span one data unit across 5.9in and one down 2.8in has no single
    meaning for "0.03 long" or "perpendicular". The label is pushed along the segment
    normal *and* backed by an opaque box: a label parked on its own midpoint is a
    label with its own arrow struck through it, which is how ``transmission`` and
    ``return bridge`` were previously drawn.
    """
    tail = start.edge_toward(end.centre)
    head = end.edge_toward(start.centre)
    tail_px, head_px = ax.transData.transform([tail, head])
    delta = head_px - tail_px
    span_px = float(np.hypot(*delta))
    span_pt = span_px * 72.0 / float(ax.figure.dpi)
    if span_pt < _MIN_ARROW_POINTS:
        raise IllustrationComputationError.arrow_too_short(span_pt)
    color = _C["orange"] if theorem else _C["blue"]
    ax.add_patch(
        FancyArrowPatch(
            tail,
            head,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=1.0,
            linestyle="--" if theorem else "-",
            color=color,
            zorder=2,
        )
    )
    if not label:
        return
    normal = np.array([-delta[1], delta[0]]) / span_px
    offset_px = label_shift * float(ax.figure.dpi) / 72.0
    anchor = ax.transData.inverted().transform(
        0.5 * (tail_px + head_px) + offset_px * normal
    )
    ax.text(
        float(anchor[0]),
        float(anchor[1]),
        label,
        ha="center",
        va="center",
        fontsize=_EDGE_LABEL_FONTSIZE,
        color=color,
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def _lane_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    """Write one lane heading in the colour of the boxes it names.

    The fills already carry a three-way distinction -- observed, latent, estimated --
    that the arrow legend does not cover, so the heading is the only place a reader
    can learn what a green box means. The opaque backing keeps a heading legible where
    a transmission arrow passes behind it.
    """
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=_LANE_FONTSIZE,
        fontweight="bold",
        va="center",
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )


def render_spatial_exposure_arc(
    output_path: Path,
    *,
    width_fraction: float = 1.0,
) -> None:
    """Render the empirical and latent lanes of the paper's core architecture.

    The estimation pair is stacked rather than run on to the right of QMLE. Laying all
    six top-band boxes on one line needs about 1.05 data units once each is wide
    enough for its own label, against an axes one unit across; the previous version
    resolved that by overrunning the axes and losing the widest box's border to the
    clip. Stacking spends the empty lower-right quadrant instead.
    """
    with figure(
        output_path,
        figsize(width_fraction, 0.50),
        frac=width_fraction,
    ) as (fig, ax):
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        specs = {
            "cloud": _BoxSpec("Text laws\n$C_i$", _TINT["blue"], _C["blue"]),
            "operator": _BoxSpec(
                "Frozen peer geometry\n$W^\\flat$", _TINT["blue"], _C["blue"]
            ),
            "returns": _BoxSpec("Returns\n$r_t$", _TINT["blue"], _C["blue"]),
            "qmle": _BoxSpec("QMLE", _TINT["blue"], _C["blue"]),
            "rho": _BoxSpec("Estimated $\\hat\\rho$", _TINT["green"], _C["green"]),
            "lambda": _BoxSpec(
                "Model-implied\n$\\hat\\lambda=\\hat\\rho/(1-\\hat\\rho)$",
                _TINT["green"],
                _C["green"],
            ),
            "primitive": _BoxSpec(
                "Stand-alone\nexposure $\\xi_i$", _TINT["orange"], _C["orange"]
            ),
            "realized": _BoxSpec(
                "Peer-adjusted\nexposure $B_i$", _TINT["orange"], _C["orange"]
            ),
        }
        boxes = _solve_exposure_arc(ax, specs)
        _draw_boxes(ax, specs, boxes)

        _diagram_arrow(ax, boxes["cloud"], boxes["operator"], theorem=False)
        _diagram_arrow(ax, boxes["operator"], boxes["qmle"], theorem=False)
        _diagram_arrow(ax, boxes["returns"], boxes["qmle"], theorem=False)
        _diagram_arrow(ax, boxes["qmle"], boxes["rho"], theorem=False)
        _diagram_arrow(ax, boxes["rho"], boxes["lambda"], theorem=False)
        _diagram_arrow(
            ax,
            boxes["cloud"],
            boxes["primitive"],
            r"transmission $T(X_i,U_i)$",
            theorem=True,
            label_shift=13.0,
        )
        _diagram_arrow(ax, boxes["operator"], boxes["realized"], theorem=True)
        _diagram_arrow(ax, boxes["primitive"], boxes["realized"], theorem=True)
        _diagram_arrow(
            ax,
            boxes["realized"],
            boxes["returns"],
            r"return bridge $\langle B_i,F_t\rangle+e_{it}$",
            theorem=True,
            label_shift=-15.0,
        )

        # Attached to the pair rather than to the 0.06-unit arrow between them: as an
        # edge label it overhung both boxes and its flat sign was cut by a border.
        ax.text(
            0.5 * (boxes["primitive"].centre[0] + boxes["realized"].centre[0]),
            boxes["primitive"].centre[1] - boxes["primitive"].height / 2.0 - 0.045,
            r"$B=\rho(\lambda)W^\flat B+(1-\rho(\lambda))\xi$",
            ha="center",
            va="center",
            fontsize=_EDGE_LABEL_FONTSIZE,
            color=_C["orange"],
        )

        _lane_label(ax, 0.010, 0.955, "Observed / constructed", _C["blue"])
        _lane_label(ax, 0.010, 0.325, "Latent exposure model", _C["orange"])
        _lane_label(
            ax,
            boxes["rho"].centre[0] - boxes["rho"].width / 2.0,
            0.955,
            "Estimated",
            _C["green"],
        )

        # Parked left of centre: the estimation column is right-aligned to the axes,
        # so its heading owns the top-right corner whatever the labels measure.
        ax.plot([0.420, 0.470], [0.955, 0.955], color=_C["blue"], linewidth=1.1)
        ax.text(0.480, 0.955, "empirical", va="center", fontsize=6.6)
        ax.plot(
            [0.570, 0.620],
            [0.955, 0.955],
            color=_C["orange"],
            linewidth=1.1,
            linestyle="--",
        )
        ax.text(0.630, 0.955, "model", va="center", fontsize=6.6)


def _isometric_limits(
    fig: plt.Figure, ax: Axes, x_limits: tuple[float, float], y_centre: float
) -> None:
    """Put x and y on one visual scale without resizing the axes.

    ``set_aspect("equal")`` reaches the same scale either by shrinking the axes box
    (``adjustable="box"``, the default, which is what pulled panels A and C off the
    row's title baseline) or by overriding the limits it was handed
    (``adjustable="datalim"``, which warns and takes the headroom with it). Solving
    for the y-range that the panel's own width-to-height ratio implies does neither:
    every axes keeps the rectangle the grid gave it, and the surplus height stays
    where ``y_centre`` puts it -- above the cloud, for panel C's key.

    ``fit_to_width`` rescales the whole canvas afterwards, which is safe here because
    it scales both dimensions by the same factor and this ratio is all that matters.
    """
    figure_width, figure_height = fig.get_size_inches()
    position = ax.get_position()
    x_span = x_limits[1] - x_limits[0]
    y_span = (
        x_span * (position.height * figure_height) / (position.width * figure_width)
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(y_centre - 0.5 * y_span, y_centre + 0.5 * y_span)


def _draw_separate_assignments(
    ax: Axes,
    illustration: W2OperatorConstructionData,
    candidate_colors: tuple[str, str, str],
) -> None:
    """Draw one target copy and one fixed matching for each candidate peer.

    Both coordinates take the same scale. The previous version shrank ``y`` by 0.12
    against 0.25 in ``x`` and left the axes free to stretch on top of that, so the
    clouds printed as horizontal slivers and every matching line came out parallel:
    the panel asserted a point-to-point correspondence it gave the reader no way to
    see. Equal scaling here and ``set_aspect("equal")`` on the axes are both needed --
    either alone still distorts.
    """
    target_x = _ASSIGNMENT_TARGET_X + _ASSIGNMENT_SCALE * illustration.target[:, 0]
    # Row labels are set off the clouds' own extents, shared across rows so they stay
    # in one column. Hard-coding them put `$j_k$` at x=1.12 against a cloud that
    # reached 1.11, so the label sat on the last peer point.
    target_label_x = float(target_x.min()) - _ASSIGNMENT_LABEL_GAP
    peer_label_x = (
        _ASSIGNMENT_PEER_X
        + _ASSIGNMENT_SCALE * float(illustration.aligned_candidates[:, :, 0].max())
        + _ASSIGNMENT_LABEL_GAP
    )
    for peer_index, (candidate, color) in enumerate(
        zip(illustration.aligned_candidates, candidate_colors, strict=True),
        start=1,
    ):
        row_y = 3.0 - peer_index
        candidate_x = _ASSIGNMENT_PEER_X + _ASSIGNMENT_SCALE * candidate[:, 0]
        target_y = row_y + _ASSIGNMENT_SCALE * illustration.target[:, 1]
        candidate_y = row_y + _ASSIGNMENT_SCALE * candidate[:, 1]
        for left_x, left_y, right_x, right_y in zip(
            target_x, target_y, candidate_x, candidate_y, strict=True
        ):
            ax.plot(
                [left_x, right_x],
                [left_y, right_y],
                color=color,
                linewidth=0.55,
                alpha=0.6,
                zorder=1,
            )
        ax.scatter(
            target_x,
            target_y,
            s=11,
            color=_C["blue"],
            edgecolor=_C["black"],
            linewidth=0.3,
            zorder=2,
        )
        ax.scatter(
            candidate_x,
            candidate_y,
            s=12,
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            zorder=2,
        )
        ax.text(
            target_label_x,
            row_y,
            r"$i$",
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )
        ax.text(
            peer_label_x,
            row_y,
            rf"$j_{peer_index}$",
            ha="left",
            va="center",
            fontsize=7.2,
            color=color,
            fontweight="bold",
        )


def _draw_interaction_row(ax: Axes, illustration: W2OperatorConstructionData) -> None:
    r"""Draw the operator row as bars whose height *is* the weight.

    The previous cells were a fixed 0.86 by 0.55 whatever the value they carried, so
    a peer weighted 0.00 took the same visual mass as the one weighted 0.61 and the
    printed number was the only encoding present. That is a table drawn as a bar
    chart, and on the paper's headline object ``W^\flat_{i\cdot}`` it reads as a claim
    about relative peer influence that the picture contradicts. Heights against a
    zero baseline, with a scale to read them off, make the row mean what it shows.
    """
    positions = np.arange(illustration.row.size, dtype=np.float64)
    cell_labels = (r"$j_1$", r"$i$", r"$j_2$", r"$j_3$")
    ax.plot(
        [_ROW_AXIS_X, _ROW_AXIS_X],
        [0.0, 1.0],
        color=_C["gray"],
        linewidth=0.7,
        zorder=1,
    )
    for tick in (0.0, 0.5, 1.0):
        ax.plot(
            [_ROW_AXIS_X, _ROW_AXIS_X + 0.09],
            [tick, tick],
            color=_C["gray"],
            linewidth=0.7,
            zorder=1,
        )
        ax.text(
            _ROW_AXIS_X - 0.07,
            tick,
            f"{tick:.1f}",
            ha="right",
            va="center",
            fontsize=6.0,
            color=_C["gray"],
        )
    ax.plot(
        [_ROW_AXIS_X, positions[-1] + 0.55],
        [0.0, 0.0],
        color=_C["black"],
        linewidth=0.7,
        zorder=1,
    )
    for position, value, label in zip(
        positions, illustration.row, cell_labels, strict=True
    ):
        is_self = int(position) == illustration.self_index
        edge = _C["orange"] if is_self else _C["blue"]
        if is_self:
            # Zero by the leave-one-out policy, not by the fit: drawn as a stub on the
            # baseline so it cannot be read as a bar that happens to be short.
            ax.plot(
                [position - _ROW_BAR_WIDTH / 2.0, position + _ROW_BAR_WIDTH / 2.0],
                [0.0, 0.0],
                color=edge,
                linewidth=1.8,
                zorder=3,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (position - _ROW_BAR_WIDTH / 2.0, 0.0),
                    _ROW_BAR_WIDTH,
                    float(value),
                    facecolor=_TINT["blue"],
                    edgecolor=edge,
                    linewidth=0.8,
                    zorder=2,
                )
            )
        ax.text(
            position,
            float(value) + 0.06,
            r"$0$" if is_self else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=edge if is_self else _C["black"],
        )
        ax.text(
            position,
            -0.10,
            label,
            ha="center",
            va="top",
            fontsize=7,
            color=edge if is_self else _C["black"],
        )
    ax.set_xlim(-1.45, 3.60)
    ax.set_ylim(-0.42, 1.24)
    ax.axis("off")


def render_w2_operator_construction(
    output_path: Path,
    data: W2OperatorConstructionData | None = None,
    *,
    width_fraction: float = 1.0,
) -> None:
    r"""Render the assignment-to-simplex construction as a reader-first sequence.

    Every axes keeps the rectangle the grid gave it. ``set_aspect`` defaults to
    ``adjustable="box"``, which reaches equal scaling by shrinking the axes instead:
    that pulled panels A and C away from the row's title baseline -- so the headers
    read B, D, A, C down the page rather than A, B, C, D across it -- and dropped
    each panel's own ``transAxes`` annotation onto its data. Expanding the data
    limits instead leaves the boxes, and therefore the headers, aligned, and turns
    the leftover height into deliberate headroom for panel C's key.
    """
    illustration = data or simulate_w2_operator_construction()
    candidate_colors = (_C["blue"], _C["orange"], _C["green"])
    with figure(
        output_path,
        figsize(width_fraction, 0.34),
        Grid(1, 4),
        frac=width_fraction,
    ) as (fig, axes):
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.16, top=0.80, wspace=0.30)
        # Two lines each: a panel here is about 84pt wide, and the previous
        # single-line headers ran to ~97pt, so all four overlapped their neighbours.
        panel_titles = (
            "A. Target\ndistribution",
            "B. Target--peer\nassignments",
            "C. Weighted\nreconstruction",
            "D. Interaction\nrow $W^\\flat_{i\\cdot}$",
        )
        for ax, title in zip(axes, panel_titles, strict=True):
            ax.set_title(title, fontsize=7.6, fontweight="bold", pad=5)

        axes[0].scatter(
            illustration.target[:, 0],
            illustration.target[:, 1],
            s=18,
            color=_C["blue"],
            edgecolor=_C["black"],
            linewidth=0.4,
            zorder=3,
        )

        _draw_separate_assignments(axes[1], illustration, candidate_colors)

        # Target on top of an *unfilled* diamond. The fit puts the reconstruction
        # within 0.03 of each target point, so at this scale the two marks coincide;
        # the previous diamond carried an opaque white face and was drawn second, so
        # it painted over every target point and left the panel showing one series
        # while its key advertised two.
        axes[2].scatter(
            illustration.target[:, 0],
            illustration.target[:, 1],
            s=18,
            color=_C["blue"],
            edgecolor=_C["black"],
            linewidth=0.4,
            label="target",
            zorder=4,
        )
        axes[2].scatter(
            illustration.reconstruction[:, 0],
            illustration.reconstruction[:, 1],
            s=62,
            marker="D",
            facecolor="none",
            edgecolor=_C["orange"],
            linewidth=0.9,
            label="reconstruction",
            zorder=3,
        )
        axes[2].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            fontsize=6.4,
            handletextpad=0.25,
            labelspacing=0.25,
            borderpad=0.0,
        )

        _draw_interaction_row(axes[3], illustration)

        for ax, x_limits, y_centre in (
            (axes[0], (-1.05, 1.05), 0.10),
            (axes[1], (-1.26, 1.20), 1.00),
            (axes[2], (-1.05, 1.05), 0.10),
        ):
            _isometric_limits(fig, ax, x_limits, y_centre)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # One note per panel, on one baseline *below* the axes rectangle, where no
        # amount of data can reach it.
        weights = ", ".join(f"{weight:.2f}" for weight in illustration.weights)
        panel_notes = (
            r"$M$ points of $\widehat{C}_i$",
            "one matching problem per peer",
            rf"$w_i=({weights})$",
            "$W^\\flat_{ii}=0$\n$\\sum_j W^\\flat_{ij}=1$",
        )
        for ax, note in zip(axes, panel_notes, strict=True):
            ax.text(
                0.5,
                -0.05,
                note,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.9,
                color=_C["gray"],
            )

        # Placed off the axes the grid actually produced, not off hard-coded figure
        # fractions that drifted into the panels as soon as the layout moved.
        for left, right in pairwise(axes):
            left_box, right_box = left.get_position(), right.get_position()
            fig.text(
                0.5 * (left_box.x1 + right_box.x0),
                0.5 * (left_box.y0 + left_box.y1),
                r"$\longrightarrow$",
                fontsize=12,
                ha="center",
                va="center",
            )


def render_paper5_illustrations(
    output_dir: Path = Path("src/latex/projects/05_spatial_pricing/src/images"),
) -> None:
    """Render the conceptual illustrations plus the support asset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    render_rank_one_collapse(output_dir / "rank_one_collapse.pgf")
    render_spatial_exposure_arc(output_dir / "spatial_exposure_arc.pgf")
    operator_data = simulate_w2_operator_construction()
    render_w2_operator_construction(
        output_dir / "w2_operator_construction.pgf", operator_data
    )


__all__ = [
    "RankOneCollapseData",
    "W2OperatorConstructionData",
    "make_symmetric_row_stochastic_w",
    "render_paper5_illustrations",
    "render_rank_one_collapse",
    "render_spatial_exposure_arc",
    "render_w2_operator_construction",
    "rescaled_covariance",
    "simulate_rank_one_collapse",
    "simulate_w2_operator_construction",
]
