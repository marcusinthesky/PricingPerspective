#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "~=3.13.0"
# dependencies = [
#     "marimo[recommended]",
#     "numpy>=2.4",
#     "scipy>=1.17",
#     "matplotlib>=3.10",
# ]
# ///
"""Wasserstein Playground: laws, covariance ceilings, and the s-APT operator.

A marimo notebook, exported to WebAssembly, that makes two results from this
dissertation tangible on distributions the reader shapes by hand. Everything runs
in the browser under Pyodide, so the Space is `sdk: static` and costs nothing.

Paper 1 (continuous bounds). Quadratic Wasserstein geometry pins a *sharp* ceiling
on the covariance admissible between two characteristic laws. In one dimension
that ceiling is attained by the comonotone coupling, so sorting two equally sized
empirical clouds reads it off directly: the correlation of the sorted clouds *is*
the ceiling, and it falls as the W2 separation grows.

Paper 5 (s-APT, spatial pricing). Target-anchored Wasserstein barycentric
reconstruction fixes an optimal assignment from the target cloud to each peer
cloud, then chooses simplex coordinates minimising reconstruction loss jointly
across peers. In one dimension the optimal assignment is the sort, so the whole
operator is a sort followed by a simplex least squares, and the fitted row is the
paper's interaction row W-flat[i] -- nonnegative, unit sum, zero on the diagonal.

Every quantity is empirical-cloud arithmetic on sorted samples, which is why the
dependency set is numpy + scipy + matplotlib -- all three ship in the Pyodide
distribution marimo pins, so the export needs no wheel of its own.
"""

from __future__ import annotations

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Wasserstein Playground",
    # Keeps a Jupyter rendering of this notebook in __marimo__/, per the
    # convention in ../README.md.
    auto_download=["ipynb"],
    # Both are resolved relative to this file and inlined into the WASM export
    # by marimo's `wasm_notebook_template`; neither is copied into the output.
    css_file="custom.css",
    html_head_file="head.html",
)


with app.setup:
    from typing import NamedTuple

    import marimo as mo
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from scipy.optimize import nnls
    from scipy.stats import gaussian_kde, skewnorm


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Wasserstein Playground

    Shape four firms' characteristic laws in the sidebar. Their W₂ separation caps
    how much their returns can co-move (paper 1), and the target-anchored
    barycentric operator (paper 5) reconstructs the target from its peers,
    returning the interaction row `W♭`.
    """)


@app.cell
def _():
    class Tier(NamedTuple):
        """One rung of the visual hierarchy: how heavily a line is drawn."""

        lw: float
        ls: str
        alpha: float

    return (Tier,)


@app.cell
def _(Tier):
    FIRMS = ("A", "B", "C", "D")
    COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")  # Okabe-Ito
    PAPER = "#f6f6f1"
    PAPER_BRIGHT = "#fcfcf9"
    INK = "#090909"
    SLIDERS = (  # label, minimum, maximum, step, one default per firm
        ("location μ", -3.0, 3.0, 0.05, (0.0, -0.8, 0.2, 0.6)),
        ("scale σ", 0.25, 2.0, 0.05, (0.9, 0.9, 0.6, 0.7)),
        ("skew α", -8.0, 8.0, 0.1, (0.0, 5.0, 0.0, -4.0)),
        ("bimodal split Δ", 0.0, 3.0, 0.05, (0.0, 0.0, 1.5, 1.2)),
    )
    # One visual hierarchy, applied identically to the densities and the return paths:
    # the anchor reads first, what the operator built from its peers reads second, and
    # the peers themselves recede into light dotted context.
    TARGET_TIER = Tier(2.6, "-", 1.0)  # the anchor reads first
    MIXTURE_TIER = Tier(1.5, "--", 0.95)  # what the operator built out of the peers
    PEER_TIER = Tier(1.1, ":", 0.55)  # context, kept light and slightly transparent
    RUG_TIER = Tier(0.75, ":", 0.5)  # sample ticks stay subordinate to every curve
    # Weight on the sum-to-one row of the simplex least squares. `nnls` minimises the
    # *sum* of squared residuals over M articles, so an absolute penalty has to clear
    # that M-dependent term by a wide margin: at 1e3 the fitted row drifts up to 2.3e-3
    # from the true constrained optimum at M=512, which is visible in the 2-dp readout;
    # at 1e6 the drift is 1.8e-7.
    PENALTY = 1.0e6
    VOL = 0.02  # per-period return volatility; sets the vertical scale only
    RUG_HEIGHT = 0.05  # rug-row height, as a share of the tallest density
    VISIBLE_SKEW = 0.5  # |α| above which the shape reads as skewed, not symmetric
    ACTIVE = 1.0e-8  # the paper's tolerance for counting a peer as active in a W♭ row
    return (
        ACTIVE,
        COLOURS,
        FIRMS,
        INK,
        MIXTURE_TIER,
        PAPER,
        PAPER_BRIGHT,
        PEER_TIER,
        PENALTY,
        RUG_HEIGHT,
        RUG_TIER,
        SLIDERS,
        TARGET_TIER,
        VISIBLE_SKEW,
        VOL,
    )


@app.cell
def _(VISIBLE_SKEW):
    def pdf(
        x: np.ndarray, loc: float, scale: float, skew: float, split: float
    ) -> np.ndarray:
        """Closed-form density of the two-component skew-normal mixture.

        The four knobs span every shape the app advertises: Δ=0 with α=0 is symmetric
        unimodal, Δ=0 with α≠0 is skewed, Δ>σ is bimodal, and the two combine.
        """
        lower = skewnorm.pdf(x, skew, loc - split, scale)
        upper = skewnorm.pdf(x, skew, loc + split, scale)
        return 0.5 * (lower + upper)

    def shape_name(scale: float, skew: float, split: float) -> str:
        """Name the shape the current knobs produce.

        An equal-weight mixture of two normals separated by 2Δ is bimodal exactly when
        Δ>σ, so the modality test is the closed-form one rather than a threshold.
        """
        tail = "skewed" if abs(skew) > VISIBLE_SKEW else "symmetric"
        return f"{tail} {'bimodal' if split > scale else 'unimodal'}"

    def draw(
        rng: np.random.Generator,
        size: int,
        loc: float,
        scale: float,
        skew: float,
        split: float,
    ) -> np.ndarray:
        """Draw `size` articles from the same mixture, returned as order statistics."""
        side = rng.choice((-1.0, 1.0), size=size)
        draws = skewnorm.rvs(
            skew, loc + side * split, scale, size=size, random_state=rng
        )
        return np.sort(draws)

    return draw, pdf, shape_name


@app.cell
def _(PENALTY, VOL):
    def barycentric_row(
        clouds: np.ndarray, target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct one row of the barycentric interaction field W♭ (paper 5).

        The clouds arrive sorted, so column m already holds the transport-aligned peer
        positions for the target's m-th order statistic. Conditional on that alignment,
        pick simplex coordinates jointly across peers -- which is why a row is not a
        scalar transform applied to each pairwise distance separately. The sum-to-one
        constraint enters `nnls` as a heavily weighted extra row, and the diagonal
        stays zero because the target is never its own candidate.
        """
        peers = [j for j in range(len(clouds)) if j != target]
        aligned = np.column_stack([clouds[j] for j in peers])
        lhs = np.vstack([aligned, np.full(len(peers), PENALTY)])
        weights, _ = nnls(lhs, np.concatenate([clouds[target], [PENALTY]]))
        total = weights.sum()
        weights = weights / total if total else np.full(len(peers), 1.0 / len(peers))
        row = np.zeros(len(clouds))
        row[peers] = weights
        return row, aligned @ weights

    def geometry(clouds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pairwise W2 distances and the sharp correlation ceiling of paper 1.

        Sorting couples every firm to one common uniform, so the pairwise covariance
        endpoints are attained *simultaneously*: the ceiling is a genuine positive
        semi-definite correlation matrix, not a pasting-together of unrelated pairwise
        bounds, and it can be factored directly to draw a return panel.
        """
        distance = np.sqrt(((clouds[:, None, :] - clouds[None, :, :]) ** 2).mean(-1))
        return distance, np.corrcoef(clouds)

    def market(
        rng: np.random.Generator,
        ceiling: np.ndarray,
        periods: int,
        alignment: float,
    ) -> np.ndarray:
        """Draw a return panel at `alignment` of the covariance ceiling.

        Shrinking the ceiling toward the identity is paper 1's nonnegative
        transport-excess term: the gap between a realised arrangement and the
        maximum-covariance arrangement of the same characteristic laws.
        """
        eye = np.eye(len(ceiling))
        chol = np.linalg.cholesky(
            alignment * ceiling + (1 - alignment) * eye + 1e-9 * eye
        )
        return VOL * (rng.standard_normal((periods, len(ceiling))) @ chol.T)

    return barycentric_row, geometry, market


@app.cell
def _(RUG_HEIGHT, RUG_TIER, Tier):
    def stroke(
        axis: Axes,
        x: np.ndarray,
        y: np.ndarray,
        colour: str,
        tier: Tier,
        label: str,
    ) -> None:
        """Draw one line at the given rung of the hierarchy."""
        axis.plot(
            x, y, color=colour, lw=tier.lw, ls=tier.ls, alpha=tier.alpha, label=label
        )

    def rug_band(unit: float, row: int) -> tuple[float, float]:
        """Vertical extent of rug row `row`, stacked below the densities."""
        return -unit * RUG_HEIGHT * (row + 1.35), -unit * RUG_HEIGHT * (row + 0.35)

    def rug(
        axis: Axes, samples: np.ndarray, unit: float, row: int, colour: str
    ) -> None:
        """Dotted ticks for one cloud, in its own row of the margin below."""
        axis.vlines(
            samples,
            *rug_band(unit, row),
            color=colour,
            lw=RUG_TIER.lw,
            ls=RUG_TIER.ls,
            alpha=RUG_TIER.alpha,
        )

    return rug, rug_band, stroke


@app.cell
def _(
    COLOURS,
    FIRMS,
    MIXTURE_TIER,
    PAPER,
    PEER_TIER,
    TARGET_TIER,
    pdf,
    rug,
    rug_band,
    shape_name,
    stroke,
):
    def laws_figure(
        params: np.ndarray,
        clouds: np.ndarray,
        recon: np.ndarray,
        target: int,
        *,
        kde: bool,
    ) -> Figure:
        """Left panel: closed-form densities, dotted sample ticks, reconstructed law."""
        grid = np.linspace(clouds.min() - 1.0, clouds.max() + 1.0, 512)
        densities = [pdf(grid, *p) for p in params]
        reconstructed = gaussian_kde(recon)(grid)
        unit = max(d.max() for d in [*densities, reconstructed])
        fig = Figure(figsize=(7.0, 6.0), layout="constrained", facecolor=PAPER)
        axis = fig.subplots()
        axis.set_facecolor(PAPER)
        for i, (name, colour) in enumerate(zip(FIRMS, COLOURS, strict=True)):
            anchor = i == target
            tint = "black" if anchor else colour
            suffix = " ← target" if anchor else ""
            label = f"{name} · {shape_name(*params[i, 1:])}{suffix}"
            stroke(
                axis,
                grid,
                densities[i],
                tint,
                TARGET_TIER if anchor else PEER_TIER,
                label,
            )
            if kde:
                # Dash-dot keeps the empirical overlay clear of the solid/dashed/dotted
                # roles above; the anchor's KDE goes grey so it never reads as W♭.
                axis.plot(
                    grid,
                    gaussian_kde(clouds[i])(grid),
                    color="#555555" if anchor else colour,
                    lw=1.0,
                    ls="-.",
                    alpha=0.7 if anchor else 0.45,
                    label="sample KDE" if anchor else "_kde",
                )
            rug(axis, clouds[i], unit, i, tint)
        stroke(axis, grid, reconstructed, "black", MIXTURE_TIER, "W♭ reconstruction")
        rug(axis, recon, unit, len(FIRMS), "black")
        axis.set(xlabel="characteristic coordinate", ylabel="density")
        axis.set_ylim(rug_band(unit, len(FIRMS))[0], None)
        axis.set_title("Characteristic laws, sampled articles, and W♭", fontsize=11)
        axis.legend(fontsize=9, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        return fig

    return (laws_figure,)


@app.cell
def _(COLOURS, FIRMS, INK, PAPER, PAPER_BRIGHT):
    def pairwise_figure(distance: np.ndarray, ceiling: np.ndarray) -> Figure:
        """Middle panel: pairwise W2 separation and its correlation ceiling."""
        fig = Figure(figsize=(7.0, 6.0), layout="constrained", facecolor=PAPER)
        axis = fig.subplots()
        axis.set_facecolor(PAPER)
        diagonal = np.eye(len(FIRMS), dtype=bool)
        off = ceiling[~diagonal]
        image = axis.imshow(
            np.where(diagonal, np.nan, ceiling),
            cmap=colormaps["Greys"].with_extremes(bad=PAPER_BRIGHT),
            vmin=off.min(),
            vmax=off.max(),
        )
        fig.colorbar(image, ax=axis, fraction=0.045, label="ρ̄ ceiling")
        axis.set(xticks=range(len(FIRMS)), yticks=range(len(FIRMS)))
        axis.set(xticklabels=FIRMS, yticklabels=FIRMS)
        for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
            for label, colour in zip(labels, COLOURS, strict=True):
                label.set_color(colour)
                label.set_fontweight("bold")
        for i, j in np.ndindex(ceiling.shape):
            if i == j:
                continue
            tone = "white" if ceiling[i, j] > off.mean() else INK
            cell = f"ρ̄ {ceiling[i, j]:.2f}\nW₂ {distance[i, j]:.2f}"
            axis.text(j, i, cell, ha="center", va="center", fontsize=8.0, color=tone)
        axis.set_title("Pairwise W₂ geometry", fontsize=11)
        axis.tick_params(length=0)
        axis.spines[:].set_visible(False)
        return fig

    return (pairwise_figure,)


@app.cell
def _(COLOURS, FIRMS, INK, MIXTURE_TIER, PAPER, PEER_TIER, TARGET_TIER, stroke):
    def returns_figure(
        returns: np.ndarray,
        row: np.ndarray,
        target: int,
    ) -> Figure:
        """Right panel: correlated return paths and the barycentric peer average."""
        fig = Figure(figsize=(7.0, 6.0), layout="constrained", facecolor=PAPER)
        bottom = fig.subplots()
        bottom.set_facecolor(PAPER)
        peer_average = returns @ row
        for i, (name, colour) in enumerate(zip(FIRMS, COLOURS, strict=True)):
            anchor = i == target
            path = returns[:, i].cumsum()
            periods = np.arange(len(path))
            tier = TARGET_TIER if anchor else PEER_TIER
            stroke(bottom, periods, path, INK if anchor else colour, tier, name)
        total = peer_average.cumsum()
        stroke(
            bottom, np.arange(len(total)), total, INK, MIXTURE_TIER, "W♭ peer average"
        )
        realised = np.corrcoef(returns[:, target], peer_average)[0, 1]
        bottom.set(xlabel="period", ylabel="cumulative return")
        pair = f"corr({FIRMS[target]}, peer average) = {realised:+.2f}"
        bottom.set_title(f"Correlated returns · {pair}", fontsize=11)
        bottom.legend(fontsize=9, ncols=5, framealpha=0.85, edgecolor="none")
        bottom.spines[["top", "right"]].set_visible(False)
        return fig

    return (returns_figure,)


@app.cell
def _(FIRMS, SLIDERS):
    target_ui = mo.ui.dropdown(FIRMS, value=FIRMS[0], label="Target firm (anchor)")
    size_ui = mo.ui.slider(32, 512, step=32, value=192, label="Articles per firm (M)")
    periods_ui = mo.ui.slider(60, 1000, step=10, value=250, label="Return periods (T)")
    alignment_ui = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.7, label="Share of the covariance ceiling"
    )
    seed_ui = mo.ui.slider(0, 999, step=1, value=7, label="Seed")
    kde_ui = mo.ui.checkbox(value=False, label="Overlay sample KDE")
    # One flat array so `.value` collects every knob in firm-major order, which is
    # exactly the (len(FIRMS), len(SLIDERS)) reshape the compute cell wants. The
    # individual elements are still laid out per firm in the accordion below.
    shape_ui = mo.ui.array(
        [
            mo.ui.slider(lo, hi, step=step, value=default[i], label=label)
            for i in range(len(FIRMS))
            for label, lo, hi, step, default in SLIDERS
        ]
    )
    return (
        alignment_ui,
        kde_ui,
        periods_ui,
        seed_ui,
        shape_ui,
        size_ui,
        target_ui,
    )


@app.cell(hide_code=True)
def _(
    FIRMS,
    SLIDERS,
    alignment_ui,
    kde_ui,
    periods_ui,
    seed_ui,
    shape_ui,
    size_ui,
    target_ui,
):
    mo.sidebar(
        [
            mo.md("### Controls"),
            target_ui,
            size_ui,
            periods_ui,
            alignment_ui,
            seed_ui,
            kde_ui,
            mo.accordion(
                {
                    f"Firm {name}": mo.vstack(
                        [shape_ui[i * len(SLIDERS) + k] for k in range(len(SLIDERS))]
                    )
                    for i, name in enumerate(FIRMS)
                }
            ),
        ]
    )


@app.cell
def _(
    FIRMS,
    alignment_ui,
    barycentric_row,
    draw,
    geometry,
    market,
    periods_ui,
    seed_ui,
    shape_ui,
    size_ui,
    target_ui,
):
    rng = np.random.default_rng(int(seed_ui.value))
    params = np.reshape(shape_ui.value, (len(FIRMS), -1))
    clouds = np.stack([draw(rng, int(size_ui.value), *p) for p in params])
    target = FIRMS.index(target_ui.value)
    row, recon = barycentric_row(clouds, target)
    distance, ceiling = geometry(clouds)
    returns = market(rng, ceiling, int(periods_ui.value), alignment_ui.value)
    return ceiling, clouds, distance, params, recon, returns, row, target


@app.cell(hide_code=True)
def _(
    ceiling,
    clouds,
    distance,
    kde_ui,
    laws_figure,
    pairwise_figure,
    params,
    recon,
    returns,
    returns_figure,
    row,
    target,
):
    mo.hstack(
        [
            laws_figure(params, clouds, recon, target, kde=kde_ui.value),
            pairwise_figure(distance, ceiling),
            returns_figure(returns, row, target),
        ],
        widths="equal",
        gap=1,
    )


@app.cell(hide_code=True)
def _(ACTIVE, FIRMS, alignment_ui, clouds, recon, row, size_ui, target):
    residual = np.sqrt(((clouds[target] - recon) ** 2).mean())
    coords = ", ".join(f"{w:.2f}" for w in row)
    active = int((row > ACTIVE).sum())
    mo.md(
        f"**W♭[{FIRMS[target]}·] = ({coords})** over ({', '.join(FIRMS)}) — {active} "
        f"active peers, fitted on {int(size_ui.value)} aligned articles, "
        f"reconstruction "
        f"RMSE **{residual:.3f}** in characteristic units.\n\n"
        "The weights are barycentric coordinates on *aligned* peer positions — not "
        "probabilities of drawing a peer's article, not a tradable replicating "
        "portfolio, and not causal influence. The middle pairwise ceiling is the "
        "maximum-covariance arrangement of these laws; returns are drawn at "
        f"{alignment_ui.value:.0%} of it, leaving the rest as paper 1's nonnegative "
        "transport excess."
    )


if __name__ == "__main__":
    app.run()
