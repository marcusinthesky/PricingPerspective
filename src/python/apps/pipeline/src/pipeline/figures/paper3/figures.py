"""Paper 3 publication figures and generated-number bindings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.figures._common import (
    _C,
    _TINT,
    _load_yaml,
    _save,
    _style,
    categorical_axis,
    figsize,
    fit_to_width,
)
from pipeline.figures.paper3.numbers import Paper3NumberInputs, _write_paper3_numbers

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Paper3FigurePaths:
    """Artifacts consumed by the combined Paper 3 publication renderer."""

    paper3_dir: Path
    pricefree_dir: Path
    output_dir: Path
    numbers_path: Path
    params_path: Path
    mc_validation_dir: Path
    hedging_validation_dir: Path
    cross_model_summary: Path
    size_power_path: Path
    certificate_dir: Path


def _fig_conceptual_pipeline(out: Path) -> None:
    """Render the governed observed-to-decision object flow."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.95, 0.24))
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(-0.55, 0.55)
    ax.axis("off")
    labels = (
        r"$C_i$",
        r"$W_2(C_i,C_j)$",
        r"$\ell_{ij}$",
        r"$\mathcal{C}(q)$",
        "variance bound",
        r"$x^\star$",
    )
    colours = (
        _C["blue"],
        _C["blue"],
        _C["orange"],
        _C["orange"],
        _C["green"],
        _C["purple"],
    )
    positions = (0.35, 1.4, 2.45, 3.45, 4.55, 5.7)
    half_widths = (0.18, 0.45, 0.22, 0.30, 0.62, 0.20)
    for idx, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.text(
            positions[idx],
            0.0,
            label,
            ha="center",
            va="center",
            color=colour,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": colour,
            },
        )
        if idx:
            maintained = label == r"$\ell_{ij}$"
            ax.annotate(
                "",
                xy=(positions[idx] - half_widths[idx] - 0.05, 0.0),
                xytext=(positions[idx - 1] + half_widths[idx - 1] + 0.05, 0.0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": _C["gray"],
                    "linestyle": "--" if maintained else "-",
                },
            )
    fig.tight_layout()
    _save(fig, out)


def _fig_certificate_surface(out: Path, surface: pd.DataFrame) -> None:
    """Plot the descriptive certificate sensitivity for the two fixed portfolios."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(0.95, 0.42), sharey=True)
    portfolio_labels = (
        ("equal_weight", "Equal weight"),
        ("inverse_volatility", "Inverse volatility"),
    )
    colours = (_C["blue"], _C["orange"], _C["green"], _C["purple"])
    for ax, (portfolio, title) in zip(axes, portfolio_labels, strict=True):
        subset = surface[surface["portfolio"] == portfolio]
        for colour, scale in zip(colours, sorted(subset["L"].unique()), strict=True):
            line = subset[subset["L"] == scale].sort_values("tau")
            ax.plot(
                line["tau"],
                100.0 * line["certificate_credit"],
                marker="o",
                color=colour,
                label=rf"$L={scale:g}$",
            )
        ax.set_title(title)
        ax.set_xlabel(r"Common slack $\tau$")
        ax.grid(visible=True, linestyle=":", alpha=0.45)
    axes[0].set_ylabel("Certificate credit (%)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def _fig_calibration_coverage(
    out: Path, summary: dict[str, object], coverage_target: float
) -> None:
    """Plot calibration coverage against the declared grid and its target."""
    cells = summary.get("calibration_cells")
    if not isinstance(cells, list) or not cells:
        message = "calibration-coverage figure input has no calibration cells"
        raise TypeError(message)
    frame = pd.DataFrame(cells)
    for column in ("L", "tau", "coverage_lower"):
        if column not in frame.columns:
            message = f"calibration cells are missing the {column!r} column"
            raise TypeError(message)
    _style()
    # The panel must be tall enough for the rotated y-label; at 0.46 the label
    # exceeds the axes height and the trailing unit is clipped.
    fig, ax = plt.subplots(figsize=figsize(0.72, 0.54))
    colours = (_C["blue"], _C["orange"], _C["green"], _C["purple"])
    slacks = sorted(frame["tau"].unique())
    for colour, slack in zip(colours, slacks, strict=True):
        line = frame[frame["tau"] == slack].sort_values("L")
        ax.plot(
            line["L"],
            100.0 * line["coverage_lower"],
            marker="o",
            color=colour,
            label=rf"$\tau={slack:g}$",
        )
    ax.axhline(
        100.0 * coverage_target,
        linestyle="--",
        color=_C["gray"],
        label="Calibration rule",
    )
    ax.set_xlabel(r"Carrier constant $L$ (declared grid)")
    # The long form of this label does not fit the panel height and is clipped;
    # the caption carries "calibration coverage, 95% lower endpoint" in full.
    ax.set_ylabel("Coverage lower endpoint (\\%)")
    ax.grid(visible=True, linestyle=":", alpha=0.45)
    # Every series rises left to right and the rule line occupies the top, so
    # the free region is bottom-right. Drop the floor to keep the legend clear
    # of the lowest series instead of letting it overlap.
    lower = 100.0 * float(frame["coverage_lower"].min())
    upper = max(100.0 * coverage_target, 100.0 * float(frame["coverage_lower"].max()))
    ax.set_ylim(lower - 0.45 * (upper - lower), upper + 0.05 * (upper - lower))
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, out)


def _fig_news_variance_benchmark(out: Path, summary: dict[str, object]) -> None:
    """Plot standardized in-sample variance relative to the sample GMV."""
    benchmark = summary.get("news_only_benchmark")
    if not isinstance(benchmark, dict):
        message = "news-variance figure input has no news-only benchmark"
        raise TypeError(message)
    if benchmark.get("status") != "full_sample_descriptive":
        message = "news-variance figure input has an unsupported status"
        raise ValueError(message)
    portfolios = benchmark.get("portfolios")
    if not isinstance(portfolios, dict):
        message = "news-variance figure input has no portfolio diagnostics"
        raise TypeError(message)
    news = portfolios.get("news_only")
    equal = portfolios.get("equal_weight")
    sample_gmv = portfolios.get("sample_gmv")
    if not all(isinstance(item, dict) for item in (news, equal, sample_gmv)):
        message = "news-variance figure input has incomplete portfolio diagnostics"
        raise TypeError(message)
    news = cast("dict[str, object]", news)
    equal = cast("dict[str, object]", equal)
    sample_gmv = cast("dict[str, object]", sample_gmv)

    def summary_float(portfolio: dict[str, object], key: str) -> float:
        value = portfolio.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            message = f"news-variance figure field {key!r} must be numeric"
            raise TypeError(message)
        return float(value)

    labels = ("News-only", "Equal risk weights", "Sample GMV")
    values = np.asarray(
        [
            100.0 * summary_float(news, "gmv_relative_variance"),
            100.0 * summary_float(equal, "gmv_relative_variance"),
            100.0 * summary_float(sample_gmv, "gmv_relative_variance"),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        message = "news-variance figure values must be finite and positive"
        raise ValueError(message)
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.72, 3.6 / 5.4))
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        color=(_C["blue"], _C["orange"], _C["green"]),
        edgecolor=_C["black"],
        linewidth=0.35,
    )
    ax.axhline(100.0, color=_C["gray"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    categorical_axis(ax)
    ax.set_ylabel("Standardized variance\n(relative to sample GMV)")
    ax.set_ylim(0.0, max(110.0, float(values.max()) * 1.18))
    ax.grid(visible=True, axis="y", linestyle=":", alpha=0.45)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fit_to_width(fig, 0.72)
    _save(fig, out)


def _fig_certificate_validation(out: Path, random_portfolios: pd.DataFrame) -> None:
    """Compare certified with subsequently realized diversification."""
    _style()
    frame = random_portfolios.copy()
    amplitude_squared = frame["operational_bound"] / (1.0 - frame["certificate_credit"])
    frame["realized_diversification"] = (
        1.0 - frame["realized_variance"] / amplitude_squared
    )
    # The governed draw contains the same 500 fixed-seed portfolio identities at
    # every formation.  Plot each identity once, averaging its chronological
    # outcomes, so the PGF remains publication-sized without changing the
    # predeclared random-portfolio population.
    frame = (
        frame.groupby("portfolio_id", as_index=False)[
            ["certificate_credit", "realized_diversification"]
        ]
        .mean()
        .sort_values("portfolio_id")
    )
    fig, ax = plt.subplots(figsize=figsize(0.72, 0.72))
    ax.scatter(
        100.0 * frame["certificate_credit"],
        100.0 * frame["realized_diversification"],
        s=9,
        alpha=0.35,
        color=_C["blue"],
        linewidths=0.0,
    )
    limits = np.asarray(
        [
            min(-2.5, 100.0 * frame["realized_diversification"].min()),
            max(
                100.0 * frame["certificate_credit"].max(),
                100.0 * frame["realized_diversification"].max(),
            ),
        ]
    )
    ax.plot(limits, limits, color=_C["orange"], linestyle="--", label="Bound line")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Certified diversification (%)")
    ax.set_ylabel("Subsequently realized diversification (%)")
    ax.grid(visible=True, linestyle=":", alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out)


def _fig_cov_anatomy(out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.9, 3.6 / 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # 2x2 schematic of Sigma over assets {i, j}. Diagonal runs top-left ->
    # bottom-right; cells are 1.2 x 1.2. Top row = asset i, bottom row = asset j.
    x_left, x_right = 2.5, 6.3
    y_top, y_bot = 4.0, 1.0
    cell = 1.2
    ax.add_patch(
        plt.Rectangle((2.5, 1.0), 5, 4.2, fill=False, lw=1.1, edgecolor=_C["black"])
    )
    # Diagonal variance cells: (i,i) top-left, (j,j) bottom-right.
    ax.add_patch(
        plt.Rectangle(
            (x_left, y_top),
            cell,
            cell,
            facecolor=_C["sky"],
            alpha=0.5,
            edgecolor=_C["blue"],
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x_right, y_bot),
            cell,
            cell,
            facecolor=_C["sky"],
            alpha=0.5,
            edgecolor=_C["blue"],
        )
    )
    ax.text(
        x_left + cell / 2,
        y_top + cell / 2,
        r"$\sigma_i^2$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        x_right + cell / 2,
        y_bot + cell / 2,
        r"$\sigma_j^2$",
        ha="center",
        va="center",
        fontsize=10,
    )
    # Off-diagonal cells sit at (i,j) top-right and its mirror (j,i) bottom-left,
    # aligned with their own row and column -- the symmetric off-diagonal pair.
    for ox, oy in ((x_right, y_top), (x_left, y_bot)):
        ax.add_patch(
            plt.Rectangle(
                (ox, oy),
                cell,
                cell,
                facecolor=_C["orange"],
                alpha=0.35,
                edgecolor=_C["orange"],
            )
        )
    ax.text(
        x_right + cell / 2,
        y_top + cell / 2,
        r"$\Sigma_{ij}$",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(
        x_left + cell / 2,
        y_bot + cell / 2,
        r"$\Sigma_{ji}$",
        ha="center",
        va="center",
        fontsize=9,
    )
    # Formula callout in the empty centre, with a leader line to the (i,j) cell.
    ax.annotate(
        r"$\Sigma^{\max}_{ij}=\frac{1}{2}\left(\sigma_i^2+\sigma_j^2"
        r"-\kappa^2\widehat W_{2,ij}^{\,2}\right)$",
        xy=(x_right, y_top + cell / 2),
        xytext=(4.05, 3.05),
        fontsize=8,
        va="center",
        arrowprops={"arrowstyle": "->", "color": _C["gray"], "lw": 0.75},
    )
    ax.text(
        5,
        5.6,
        r"$\widehat\Sigma_{\mathrm{W2}}$: return variances on the diagonal;"
        "\n"
        r"each off-diagonal blends "
        r"both variances with the text-law distance $\widehat W_{2,ij}$",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.text(
        5,
        0.35,
        r"single scale $\kappa$ calibrated once on returns, reused across every pair",
        ha="center",
        va="center",
        fontsize=7,
        color=_C["gray"],
    )
    ax.set_title("W2 covariance-envelope anatomy")
    fit_to_width(fig, 0.9)
    _save(fig, out)


def _fig_dual_status(out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, title, body, color in (
        (
            axes[0],
            "Lean-verified geometry",
            (
                r"W2 Fr\'echet ceiling"
                "\n"
                r"$+$ nonnegative excess"
                "\n"
                r"under stated hypotheses"
            ),
            _C["green"],
        ),
        (
            axes[1],
            "Empirical restriction",
            (
                r"Scalar transmission"
                "\n"
                r"$+$ zero transport excess"
                "\n"
                r"$\Rightarrow$ GMM rejects equality"
            ),
            _C["orange"],
        ),
    ):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.08, 0.12),
                0.84,
                0.76,
                facecolor=_TINT["neutral"],
                edgecolor=color,
                lw=1.5,
            )
        )
        ax.text(
            0.5,
            0.78,
            title,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        ax.text(0.5, 0.42, body, ha="center", va="center", fontsize=8)
    fig.suptitle("Formal theorem versus identifying benchmark", fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, out)


def _load_regret_grid(
    regret_yaml: Path,
) -> tuple[list[int], list[int], dict[tuple[int, int], dict[str, int | float | str]]]:
    """Load and validate the complete regret grid used by the regime map."""
    if not regret_yaml.exists():
        message = (
            f"regime-map input missing: {regret_yaml} (expected a regret "
            "summary yaml with a 'median_regret_vs_oracle_by_n_T' mapping)"
        )
        raise FileNotFoundError(message)
    summary = _load_yaml(regret_yaml)
    regret = summary.get("median_regret_vs_oracle_by_n_T")
    if not isinstance(regret, dict) or not regret:
        message = (
            f"regime-map input malformed: {regret_yaml} has no non-empty "
            "'median_regret_vs_oracle_by_n_T' mapping"
        )
        raise ValueError(message)

    ns: list[int] = []
    ts: list[int] = []
    for cell in regret.values():
        if not isinstance(cell, dict) or "n_assets" not in cell or "T" not in cell:
            message = (
                f"regime-map input malformed: cell in {regret_yaml} missing "
                "'n_assets'/'T' keys"
            )
            raise ValueError(message)
        n_val, t_val = int(cell["n_assets"]), int(cell["T"])
        if n_val not in ns:
            ns.append(n_val)
        if t_val not in ts:
            ts.append(t_val)
    ns.sort()
    ts.sort()
    if not ns or not ts:
        message = f"regime-map input malformed: {regret_yaml} yields empty n/T axes"
        raise ValueError(message)

    by_nt = {(int(cell["n_assets"]), int(cell["T"])): cell for cell in regret.values()}
    return ns, ts, by_nt


def _fig_regime_map(out: Path, regret_yaml: Path) -> None:
    """Build the regret regime-map from validated live regret-summary inputs."""
    _style()
    ns, ts, by_nt = _load_regret_grid(regret_yaml)

    sample = np.full((len(ns), len(ts)), np.nan)
    dist = np.full((len(ns), len(ts)), np.nan)
    for i, n in enumerate(ns):
        for j, t in enumerate(ts):
            cell = by_nt.get((n, t))
            if cell is None or "smvo" not in cell or "pf_distmvo_data" not in cell:
                message = (
                    f"regime-map input malformed: missing 'smvo'/'pf_distmvo_data' "
                    f"regret for n={n}, T={t} in {regret_yaml}"
                )
                raise ValueError(message)
            sample[i, j] = float(cell["smvo"])
            dist[i, j] = float(cell["pf_distmvo_data"])
    win = dist < sample
    fig, ax = plt.subplots(figsize=figsize(0.8, 3.6 / 5.4))
    cmap = mpl.colors.ListedColormap([_TINT["orange"], _TINT["blue"]])
    ax.imshow(win.astype(float), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([str(t) for t in ts])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    categorical_axis(ax, "both")
    ax.set_xlabel(r"Estimation window $T$")
    ax.set_ylabel(r"Assets $n$")
    ax.set_title(
        r"Regions where $\hat{\Sigma}_{\mathrm{W2}}$ outperforms "
        r"the sample covariance"
    )
    for i in range(len(ns)):
        for j in range(len(ts)):
            ax.text(
                j, i, "D" if win[i, j] else "S", ha="center", va="center", fontsize=8
            )
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_TINT["blue"],
            edgecolor=_C["black"],
            lw=0.35,
            label=r"$\hat{\Sigma}_{\mathrm{W2}}$ lower regret (W2)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_TINT["orange"],
            edgecolor=_C["black"],
            lw=0.35,
            label="Sample lower regret (S)",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    fig.tight_layout()
    fit_to_width(fig, 0.8)
    _save(fig, out)


def _fig_sharpe_bars(out: Path, results: pd.DataFrame) -> None:
    _style()
    df = results
    if "constraint" in df.columns:
        df = df[df["constraint"] == "long_only"]
    if "period" in df.columns:
        df = df[df["period"] == "full"]
    if "source" in df.columns:
        df = df[df["source"] == "baseline"]
    methods = ["sample", "ledoit_wolf", "sigma_dist", "sigma_shrink"]
    labels = {
        "sample": "Sample",
        "ledoit_wolf": "LW",
        "sigma_dist": r"$\hat{\Sigma}_{\mathrm{W2}}$",
        "sigma_shrink": r"$\hat{\Sigma}(\lambda_t)$",
    }
    windows = sorted(df["window"].unique())
    x = np.arange(len(windows))
    width = 0.18
    fig, ax = plt.subplots(figsize=figsize(0.9, 3.6 / 6.2))
    for i, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("window")
        vals = [
            float(sub.loc[w, "sharpe_net_10bps"]) if w in sub.index else np.nan
            for w in windows
        ]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=labels.get(m, m),
            edgecolor=_C["black"],
            lw=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in windows])
    categorical_axis(ax)
    ax.set_xlabel(r"Estimation window $T$")
    ax.set_ylabel(r"Net Sharpe @ 10 bps")
    ax.set_title(r"Long-only OOS Sharpe by estimator")
    # Below the axes: at the printed width the fourth entry, $\hat{\Sigma}(\lambda_t)$,
    # ran past the right spine.
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.2,
    )
    ax.axhline(0, color=_C["gray"], lw=0.5)
    ax.grid(visible=True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fit_to_width(fig, 0.9)
    _save(fig, out)


def _fig_shrinkage_weight(out: Path, results: pd.DataFrame) -> None:
    """Plot the production blend's mean dynamic sample-covariance weight."""
    required = {
        "method",
        "constraint",
        "period",
        "source",
        "window",
        "lambda_sample_mean",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            "shrinkage-weight figure input missing columns: "
            + ", ".join(sorted(missing))
        )
    _style()
    df = results[
        (results["method"] == "sigma_shrink")
        & (results["constraint"] == "long_only")
        & (results["period"] == "full")
        & (results["source"] == "baseline")
    ]
    if df.empty:
        message = "shrinkage-weight figure input has no baseline rows"
        raise ValueError(message)
    summary = df.groupby("window", sort=True)["lambda_sample_mean"].mean()
    fig, ax = plt.subplots(figsize=figsize(0.9, 3.2 / 5.4))
    x = summary.index.to_numpy()
    y = summary.to_numpy()
    ax.plot(
        x,
        y,
        "o-",
        color=_C["blue"],
        lw=1.5,
        ms=5,
        label=r"Mean sample weight $\lambda_t$",
    )
    ax.fill_between(x, 0.0, y, color=_C["blue"], alpha=0.12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xlabel(r"Estimation window $T$")
    ax.set_ylabel(r"Mean sample-covariance weight")
    ax.set_title(r"Dynamic shrinkage weight across information regimes")
    ax.grid(visible=True, axis="y", linestyle=":", alpha=0.45)
    # Upper left: the series rises left-to-right, so `lower right` ran the legend
    # into the axis frame once it printed at full size.
    ax.legend(loc="upper left")
    fig.tight_layout()
    fit_to_width(fig, 0.9)
    _save(fig, out)


def _fig_hedging_geometry(out: Path) -> None:
    """Render the observed-characteristic to latent-exposure model boundary."""
    _style()
    fig, ax = plt.subplots(figsize=figsize(0.95, 3.4 / 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = (
        (0.45, 2.1, _TINT["blue"], "Observed article law", r"$X_i\sim C_i$"),
        (3.55, 2.1, _TINT["neutral"], "Transmission kernel", r"$B_i=T(X_i,U_i)$"),
        (
            6.65,
            2.9,
            _TINT["orange"],
            "Latent exposure law",
            r"$P_i=\mathcal{L}(B_i)$",
        ),
    )
    for x, width, color, title, formula in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, 1.35),
                width,
                1.35,
                facecolor=color,
                edgecolor=_C["black"],
                lw=0.7,
            )
        )
        ax.text(x + width / 2, 2.25, title, ha="center", fontsize=8)
        ax.text(x + width / 2, 1.77, formula, ha="center", fontsize=9)
    for start, end in ((2.55, 3.55), (5.65, 6.65)):
        ax.annotate(
            "",
            xy=(end, 2.02),
            xytext=(start, 2.02),
            arrowprops={"arrowstyle": "->", "color": _C["gray"], "lw": 1.1},
        )
    ax.text(
        5,
        0.65,
        r"Data identify $W_2(C_i,C_j)$; the covariance envelope uses "
        r"$W_{2,\Gamma}(P_i,P_j)$.",
        ha="center",
        fontsize=8,
        color=_C["gray"],
    )
    ax.set_title("Observed characteristics and latent random exposures")
    fig.tight_layout()
    fit_to_width(fig, 0.95)
    _save(fig, out)


def _fig_diversification(out: Path, results: pd.DataFrame) -> None:
    _style()
    df = results
    if "constraint" in df.columns:
        df = df[df["constraint"] == "long_only"]
    if "period" in df.columns:
        df = df[df["period"] == "full"]
    if "source" in df.columns:
        df = df[df["source"] == "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=figsize(0.95, 3.2 / 6.8))
    for method, color, sty, label in (
        ("sample", _C["vermilion"], "-", "Sample"),
        ("sigma_dist", _C["blue"], "-", r"$\hat{\Sigma}_{\mathrm{W2}}$"),
    ):
        sub = df[df["method"] == method].sort_values("window")
        axes[0].plot(
            sub["window"], sub["eff_n"], "o" + sty, color=color, label=label, lw=1.05
        )
        axes[1].plot(
            sub["window"], sub["hhi"], "s" + sty, color=color, label=label, lw=1.05
        )
    axes[0].set_title("Effective number of assets")
    axes[1].set_title("HHI concentration")
    for ax, ylab in zip(axes, (r"eff. $N$", "HHI"), strict=True):
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(ylab)
        ax.grid(visible=True, linestyle=":", alpha=0.45)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Diversification: W2 target vs sample (long-only)", fontsize=10, y=1.02
    )
    fig.tight_layout()
    fit_to_width(fig, 0.95)
    _save(fig, out)


def _fig_rolling_kappa(out: Path, results: pd.DataFrame) -> bool:
    """Rolling-kappa stability plot: median per-window kappa across T.

    Returns ``True`` if the figure was built, ``False`` if the persisted
    per-window kappa column is not yet available in ``results`` (deferred
    upstream repro) — callers must not treat the latter as an error.
    """
    if "kappa" not in results.columns:
        logger.warning(
            "rolling-kappa figure skipped: 'kappa' column absent from "
            "results.parquet (deferred paper3_empirical repro not yet run)"
        )
        return False
    _style()
    df = results
    if "constraint" in df.columns:
        df = df[df["constraint"] == "long_only"]
    if "source" in df.columns:
        df = df[df["source"] == "baseline"]
    if "period" in df.columns:
        df = df[df["period"] == "full"]
    df = df[df["method"] == "sigma_dist"].sort_values("window")
    if df.empty:
        logger.warning("rolling-kappa figure skipped: no sigma_dist rows found")
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(df["window"], df["kappa"], "o-", color=_C["blue"], lw=1.3)
    ax.set_xlabel(r"Estimation window $T$")
    ax.set_ylabel(r"Median calibrated $\kappa$")
    ax.set_title(r"Stability of $\kappa$ across rebalancing windows")
    ax.grid(visible=True, linestyle=":", alpha=0.45)
    fig.tight_layout()
    _save(fig, out)
    return True


def _fig_pit_comparison(out: Path, results: pd.DataFrame) -> bool:
    """Point-in-time vs fixed-matrix comparison figure.

    Returns ``True`` if built, ``False`` if the ``source`` column (baseline
    vs pit) is not yet persisted upstream.
    """
    if "source" not in results.columns:
        logger.warning(
            "PIT-comparison figure skipped: 'source' column absent from "
            "results.parquet (deferred paper3_empirical repro not yet run)"
        )
        return False
    _style()
    df = results
    if "constraint" in df.columns:
        df = df[df["constraint"] == "long_only"]
    if "period" in df.columns:
        df = df[df["period"] == "full"]
    df = df[df["method"] == "sigma_dist"]
    windows = sorted(df["window"].unique())
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(windows))
    width = 0.32
    for i, (src, label, color) in enumerate(
        (
            ("baseline", "Fixed matrix", _C["vermilion"]),
            ("pit", "Point-in-time", _C["blue"]),
        )
    ):
        sub = df[df["source"] == src].set_index("window")
        vals = [
            float(sub.loc[w, "sharpe_net_10bps"]) if w in sub.index else np.nan
            for w in windows
        ]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=label,
            color=color,
            edgecolor=_C["black"],
            lw=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in windows])
    categorical_axis(ax)
    ax.set_xlabel(r"Estimation window $T$")
    ax.set_ylabel(r"Net Sharpe @ 10 bps")
    ax.set_title(r"Point-in-time vs fixed W2 target ($\hat{\Sigma}_{\mathrm{W2}}$)")
    ax.legend()
    ax.axhline(0, color=_C["gray"], lw=0.6)
    ax.grid(visible=True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    _save(fig, out)
    return True


def render_paper3_figures(paths: Paper3FigurePaths) -> None:
    """Render Paper 3 figures and generated LaTeX bindings."""
    paper3_dir = paths.paper3_dir
    pricefree_dir = paths.pricefree_dir
    output_dir = paths.output_dir
    numbers_path = paths.numbers_path
    params_path = paths.params_path
    mc_validation_dir = paths.mc_validation_dir
    hedging_validation_dir = paths.hedging_validation_dir
    cross_model_summary = paths.cross_model_summary
    certificate_dir = paths.certificate_dir

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    p3_sum = _load_yaml(paper3_dir / "summary.yaml")
    results = pd.read_parquet(paper3_dir / "results.parquet")
    params = _load_yaml(params_path) if params_path.exists() else {}
    mc_validation = (
        _load_yaml(mc_validation_dir / "summary.yaml")
        if (mc_validation_dir / "summary.yaml").exists()
        else {}
    )
    hedging_validation = (
        _load_yaml(hedging_validation_dir / "summary.yaml")
        if (hedging_validation_dir / "summary.yaml").exists()
        else {}
    )
    cross_model_sum = (
        _load_yaml(cross_model_summary) if cross_model_summary.exists() else {}
    )
    _write_paper3_numbers(
        numbers_path,
        Paper3NumberInputs(
            summary=p3_sum,
            results=results,
            params=params,
            mc_validation=mc_validation,
            hedging_validation=hedging_validation,
            cross_model_summary=cross_model_sum,
            certificate_summary=(
                _load_yaml(certificate_dir / "summary.yaml")
                if (certificate_dir / "summary.yaml").exists()
                else {}
            ),
        ),
    )

    _fig_cov_anatomy(output_dir / "covariance_anatomy.pgf")
    _fig_dual_status(output_dir / "dual_status.pgf")
    _fig_regime_map(output_dir / "regime_map.pgf", pricefree_dir / "summary.yaml")
    _fig_sharpe_bars(output_dir / "sharpe_by_method.pgf", results)
    _fig_shrinkage_weight(output_dir / "shrinkage_weight.pgf", results)
    _fig_hedging_geometry(output_dir / "hedging_geometry.pgf")
    _fig_diversification(output_dir / "diversification.pgf", results)
    _fig_rolling_kappa(output_dir / "rolling_kappa.pgf", results)
    _fig_pit_comparison(output_dir / "pit_comparison.pgf", results)
    _fig_certificate_surface(
        output_dir / "certificate_surface.pgf",
        pd.read_parquet(certificate_dir / "surface.parquet"),
    )
