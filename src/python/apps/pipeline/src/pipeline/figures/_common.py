"""Shared styling, IO, colour palette, and figure scaffolding for publication figures.

Every figure module draws from ``_C`` (line/marker colours) and ``_TINT`` (pale region
fills) rather than inlining hex literals — ``tests/figures/test_palette_discipline.py``
enforces that. Use :func:`figure` for the standard style/create/save scaffold.
"""

from __future__ import annotations

import logging
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl

# PGF, not Agg: manuscript figures are written as `.pgf` so Tectonic typesets their
# text at document-compile time, in the document's own font at its own point size.
# The backend is global rather than a per-`savefig` format because every figure is
# measured by `_saved_width_in`, which writes the real thing — the measurement and
# the artifact must come from one renderer, or the fitted size describes a figure
# nobody ships. Requires `pdflatex`, which the dev shell provides as `pgf-metrics`;
# matplotlib uses it only to query text extents and never to render an artifact.
mpl.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import NullLocator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# `\textwidth` of the manuscripts, in inches: a4paper with margin=30mm gives
# 210mm - 60mm = 150mm. Authored figure widths MUST be this, or a deliberate
# fraction of it — a PGF figure is placed at natural size and there is no width
# argument to rescale it, which is the whole point: text comes out at the
# document's own point size instead of whatever a scale factor happened to be.
#
# This is the `arxiv`/`draft` measure, and the only one: per-target variants
# would double the figure build, and the shared body may not be conditional on
# its target.
TEXTWIDTH_IN = 5.906


def figsize(frac: float, aspect: float) -> tuple[float, float]:
    r"""Return the authored size for a figure printed at ``frac`` of ``\textwidth``.

    Authoring a figure wider than it prints and letting the manuscript shrink it also
    shrinks its text: a 10.2in figure placed at 0.8 of the text block renders its 8pt
    tick labels at 3.7pt against 11pt body copy. Authoring at the printed size instead
    makes rcParam point sizes mean what they say.

    ``frac`` must match the ``\ppfigure{<frac>}{...}`` in the manuscript that includes
    it; ``aspect`` is height/width, so passing the original ratio preserves the printed
    geometry exactly while correcting the type size.
    """
    width = TEXTWIDTH_IN * frac
    return (width, width * aspect)


def categorical_axis(ax: plt.Axes, which: Literal["x", "y", "both"] = "x") -> None:
    """Drop minor ticks from an axis whose coordinates are category indices.

    ``_style()`` turns minor ticks on globally, which is what a continuous or log
    axis wants and what a categorical one cannot use: matplotlib subdivides the
    integer positions, so four bars acquire eighteen minor ticks between them and
    an ``imshow`` heatmap gets ticks at fractional cells. Call this on any axis
    whose ticks were placed by ``set_xticks``/``set_yticks`` against a fixed label
    list, naming only the categorical side — a bar chart's value axis is
    continuous and keeps its minor ticks.
    """
    if which in {"x", "both"}:
        ax.xaxis.set_minor_locator(NullLocator())
    if which in {"y", "both"}:
        ax.yaxis.set_minor_locator(NullLocator())


# Okabe-Ito: the standard colourblind-safe qualitative set. Saturated, for lines,
# markers, and edges.
_C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#666666",
}

# Pale region fills — a register `_C` does not cover, since a saturated line colour
# used as a large patch fill reads as an opaque block. Named after the `_C` entry each
# one tints, so a fill and its edge can be stated as a matched pair.
#
# `blue`/`orange` is the CVD-safe opposition and is the ONLY sanctioned pair for a
# binary or two-way semantic distinction. Do not encode pass/fail as green/vermilion
# tints: under deuteranopia and protanopia both collapse to the same pale beige, and a
# fill carries no second channel the way an edge or hatch does.
_TINT = {
    "neutral": "#F8F8F8",
    "blue": "#DCEBF5",
    "orange": "#FBEEDA",
    "green": "#E8F8F5",
    "vermilion": "#FDEDEC",
}


@dataclass(frozen=True, slots=True)
class Grid:
    """Subplot geometry for :func:`figure`.

    A configuration object rather than six keyword parameters: the repo's arity
    ratchet (``tests/test_arity_architecture.py``) rejects the flat form, and these
    six always vary together as one decision about panel layout.
    """

    nrows: int = 1
    ncols: int = 1
    sharex: bool = False
    sharey: bool = False
    height_ratios: Sequence[float] | None = None
    width_ratios: Sequence[float] | None = None


_SINGLE_PANEL = Grid()


@contextmanager
def figure(
    path: Path,
    figsize: tuple[float, float],
    grid: Grid = _SINGLE_PANEL,
    frac: float | None = None,
) -> Iterator[Any]:
    """Style, create, and save a figure; close it even if the body raises.

    Folds the ``_style()`` / ``plt.subplots(...)`` / ``_save(fig, path)`` scaffold that
    every figure function repeats. Yields whatever ``plt.subplots`` returns, so both
    single- and multi-axes forms work::

        with figure(out / "x.pgf", (5.2, 3.4)) as (fig, ax):
            ax.plot(...)

        with figure(out / "y.pgf", (7.2, 3.0), Grid(1, 2, sharey=True)) as (fig, axes):
            axes[0].plot(...)

    ``_style()`` mutates global ``plt.rcParams``; calling it here means a figure
    function can no longer forget it and silently inherit the previous figure's state.

    On success the figure is written and closed. If the body raises, the figure is
    closed **without** being written — a half-drawn figure on disk would look like a
    successful render to every downstream consumer, including DVC.
    """
    _style()
    created = plt.subplots(
        grid.nrows,
        grid.ncols,
        figsize=figsize,
        sharex=grid.sharex,
        sharey=grid.sharey,
        height_ratios=grid.height_ratios,
        width_ratios=grid.width_ratios,
    )
    fig = created[0]
    try:
        yield created
    except BaseException:
        plt.close(fig)
        raise
    if frac is not None:
        fit_to_width(fig, frac)
    _save(fig, path)


_PAD_IN = 0.02
_FIT_ITERATIONS = 4
_FIT_TOLERANCE = 0.004

# PGF is the only format. `.pdf` was removed 2026-08-01: see `_save`'s docstring.
_FORMATS = {".pgf": "pgf"}

# Vector marks a single panel may carry. Dyad-level clouds are quadratic in the
# roster -- 1,326 pairs at 52 firms, 4,950 at 100 -- and PGF emits one path per
# mark, so a full cloud exhausted TeX's main memory (a 15 MB, 113k-line .pgf).
# `rasterized=True` fixes the size but makes matplotlib emit
# `\includegraphics` of a sidecar PNG, which the rQUF2e journal class rejects
# while the arXiv article class accepts it -- so rasterizing would build one
# venue and not the other. Thinning keeps every target on the same pure-vector
# path. The cap is set just above the 1,326 marks the 52-firm figures carried,
# which TeX handled comfortably.
_MAX_VECTOR_MARKS = 1500


def thin_for_vector(
    *columns: NDArray[np.float64], cap: int = _MAX_VECTOR_MARKS, seed: int = 20260901
) -> tuple[tuple[NDArray[np.float64], ...], int]:
    """Deterministically thin parallel arrays to ``cap`` marks.

    Returns the thinned columns and the ORIGINAL count, so a caller can label
    the exhibit with the population it was drawn from rather than silently
    showing fewer points than it claims. Sampling is without replacement under a
    fixed seed, so the same dyads are shown on every rebuild.
    """
    total = len(columns[0])
    if total <= cap:
        return columns, total
    index = np.random.default_rng(seed).choice(total, size=cap, replace=False)
    index.sort()
    return tuple(np.asarray(column)[index] for column in columns), total


# A `.pgf` states its own natural size in the first line of the picture, in inches.
# `tests/figures/test_print_size.py` reads the same number back off disk.
_PGF_WIDTH = re.compile(
    r"\\pgfpathrectangle\{\\pgfpointorigin\}\{\\pgfqpoint\{([0-9.]+)in\}"
)


def _saved_width_in(fig: plt.Figure) -> float:
    r"""Return the width ``_save`` would write for ``fig``, in inches.

    Measured by actually writing the figure, because no cheaper proxy is reliable.
    ``fig.get_tightbbox()`` is the obvious candidate and is wrong twice over: under
    ``constrained`` layout the solver re-runs during the save draw and moves the
    content box afterwards, and the PGF backend's own tight-bbox pass does not
    reproduce ``get_tightbbox`` exactly. Fitting against either left figures up to
    4% off while the loop reported convergence.

    Written to a scratch directory rather than an in-memory buffer: matplotlib
    refuses to stream PGF for a figure containing raster graphics ("streamed
    pgf-code does not support raster graphics"), because an ``imshow`` panel has to
    put its sidecar PNGs somewhere. A real directory gives it one, and it is
    discarded with the probe. The LaTeX metric process matplotlib uses is
    persistent across calls, so repeated writes do not re-pay its startup.
    """
    with tempfile.TemporaryDirectory() as scratch:
        probe = Path(scratch) / "probe.pgf"
        fig.savefig(probe, format="pgf", bbox_inches="tight", pad_inches=_PAD_IN)
        match = _PGF_WIDTH.search(probe.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else 0.0


def fit_to_width(fig: plt.Figure, frac: float) -> None:
    r"""Resize ``fig`` so the *saved* figure is ``frac`` of ``\textwidth`` wide.

    :func:`figsize` sizes the **canvas**, but ``_save`` writes with
    ``bbox_inches="tight"``, so the file on disk is the trimmed content box. For a
    plot whose axes fill the canvas the two agree to a few percent. For a schematic
    -- ``axis("off")`` with hand-placed patches, or an ``set_aspect("equal")`` axes
    -- the drawn content can occupy as little as 63% of the canvas. Under PDF that
    trimmed box was then scaled back up by ``\includegraphics``; under PGF it is
    placed at natural size and simply does not fill its column.

    Scaling the canvas scales patch geometry but *not* point-valued text, so the
    saved width moves by less than the scale factor and one pass under-corrects;
    hence the fixed-point loop. Call this immediately before saving, after every
    artist has been added.

    The loop gives up rather than iterating when an error stops improving. That
    happens when the widest artist is point-valued text -- a long ``suptitle``, say
    -- which does not move at all as the canvas is scaled: continuing would shrink
    the canvas under a title that stays put, distorting the figure while the saved
    width never budges. In that case the figure keeps its authored size and the
    manuscript-fraction guard reports it, because the fix is editorial (shorten or
    downsize the offending text), not geometric.
    """
    target = frac * TEXTWIDTH_IN
    best_size = fig.get_size_inches().copy()
    best_error = float("inf")
    for _ in range(_FIT_ITERATIONS):
        saved = _saved_width_in(fig)
        if saved <= 0.0:
            break
        error = abs(saved - target)
        if error <= _FIT_TOLERANCE:
            return
        if error >= best_error:
            break
        best_error, best_size = error, fig.get_size_inches().copy()
        width, height = best_size
        scale = target / saved
        fig.set_size_inches(width * scale, height * scale)
    fig.set_size_inches(*best_size)


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _style() -> None:
    """Apply the shared rcParams. Idempotent; :func:`figure` calls it for you."""
    # `text.usetex` stays False. The PGF backend routes every label through LaTeX
    # regardless — `RendererPgf.get_text_width_height_descent` calls `LatexManager`
    # unconditionally — so setting it changes nothing except to demand `cm-super`,
    # which `pgf-metrics` does not carry. `pgf.rcfonts=False` is what makes a figure
    # inherit the document's own font rather than restating matplotlib's.
    # See `.context/plan/infra-tooling/tasks/t38-figures-pgf-pilot/README.md`.
    # The frame/tick block below is taken from SciencePlots' `science.mplstyle`
    # by hand rather than by depending on it: that style sets `text.usetex=True`
    # and a `prop_cycle` whose 2nd and 4th entries are a green/red pair, and it
    # sets neither `pgf.texsystem` nor `pgf.rcfonts`, so it could only layer
    # under this function and never replace it. See `src/latex/ARCHITECTURE.md`,
    # §Rejected alternatives.
    #
    # Its `xtick.top`/`ytick.right` are deliberately NOT taken: they draw ticks
    # on the two spines `axes.spines.*` above removes, leaving ticks floating
    # with nothing to sit on. Tick sizes and widths are taken with the direction,
    # because 0.8pt-wide default ticks on a 0.5pt spine read as a mismatch.
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # 20 of the 23 legend call sites passed this by hand; the three that
            # did not are in a paper-specific figure module and now lose their
            # frames too.
            "legend.frameon": False,
            "axes.linewidth": 0.5,
            "xtick.direction": "in",
            "xtick.major.size": 3,
            "xtick.major.width": 0.5,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.5,
            "xtick.minor.visible": True,
            "ytick.direction": "in",
            "ytick.major.size": 3,
            "ytick.major.width": 0.5,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.5,
            "ytick.minor.visible": True,
            "text.usetex": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    r"""Write ``fig`` as PGF and close it.

    ``.pgf`` is the only format, so figure text is typeset by Tectonic in the
    document's font at the document's point size.

    **Why ``.pdf`` is gone (2026-08-01).** It was kept for the figures no
    manuscript includes -- diagnostics read by opening them, which a ``.pgf``
    cannot be without a LaTeX run. That rationale was sound but the format never
    worked once t38 made PGF the global backend in ``7b87d1ad``: under PGF,
    ``savefig(format="pdf")`` does not use matplotlib's pure-Python PDF writer.
    It emits a standalone wrapper and shells out to ``pdflatex``, and that wrapper
    hard-codes ``\\usepackage{hyperref}`` and ``\\usepackage{geometry}``
    (``backend_pgf.py:848-850``, unconditional). The dev shell's ``pgf-metrics``
    closure carries neither -- it is measured for matplotlib's PGF *header* only
    (``infra/nix/pkgs/pgf-metrics.nix``) -- so every ``.pdf`` save died with
    ``! LaTeX Error: File `hyperref.sty' not found.`` several hundred log lines
    into a ``dvc repro``. Admitting a format that cannot be produced is worse than
    refusing it, so this now fails at the call site.

    To *look at* a figure, compile it: ``just latex::preview <path.pgf>``.

    A ``.pgf`` is not necessarily one file: ``imshow`` cannot be vector paths, so
    matplotlib writes ``<name>-imgN.png`` beside it and refers to each by bare
    filename. That is why manuscripts include figures with ``\\import`` rather than
    ``\\input``, and why the DVC ``outs:`` for a rasterising figure list sidecars.
    """
    if path.suffix not in _FORMATS:
        message = (
            f"unsupported figure format {path.suffix!r} for {path}; "
            f"figures are written as {', '.join(sorted(_FORMATS))}. "
            "To view one, run `just latex::preview <path.pgf>`."
        )
        raise ValueError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, format=_FORMATS[path.suffix], bbox_inches="tight", pad_inches=_PAD_IN
    )
    plt.close(fig)
    logger.info("Saved %s", path)
