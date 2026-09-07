"""Paper 1 referee-response diagnostic figures.

Split out from ``figures/paper1.py`` because these diagnostic figures depend on
a different data slice (the per-width truncated W2 tests for Figure 1; the
full-width W2 distances + the universe sector table for Figure 2) and were
added later, purely to pre-empt referee objections. See
``figures/paper1_illustrations.py`` docstring for the rationale behind
splitting figure modules by dependency closure rather than growing one file.

FIGURE 1 (``matryoshka_distance_preservation.pgf``): the manuscript truncates
embeddings to k=64 with almost no change in the return association. The claim
under test is about the *distance matrix*, so this plots the truncated W2
distance against the full-width one per dyad, at each reported width, with the
rank correlation annotated.

An earlier version of this figure plotted the variance-explained spectrum of
the pooled embedding matrix instead. That was the wrong diagnostic and it
argued against the manuscript: in native Matryoshka order the leading 64
coordinates carry ~3% of raw variance, which says nothing about whether the
induced geometry survives truncation. The measurement that does bear on the
claim is agreement between the truncated and full-width distance matrices, and
it turns out to be partial (Spearman ~0.75 at k=64) -- so the honest reading is
that the return-relevant structure is coarse enough to survive a lossy
compression, not that truncation is near-lossless. The paper's W2
pipeline uses full native coordinates with a Euclidean ground metric and no PCA
anywhere.

FIGURE 2 (``sector_energy_heatmap.pgf``): a same-sector co-membership matrix
next to the pairwise W2-distance matrix, both over one sector-sorted
ticker order, so a reader can see cross-sector pairs sitting close in W2
space -- the visual counterpart of the paper's DISCO sector-dispersion
number.

The figure implementations live in the private :mod:`.._diagnostics` package
(t45); this module owns the ``Paper1DiagnosticPaths`` contract, the
``render_paper1_diagnostics`` entry point, and the helper surface the p1 CLI and
``tests/papers/paper1/test_paper1_fwl_panel.py`` address by name.
"""

from __future__ import annotations

from pipeline.figures.paper1._diagnostics.contracts import (
    PRIMARY_SPECIFICATION_INDEX,
    Paper1DiagnosticPaths,
)
from pipeline.figures.paper1._diagnostics.driver import render_paper1_diagnostics
from pipeline.figures.paper1._diagnostics.fwl import (
    _fig_fwl_primary,
    _panel_fwl,
    _primary_slope,
)

# `_fig_fwl_primary`, `_panel_fwl` and `_primary_slope` are imported by
# `tests/papers/paper1/test_paper1_fwl_panel.py:22` through this module path
# (t45). NOTE for anyone adding a monkeypatch: `_save` is now resolved in
# `_diagnostics.<module>`'s globals, not here -- patch the owning module.
__all__ = [
    "PRIMARY_SPECIFICATION_INDEX",
    "Paper1DiagnosticPaths",
    "_fig_fwl_primary",
    "_panel_fwl",
    "_primary_slope",
    "render_paper1_diagnostics",
]
