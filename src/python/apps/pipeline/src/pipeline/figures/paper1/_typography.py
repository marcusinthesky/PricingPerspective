"""Paper 1 chart typography: a 9 pt floor on every text element.

The PGF figures are placed at natural size in the arXiv manuscript, so the
floor keeps axis labels and legends legible at their printed size. Every
Paper 1 figure applies :func:`paper1_style` instead of the shared ``_style``
and sizes any explicit text at ``MIN_PT`` or larger.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from pipeline.figures._common import _style

MIN_PT = 9.0


def paper1_style() -> None:
    """Apply the shared rcParams, then raise every sub-9 pt default to the floor."""
    _style()
    plt.rcParams.update(
        {
            "font.size": MIN_PT,
            "axes.labelsize": MIN_PT,
            "legend.fontsize": MIN_PT,
            "legend.title_fontsize": MIN_PT,
            "xtick.labelsize": MIN_PT,
            "ytick.labelsize": MIN_PT,
        }
    )
