"""S1 — raw sample clouds, before any pairwise work.

Position
--------
rank 1

Consumes
--------
raw arrays

Produces
--------
packed clouds, reduced/projected samples, reference alternatives

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

Unlike the other stages, ``sample`` was populated by t46.3 itself rather than
by a wave-B agent: its three modules (``packed``, ``reduction``,
``alternatives``) were already at the ``jcor`` root before t46 and appeared in
no wave-B ``Owns`` list. They import nothing from ``jcor`` at all, which is what
makes them genuine rank-1 leaves. Re-exports are appended per-owner, in the
region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---

# --- t46.3 (packed, reduction, alternatives) ---
from jcor.sample.alternatives import AltSpec, draw
from jcor.sample.packed import (
    DEFAULT_PAIR_CHUNK,
    PackedClouds,
    checked_packed_clouds,
    masked_mean,
    masked_row_mask,
    masked_sum,
    pack,
    pair_chunks,
)
from jcor.sample.reduction import Projection, fit_projection

__all__ += [
    "DEFAULT_PAIR_CHUNK",
    "AltSpec",
    "PackedClouds",
    "Projection",
    "checked_packed_clouds",
    "draw",
    "fit_projection",
    "masked_mean",
    "masked_row_mask",
    "masked_sum",
    "pack",
    "pair_chunks",
]
