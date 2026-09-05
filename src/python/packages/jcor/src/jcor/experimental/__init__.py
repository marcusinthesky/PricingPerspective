"""Staging area for ported-but-unstabilized methods.

Position
--------
no rank — stable stages may NOT import this package

Consumes
--------
anything

Produces
--------
anything.  **NO semver promise.**  Promote into a stage once the result
contract and the axiom battery pass

Boundary rule (t46): a stage may import only *earlier* stages, plus
``jcor.core`` and ``jcor.optimize``. Siblings inside a stage may import
each other so long as the graph stays acyclic. Enforced by
stage-layer review and ``just python::analyze-cycles``
(within-stage).

This package is a t46 skeleton: modules land here via the t46.4-t46.9
migrations. Re-exports are appended per-owner, in the region marked below.
"""

from __future__ import annotations

__all__: list[str] = []

# --- t46 migration re-exports; each task appends only to its own region ---
