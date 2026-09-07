"""Property batteries for verifying declared contracts — test-time helpers.

Not a stage. ``jcor.testing`` sits outside the
``sample → ground → discrepancy → geometry → operator → model → inference →
decision`` chain and holds no method code: it takes a callable and checks what
that callable actually does. It therefore imports ``jcor.core`` and nothing else
from ``jcor``, which keeps it importable from any stage's tests without creating
a cross-stage edge.

Ships in the wheel (rather than living under ``tests/``) so that downstream
packages — ``simulation``, ``pipeline`` — can assert the same axioms about their
own dissimilarities.

:mod:`jcor.testing.fixtures` is the pytest half of the same idea: it is
registered as a ``pytest11`` entry point, so those packages inherit the shared
fixtures by depending on jcor. It is deliberately **not** re-exported here — the
names below are library helpers a caller imports, while the fixtures are
supplied by pytest and must never be imported by name.
"""

from __future__ import annotations

__all__ = [
    "assert_axioms",
    "assert_negative_type",
    "dissimilarity_matrix",
    "measure_axioms",
    "negative_type_eigmin",
]

from jcor.testing.axioms import (
    assert_axioms,
    assert_negative_type,
    dissimilarity_matrix,
    measure_axioms,
    negative_type_eigmin,
)
