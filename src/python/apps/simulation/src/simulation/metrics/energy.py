"""Simulation-facing compatibility wrappers for canonical energy statistics.

The V-statistic definition and bounded upper-triangle implementation live in
:mod:`jcor.discrepancy.energy`.  This module retains the simulation API and its
domain-specific defaults without carrying a second implementation of the same
arithmetic.

Why jaxtyping and ``jcor.ground`` are imported at runtime here
--------------------------------------------------------------
Under ``from __future__ import annotations`` a ``TYPE_CHECKING``-guarded name is
unresolvable at runtime, and beartype then **silently skips** the annotation
rather than failing — the probe is recorded in :mod:`jcor.core.typing`'s module
docstring. Leaving ``flake8-type-checking``'s ``strict`` mode to push the
jaxtyping import back under the guard would make every shape string below
decoration in *both* planes, since pyrefly does not verify shape algebra either.

``GroundDistanceSelection`` is un-guarded for the same reason and gains a real
check: unlike its two siblings in :mod:`simulation.metrics.dispatch` (see that
module's docstring for why those stay guarded), it annotates the ``metric``
*parameter* of :func:`pairwise_energy_distance` — the boundary at which the
value dispatch forwards is validated against the closed ``Literal`` behind the
alias. The import costs nothing: the ``jcor.discrepancy.energy`` line above it
already executes ``jcor/__init__.py``, which eagerly imports the ``ground``
subpackage, so this binds an already-initialised module.
"""

from __future__ import annotations

import jax
from jaxtyping import Array, Float  # noqa: TC002  # runtime; see docstring
from jcor.discrepancy.energy import energy_distance, energy_distance_matrix
from jcor.ground import GroundDistanceSelection  # noqa: TC002  # runtime


def pairwise_energy_distance(
    samples: Float[Array, "n m d"],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
    row_chunk: int = 8,
) -> Float[Array, "n n"]:
    """Compute pairwise energy distances through the canonical jcor primitive.

    Delegates to :func:`jcor.discrepancy.energy.energy_distance_matrix`.

    This compatibility wrapper preserves simulation's stacked-panel input,
    Euclidean default, and ``row_chunk`` parameter while delegating the
    V-statistic and bounded upper-triangle evaluation to the canonical jcor
    primitive.  The canonical primitive owns pair chunking, so ``row_chunk``
    is retained for call compatibility but has no execution effect.

    Args:
        samples: Panel of sample clouds, shape ``(n, m, d)``.
        exponent: Distance exponent α ∈ (0, 2).  Must match the value used
            when calling :func:`jcor.discrepancy.energy.energy_distance`.
        metric: Base distance metric passed to :func:`jcor.ground.metrics.cdist`.
            Default ``"euclidean"``.
        row_chunk: Legacy simulation chunk-size parameter. Accepted for API
            compatibility; jcor's canonical pair-chunk policy is used.

    Returns:
        Symmetric (n, n) matrix; diagonal is zero by construction.

    """
    _ = row_chunk
    sample_list = [samples[i] for i in range(samples.shape[0])]
    return energy_distance_matrix(sample_list, exponent=exponent, metric=metric).values


def energy_distances_to_target(
    target: Float[Array, "m d"],
    candidates: Float[Array, "K m d"],
    exponent: float = 1.0,
    metric: GroundDistanceSelection = "euclidean",
) -> Float[Array, " K"]:
    """Energy distance from each candidate cloud through canonical jcor scalars.

    This is an intentional target-to-many compatibility wrapper: unlike the
    equal-size matrix API, its output omits target-to-target and
    candidate-to-candidate entries.  It therefore vmaps the canonical scalar
    statistic, matching the batched hedging-error boundary.

    Args:
        target: Fixed reference sample cloud, shape ``(m, d)``.
        candidates: K candidate sample clouds, shape ``(K, m, d)``.
        exponent: Distance exponent α ∈ (0, 2).
        metric: Base metric passed to :func:`jcor.ground.metrics.cdist`.

    Returns:
        Array of shape ``(K,)`` with energy distances to *target*.

    """
    return jax.vmap(
        lambda candidate: energy_distance(
            target, candidate, exponent=exponent, metric=metric
        )
    )(candidates)
