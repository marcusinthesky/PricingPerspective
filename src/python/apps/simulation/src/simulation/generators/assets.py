"""Asset-exposure generators.

All generators are pure functions of a :class:`jax.random.PRNGKey` and static
config args, producing **one replication**.  Batch via :func:`jax.vmap`.

Source: ``src/python/apps/simulation/src/simulation/validation/variance_floor.py``,
``sample_exposures``.  The donor implementation used a Python loop
over assets; this port vectorises via a single :func:`jax.random.normal` draw
plus index-based gather.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Imported at RUNTIME, deliberately not under `if TYPE_CHECKING:`. Paired with
# `from __future__ import annotations`, a guarded `Float`/`Array` is an
# unresolvable name at runtime and beartype then **silently skips** the
# annotation rather than failing -- probed and recorded in
# `packages/jcor/src/jcor/core/typing.py:10-27`. `flake8-type-checking` in
# strict mode wants the import back inside the block; the `noqa` is that cost.
from jaxtyping import Array, Float, Int  # noqa: TC002


class ClusterSample(NamedTuple):
    """Return type of :func:`cluster_embeddings`.

    Attributes:
        samples: Per-asset sample clouds, shape ``(n, m, d)``.
        betas: Per-asset sample means (exposure vectors), shape ``(n, d)``.
        assignments: Cluster index of each asset, shape ``(n,)``.
        centers: Cluster centre vectors, shape ``(k, d)``.

    """

    # These four shape strings resolve at runtime but are still **inert** as a
    # gate, in both planes. The import hook decorates every class it rewrites
    # (`jaxtyping/_import_hook.py:170-188`), yet `jaxtyped` wraps `__init__`
    # only when `dataclasses.is_dataclass` holds and returns any other class
    # untouched (`jaxtyping/_decorator.py:294-308`) -- a `NamedTuple` is not a
    # dataclass. Nor does the return annotation reach them: beartype 0.22.9
    # still type-checks a `NamedTuple` subclass by `isinstance` alone, its
    # field-level check an open FIXME (`beartype/_util/hint/pep/proposal/
    # pep484/pep484namedtuple.py:20-26`). They are kept resolvable so
    # `get_type_hints` and any future deep check agree with the code, which
    # they do: `cluster_embeddings` builds exactly (n, m, d), (n, d), (n,),
    # (k, d) below. Enforcement would need a registered dataclass, not a
    # stronger string.
    samples: Float[Array, "n m d"]
    betas: Float[Array, "n d"]
    assignments: Int[Array, "n"]
    centers: Float[Array, "k d"]


def cluster_embeddings(
    key: jax.Array,
    n: int,
    m: int,
    d: int,
    k: int,
    center_scale: float = 1.0,
    spread: float = 0.3,
) -> ClusterSample:
    """Draw cluster-structured asset exposure samples.

    Replicates the distribution of the donor ``sample_exposures`` function
    without Python loops over assets or samples.

    Distribution:
        * ``centers[j] ~ N(0, center_scale² · I_d)``  for ``j = 0 … k-1``
        * ``assignments[i] ~ Uniform({0, …, k-1})``  for ``i = 0 … n-1``
        * ``samples[i, t] = centers[assignments[i]] + spread · ε``
          where ``ε ~ N(0, I_d)``
        * ``betas[i] = mean_t samples[i, t]``   (per-asset sample mean)

    Args:
        key: JAX PRNG key.
        n: Number of assets.
        m: Number of sample points per asset.
        d: Embedding dimension.
        k: Number of clusters.
        center_scale: Standard deviation of cluster centres.  Default ``1.0``.
        spread: Per-sample noise standard deviation around the cluster centre.
            Default ``0.3`` (matches donor).

    Returns:
        :class:`ClusterSample` named tuple.

    """
    key_centers, key_assign, key_noise = jax.random.split(key, 3)

    # (k, d) cluster centres
    centers = center_scale * jax.random.normal(key_centers, shape=(k, d))

    # (n,) cluster assignments — uniform over {0, …, k-1}
    assignments = jax.random.randint(key_assign, shape=(n,), minval=0, maxval=k)

    # (n, m, d) per-asset samples: gather center then add spread noise
    # centers[assignments] has shape (n, d); broadcast to (n, m, d)
    assigned_centers = centers[assignments]  # (n, d)
    noise = spread * jax.random.normal(key_noise, shape=(n, m, d))
    samples = assigned_centers[:, None, :] + noise  # (n, m, d)

    # (n, d) per-asset means
    betas = jnp.mean(samples, axis=1)  # (n, d)

    return ClusterSample(
        samples=samples,
        betas=betas,
        assignments=assignments,
        centers=centers,
    )
