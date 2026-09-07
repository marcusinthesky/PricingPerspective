"""Deterministic per-cell PRNGKey generation.

The core primitive is :func:`cell_key`, which maps ``(base_seed, *labels)``
to a unique, independent :class:`jax.Array` PRNGKey.  Grid reordering
**never shifts streams**: each cell's key depends only on the full
``(base_seed, label₁, label₂, …)`` tuple, not on iteration order.

Design
------
Labels are converted to a stable 32-bit integer via MD5 (deterministic
across Python processes, unlike ``hash()`` which is salted since Python 3.3).
:func:`jax.random.fold_in` folds each label's hash into the running key
sequentially, so the key is order-sensitive:
``cell_key(s, "n10", "d8") != cell_key(s, "d8", "n10")``.
This is intentional — callers should fix a canonical label ordering.

Typical use: a parameter-grid Monte-Carlo experiment where each grid cell
(e.g. ``(n_assets, embedding_dim)``) needs its own independent, reproducible
PRNG stream regardless of the order cells are visited or parallelised.
"""

from __future__ import annotations

import hashlib
from functools import partial

import jax

from jcor.core.typing import Array, Num, PRNGKey, Static  # noqa: TC001

__all__ = ["cell_key", "permutation_batch"]


def _label_to_uint32(label: object) -> int:
    """Convert an arbitrary label to a stable 32-bit unsigned integer.

    Uses MD5 of ``str(label)`` encoded as UTF-8; takes the first 4 bytes.
    Stable across Python processes (no hash-randomisation).

    Args:
        label: Any object with a string representation.

    Returns:
        A 32-bit integer in ``[0, 2³²-1]``.

    """
    raw = str(label).encode("utf-8")
    digest = hashlib.md5(raw, usedforsecurity=False).digest()
    # Take first 4 bytes as little-endian unsigned int.
    return int.from_bytes(digest[:4], byteorder="little") & 0xFFFF_FFFF


def cell_key(base_seed: int, *labels: object) -> PRNGKey:
    """Return a deterministic PRNGKey for a grid cell identified by labels.

    The key is derived by:
      1. Creating a root key from ``base_seed`` via :func:`jax.random.PRNGKey`.
      2. Folding in each label's stable 32-bit hash via
         :func:`jax.random.fold_in`.

    Grid reordering never shifts streams because each cell's key depends
    only on the complete ``(base_seed, *labels)`` tuple.

    Args:
        base_seed: Global base seed for the experiment.
        *labels: Ordered sequence of identifiers for this grid cell, e.g.
            ``n_assets``, ``embedding_dim``.  Use a canonical ordering.

    Returns:
        JAX PRNGKey array for this cell.

    Examples:
        >>> k1 = cell_key(42, 10, 8)  # n_assets=10, d=8
        >>> k2 = cell_key(42, 20, 8)  # different cell → different key
        >>> k3 = cell_key(42, 10, 8)  # same cell → identical key
        >>> bool(jnp.all(k1 == k3))
        True

    """
    key = jax.random.PRNGKey(base_seed)
    for label in labels:
        data = _label_to_uint32(label)
        key = jax.random.fold_in(key, data)
    return key


@partial(jax.jit, static_argnames=("n_permutations",))
def permutation_batch(
    key: PRNGKey,
    values: Num[Array, "items"],  # noqa: F821, UP037  # jaxtyping shape
    n_permutations: Static[int],
) -> Num[Array, "permutations items"]:  # noqa: F722  # jaxtyping shape
    """Draw a batch of independent permutations under one compiled transform.

    ``n_permutations`` determines the leading result shape and is therefore a
    static argument. Public callers validate that it is positive before this
    transformable primitive is entered.

    Args:
        key: JAX key derived from :func:`cell_key`.
        values: One-dimensional values to permute.
        n_permutations: Static number of independent permutations.

    Returns:
        Permuted values with shape ``(n_permutations, values.shape[0])``.

    """
    keys = jax.random.split(key, n_permutations)
    return jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, values)
