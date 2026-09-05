"""Generators sub-package for the simulation library.

All generators are pure functions of a :class:`jax.random.PRNGKey` plus
static configuration arguments.  They produce **one replication**; callers
batch via :func:`jax.vmap`.

Sub-modules
-----------
fields  : Gaussian random field samplers (white, exponential, Matérn-3/2).
assets  : Cluster-structured exposure draws (``cluster_embeddings``).
returns : Factor-covariance and Gaussian panel return draws.

Alternative-hypothesis samplers for size/power experiments now live in
:mod:`jcor.sample.alternatives` (``AltSpec``, ``draw``) — generic two-sample-test
infrastructure with no simulation-specific coupling.
"""

from simulation.generators import assets, fields, returns

__all__ = ["assets", "fields", "returns"]
