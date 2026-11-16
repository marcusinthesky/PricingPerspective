"""Per-stage ownership of JAX's global float64 policy.

Statadic follows its caller's precision policy. Each numerical command whose
contract requires float64 calls :func:`own_float64` before constructing arrays.
Other commands retain their declared precision. Rank-sensitive dyadic fitting,
portfolio certificates and distance calculations require this explicit policy
because JAX's default float32 can change their numerical decisions.

Precision selection lives in command functions so Semflow includes it in each
stage's transitive numerical identity. Imports do not select precision. Spawned
worker entrypoints must select precision and configure the shared compilation
cache before constructing their first JAX arrays.
"""

from __future__ import annotations


def own_float64() -> None:
    """Enable JAX float64 for this stage process.

    Call as the first statement of a CLI command whose numerical contract is
    float64 -- before the body constructs its first array, since JAX
    canonicalizes dtypes at construction and a later flip cannot recover
    precision already rounded away.

    **Called from the command function, not from the Typer callback, on
    purpose.** ``semflow`` fingerprints a stage from its callback's transitive
    symbol graph, and ``cli.app._bootstrap`` is not in it: an ownership table
    read there would be invisible to ``dvc status``, so moving a stage in or out
    of float64 would not invalidate the stage it changes. Calling it inside the
    command puts this module in the fingerprint, which is the whole point of the
    ``semflow`` layer. The call sites are the registry -- there is no
    second list to drift out of sync with them.
    """
    # Imported here so that importing this module does not itself pull in JAX.
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", val=True)
