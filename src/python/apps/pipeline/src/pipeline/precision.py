"""Per-stage ownership of JAX's global float64 policy.

``jcor`` used to open a door-scoped ``jax.enable_x64`` inside 51 of its own
functions. That made the library the owner of a process-global, numerics-
affecting flag, which :mod:`dotell._jax_cache` already names as the one
thing the shared config site must never do: dtype policy is "owned per stage".
The doors are gone, so the ownership has to live here instead -- a library that
respects the caller's precision needs a caller that has one.

**Why per-stage and not a blanket flip.** Turning x64 on for every stage would
perturb stages that are deliberately float32 today. Legacy optimization code
(``:239``) is the worked example: it certifies its Frank-Wolfe gap in NumPy
float64 *because* ``qpax`` runs under JAX's float32 default, and says plainly
that flipping the global flag "would perturb every other JAX stage". The
commands that call :func:`own_float64` are exactly those whose transitive symbol
graph reaches the removed doors -- the pipeline entrypoints ``semflow`` reported
when the doors were deleted -- and nothing else.

The set is not frozen: a stage that starts reaching JAX joins it.
``compute-neighbour-outcome-split`` did so in t80, when it stopped computing its
own NumPy energy and began reading the shared geometry component through
``jcor.discrepancy.metrize.sqrt_energy_functional``. Unowned, that projection
truncated its float64 input to float32 and cost the contract's recorded
``raw_value`` about eight significant digits while leaving every 3-decimal
published binding unchanged -- which is exactly the kind of silent loss the
per-stage ownership rule exists to make impossible to ship unnoticed.

**Measured consequence of *not* owning it** (full ``dvc repro``, doors removed,
no caller ownership):

* ``compute-dyadic-confound`` / ``compute-dyadic-oos`` --
  ``DyadicComputationError: only 0/1999 valid node-bootstrap draws``. The rank
  selector in ``jcor.operators.design`` runs on designs measured at condition
  number >= 1e9 (worst 1e99), so float32 leaves its threshold test no signal.
* ``h1-pilot`` / ``h1-oos-coverage`` -- ``Wolak statistic nonnegative-QP solver
  rejected 1/1 lanes``.
* ``paper3-empirical`` -- ``psd_repair.converged`` flipped ``False -> True``, a
  false convergence that reaches published numbers through
  ``figures.paper3.numbers``.

Everything else drifted at a median relative move of 6.3e-6 (Mantel p-value
bit-identical), which is why the fix is scoped rather than global.

**Simulation app.** Not covered here and not needing it:
``simulation.backend.configure(precision="float64")`` already owns the flag, and
its four Monte Carlo stages came back byte-identical across the same run.

**Spawn.** A spawned interpreter inherits no process-global JAX flag, so a call
in the parent's command function reaches the parent only. The one stage tree
that fans out to ``spawn`` workers owns the flag at
module import (``_hedge/contracts.py:22``) precisely so its cold-importing
workers get it too; a stage that grows a process pool must do the same.
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
    ``semflow`` layer. The 21 call sites are the registry -- there is no
    second list to drift out of sync with them.
    """
    # Imported here so that importing this module does not itself pull in JAX.
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", val=True)
