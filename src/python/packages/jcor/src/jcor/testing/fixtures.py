"""Fixtures every workspace member that depends on jcor inherits.

Registered as a ``pytest11`` entry point (see this package's ``pyproject.toml``),
so a member gets these by depending on ``jcor`` — no ``conftest.py`` copy, no
``pytest_plugins``. The latter is not an option: pytest honours it only in the
**rootdir** conftest, and in this uv workspace the rootdir is ``src/python``,
which owns no conftest at all.

**What does not belong here.** ``tests/conftest.py`` installs jaxtyping's
``install_import_hook("jcor", "beartype.beartype")`` for jcor's own suite. That
stays in the conftest deliberately: an entry-point plugin loads for *every*
member's pytest run, and beartype-instrumenting all of jcor for
``pipeline``/``simulation`` would be a large, silent slowdown of suites that
are not testing jcor's annotations.

**Why the imports are inside the fixtures.** The plugin module is imported at
pytest startup by every run in the shared venv, including ``crawler``,
``dotell`` and ``semflow``, none of which touch JAX. A module-level
``import jax`` would put ~1 s of import on each of them for fixtures they never
request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

__all__ = ["prng_key", "seeded_key", "tolerances"]

#: Seed every ``prng_key`` consumer shares. A constant, not a per-test draw:
#: these suites assert numerical identities, and a fixture that varied the seed
#: would turn a tolerance failure into an unreproducible one.
SEED = 42

#: Absolute/relative tolerances for comparing float64 statistics accumulated over
#: O(n^2) terms. 1e-9 absolute would report summation noise as a real difference;
#: see ``tests/_cases.py``, which sets the same pair for the axiom battery.
ATOL = 1e-9
RTOL = 1e-7


@pytest.fixture
def prng_key() -> jax.Array:
    """Return a deterministic PRNG key for reproducible tests.

    Returns:
        ``jax.random.PRNGKey(SEED)``.

    """
    import jax  # noqa: PLC0415  # see the module docstring: startup cost

    return jax.random.PRNGKey(SEED)


@pytest.fixture
def seeded_key() -> Callable[[int], jax.Array]:
    """Return a factory for independent, reproducible PRNG keys.

    Use this where one test needs several unrelated draws — folding a per-draw
    integer into the shared seed keeps them independent without introducing a
    second literal seed that has to be kept in step with :data:`SEED`.

    Returns:
        ``make(index)`` returning a key derived from ``SEED`` and ``index``.

    """
    import jax  # noqa: PLC0415  # see the module docstring: startup cost

    def make(index: int) -> jax.Array:
        return jax.random.fold_in(jax.random.PRNGKey(SEED), index)

    return make


@pytest.fixture
def tolerances() -> tuple[float, float]:
    """Return the shared ``(atol, rtol)`` for float64 statistical comparisons.

    Returns:
        ``(ATOL, RTOL)``.

    """
    return ATOL, RTOL
