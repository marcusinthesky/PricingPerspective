"""``pytest11`` entry point for :mod:`jcor.testing.fixtures` — a deliberate shim.

The fixtures live in :mod:`jcor.testing.fixtures`. This module exists only to
delay importing them, and the reason is a measured regression rather than
tidiness.

pytest loads ``pytest11`` entry points in ``Config._preparse``, **before** it
loads any ``conftest.py``. ``jcor/__init__.py`` imports every stage eagerly, so
pointing the entry point straight at ``jcor.testing.fixtures`` imported the
whole package at that moment — which is strictly earlier than
``tests/conftest.py`` calling ``install_import_hook("jcor", "beartype.beartype")``.
jaxtyping's hook only instruments modules imported *after* it is installed, so
the effect was to silently disable runtime shape checking for the entire suite:
``tests/core/test_typing.py::test_dmat_shape_is_enforced_at_runtime`` was the
tripwire that caught it, and it is the only reason those shape strings are
load-bearing at all.

Registering in ``pytest_configure`` moves the import after conftest loading, so
the hook is in place first. The alternative — installing the hook here — would
beartype-instrument all of jcor for ``pipeline`` and ``simulation`` runs too,
which is a large slowdown of suites that are not testing jcor's annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the shared jcor fixtures once conftest loading is done.

    Args:
        config: The active pytest config, whose plugin manager receives the
            fixture module.

    """
    from jcor.testing import fixtures  # noqa: PLC0415  # see the module docstring

    config.pluginmanager.register(fixtures, "jcor-testing-fixtures")
