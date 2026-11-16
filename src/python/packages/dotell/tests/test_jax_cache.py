"""Unit tests for the shared JAX compilation-cache policy (t63).

These assert the *configuration contract* -- which process-global JAX flags the
helper leaves set, and that the on-disk cache is left in a state JAX's LRU
eviction can actually write to -- not cache hit behaviour, which needs a real
compile. The retired Paper 2 spawn-runtime suite recorded that end-to-end check;
these unit tests make no cache-hit performance claim.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from dotell import configure_jax_compilation_cache
from dotell._jax_cache import (
    CACHE_DIR_ENV,
    MAX_GIB_ENV,
    RESOLVED_DIR_ENV,
    STRICT_ENV,
)
from jax._src.lru_cache import LRUCache

if TYPE_CHECKING:
    from collections.abc import Iterator

_BYTES_PER_GIB = 2**30
_DEFAULT_MAX_SIZE = 2 * _BYTES_PER_GIB
_UNLIMITED = -1
# An implausible ns timestamp, so a restamp by the backfill is unmistakable.
_SENTINEL_ATIME = 123

_FLAGS = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_entry_size_bytes",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_compilation_cache_max_size",
    "jax_raise_persistent_cache_errors",
)


class _JaxConfig(Protocol):
    """The JAX options this module reads -- exactly the names in ``_FLAGS``.

    ``jax.config`` grows these as attributes at runtime (they exist only once
    ``jax.config.update(<name>, ...)`` has declared them, and appear in no stub),
    so a statically-typed read needs a hand-written surface. Spelled out rather
    than annotated ``Any``: ``[[tool.pyrefly.sub-config]]`` in
    src/python/pyproject.toml raises ``explicit-any`` from warn to ERROR under
    ``packages/dotell/**``, and the dynamic surface here is finite and
    already enumerated. ``jax_compilation_cache_dir`` is ``str | None`` because the
    autouse fixture reads it *before* ``configure_jax_compilation_cache`` runs,
    where JAX's own default is ``None``; the helper only ever writes a ``str``.
    """

    jax_compilation_cache_dir: str | None
    jax_persistent_cache_min_entry_size_bytes: int
    jax_persistent_cache_min_compile_time_secs: float
    jax_compilation_cache_max_size: int
    jax_raise_persistent_cache_errors: bool


def _config() -> _JaxConfig:
    """Read JAX options dynamically; they are invisible to the type checker."""
    return cast("_JaxConfig", jax.config)


@pytest.fixture(autouse=True)
def _isolate_process_globals() -> Iterator[None]:
    """Undo the process-global JAX flag and environment writes after each test.

    ``configure_jax_compilation_cache`` republishes its resolved dir into
    ``PP_JAX_CACHE_DIR_RESOLVED`` directly, so ``monkeypatch`` does not know to
    revert it and it would otherwise leak into every later test.
    """
    saved_flags = {name: getattr(_config(), name) for name in _FLAGS}
    saved_resolved = os.environ.get(RESOLVED_DIR_ENV)
    os.environ.pop(RESOLVED_DIR_ENV, None)
    yield
    for name, value in saved_flags.items():
        jax.config.update(name, value)
    if saved_resolved is None:
        os.environ.pop(RESOLVED_DIR_ENV, None)
    else:
        os.environ[RESOLVED_DIR_ENV] = saved_resolved


def test_explicit_dir_wins_and_is_republished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit dir overrides the env var and is created and republished."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "ignored"))
    monkeypatch.delenv(RESOLVED_DIR_ENV, raising=False)
    target = tmp_path / "explicit"

    resolved = configure_jax_compilation_cache(target)

    assert resolved == target.resolve()
    assert resolved.is_dir()
    assert _config().jax_compilation_cache_dir == str(resolved)
    # Republished so spawned children share the parent's directory...
    assert Path(os.environ[RESOLVED_DIR_ENV]) == resolved
    # ...without shadowing the documented knob, whose `.env`-vs-process-env
    # precedence a child's `runtime_settings()` would otherwise see inverted.
    assert Path(os.environ[CACHE_DIR_ENV]) == tmp_path / "ignored"


def test_env_dir_is_read_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env var is honoured when set *after* import (the pre-t63 bug)."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "late"))
    monkeypatch.delenv(RESOLVED_DIR_ENV, raising=False)

    resolved = configure_jax_compilation_cache()

    assert resolved == (tmp_path / "late").resolve()


def test_child_inherits_the_parents_resolved_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawned child reuses the parent's dir over its own default."""
    parent_dir = tmp_path / "from-parent"
    monkeypatch.setenv(RESOLVED_DIR_ENV, str(parent_dir))
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "would-be-mine"))

    assert configure_jax_compilation_cache() == parent_dir.resolve()


def test_relative_dir_is_resolved_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative default becomes absolute, so a child's cwd cannot shift it."""
    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = configure_jax_compilation_cache()

    assert resolved.is_absolute()
    assert resolved == (tmp_path / ".jax_cache").resolve()


def test_default_policy_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults: 0.5s compile floor, JAX-chosen entry floor, 2 GiB cap, lenient."""
    monkeypatch.delenv(MAX_GIB_ENV, raising=False)
    monkeypatch.delenv(STRICT_ENV, raising=False)

    configure_jax_compilation_cache(tmp_path / "cache")

    config = _config()
    # 0 defers to `default_min_cache_entry_size()`, which JAX substitutes at
    # cache init -- it is "let JAX pick", not "no floor".
    assert config.jax_persistent_cache_min_entry_size_bytes == 0
    assert config.jax_persistent_cache_min_compile_time_secs == pytest.approx(0.5)
    assert config.jax_compilation_cache_max_size == _DEFAULT_MAX_SIZE
    assert config.jax_raise_persistent_cache_errors is False


@pytest.mark.parametrize("raw", ["0", "-1", "not-a-number", "4", ""])
def test_max_size_never_disables_caching(
    raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env value may yield 0, which is JAX's "caching off" sentinel."""
    monkeypatch.setenv(MAX_GIB_ENV, raw)

    configure_jax_compilation_cache(tmp_path / "cache")

    assert _config().jax_compilation_cache_max_size != 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", _UNLIMITED), ("-1", _UNLIMITED), ("4", 4 * _BYTES_PER_GIB)],
)
def test_max_size_env_mapping(
    raw: str, expected: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-positive GiB budget opts out of eviction; a positive one caps it."""
    monkeypatch.setenv(MAX_GIB_ENV, raw)

    configure_jax_compilation_cache(tmp_path / "cache")

    assert _config().jax_compilation_cache_max_size == expected


def test_bad_max_size_falls_back_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unparsable budget must not take down all 34 cold-start DVC stages."""
    monkeypatch.setenv(MAX_GIB_ENV, "two gigs")

    configure_jax_compilation_cache(tmp_path / "cache")

    assert _config().jax_compilation_cache_max_size == _DEFAULT_MAX_SIZE


def test_legacy_entries_do_not_block_cache_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dir populated before the cap was enabled must stay writable.

    JAX writes a ``-atime`` sidecar only while eviction is on, but
    ``_evict_if_needed`` reads one for every ``*-cache`` file it finds. Without
    the backfill, the first write into a legacy directory raises
    ``FileNotFoundError`` and the cache silently stops accepting entries for
    good -- the regression that turning the cap on originally caused against
    this repo's own ``.jax_cache``.
    """
    monkeypatch.setenv(MAX_GIB_ENV, "1")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "jit_legacy-deadbeef-cache").write_bytes(b"stale executable")

    configure_jax_compilation_cache(legacy)

    assert (legacy / "jit_legacy-deadbeef-atime").is_file()
    # Drive JAX's own eviction path rather than trusting the sidecar's
    # existence: this is what raised before the backfill.
    cache = LRUCache(str(legacy), max_size=_BYTES_PER_GIB)
    cache.put("jit_new-cafe", b"fresh executable")
    assert cache.get("jit_new-cafe") == b"fresh executable"


def test_backfill_leaves_existing_sidecars_untouched(tmp_path: Path) -> None:
    """Re-running must not restamp live entries as freshly accessed."""
    cache_dir = tmp_path / "mixed"
    cache_dir.mkdir()
    (cache_dir / "jit_kept-1-cache").write_bytes(b"blob")
    sidecar = cache_dir / "jit_kept-1-atime"
    sidecar.write_bytes(_SENTINEL_ATIME.to_bytes(8, "little"))

    configure_jax_compilation_cache(cache_dir)

    assert int.from_bytes(sidecar.read_bytes(), "little") == _SENTINEL_ATIME


def test_backfill_writes_no_temp_files(tmp_path: Path) -> None:
    """The atomic-rename staging file must never survive a backfill."""
    cache_dir = tmp_path / "atomic"
    cache_dir.mkdir()
    (cache_dir / "jit_a-1-cache").write_bytes(b"blob")

    configure_jax_compilation_cache(cache_dir)

    assert not list(cache_dir.glob("*.tmp"))


def test_compile_populates_cache_under_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real compile writes an entry with eviction and strict mode both on."""
    monkeypatch.setenv(MAX_GIB_ENV, "1")
    monkeypatch.setenv(STRICT_ENV, "1")
    cache_dir = configure_jax_compilation_cache(tmp_path / "live")
    # This unit's kernels compile far faster than the production 0.5s floor.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    jax.jit(lambda x: (x @ x.T).sum()).lower(jnp.ones((8, 8))).compile()

    assert list(cache_dir.glob(f"*{'-cache'}"))


@pytest.mark.parametrize(
    ("raw", "expected"), [("1", True), ("true", True), ("0", False), ("", False)]
)
def test_strict_mode_is_opt_in(
    *, raw: str, expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict mode turns a silent cache-error recompile into a stage failure."""
    monkeypatch.setenv(STRICT_ENV, raw)

    configure_jax_compilation_cache(tmp_path / "cache")

    assert _config().jax_raise_persistent_cache_errors is expected
