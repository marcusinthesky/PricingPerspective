"""Canonical, host-side backtest scheduling primitives.

This module generates the *schedule* of estimation/evaluation slices that a
rolling backtest walks over — trailing, anchored (expanding-origin), and
disjoint-epoch families — plus a small point-in-time (PIT) fallback guard.
It does **not** run any estimator or optimizer; it is pure orchestration
structure, deliberately host-side (plain Python/array values, no ``jax``) per
the repo's t13 convention that scheduling/looping lives outside ``jit``.

The schedules here are re-derived from, and must stay bit-identical in their
indexing to, the three canonical backtest loops already used in production:

* :func:`trailing_schedule` reproduces the fixed-window rolling loop in
  ``pipeline.stages.papers.paper3.empirical.backtest`` (``rebal = max(1, win
  // 4)``; one eval step per ``t in range(win, T)``; rebalance on
  ``(t - win) % rebal == 0``).
* :func:`anchored_schedule` is the expanding-origin analogue (paper5-style
  anchored folds): same eval cadence, but the training slice always starts
  at 0.
* :func:`disjoint_epoch_schedule` is the trailing family confined to an
  absolute-index sub-epoch, matching the former energy-robustness rolling-within-
  fixed-epoch design.

This module is additive scaffolding for a future canonical backtest engine
(t17.2). Nothing here is wired into any paper stage or DVC pipeline yet.
"""

from __future__ import annotations

import math
import typing


class ScheduleEntry(typing.NamedTuple):
    """One estimation/evaluation step of a :class:`WindowSchedule`.

    The estimation slice is ``R[train_start:train_stop]``; the fitted
    weights are evaluated against ``R[eval_index]`` (and held until the next
    rebalance, drifting between rebalances). ``rebalance`` is ``True`` iff a
    fresh estimation is performed at this step.
    """

    train_start: int
    train_stop: int
    eval_index: int
    rebalance: bool


class WindowSchedule(typing.NamedTuple):
    """A generated schedule of :class:`ScheduleEntry` steps.

    ``window_type`` identifies the generating family (``"trailing"``,
    ``"anchored"``, or ``"disjoint"``); ``window`` and ``rebal`` record the
    parameters used to derive ``entries``.
    """

    window_type: str
    window: int
    rebal: int
    entries: tuple[ScheduleEntry, ...]


class BacktestConfig(typing.NamedTuple):
    """A fully-hashable description of one backtest sweep cell.

    Every field is a hashable, immutable value so a single ``BacktestConfig``
    instance can key a future DVC matrix cell / cache entry. See
    :func:`config_key` for a stable short string derived from it.
    """

    windows: tuple[int, ...] = ()
    cost_bps: tuple[float, ...] = ()
    scheme: str = "trailing"
    rebal_divisor: int = 4
    overlap_method: str = "stationary_bootstrap"
    block_length_scheme: str = "circular"
    embargo: int = 0
    pit_policy: str = "annual_vintage"
    min_articles: int = 30
    seed: int = 0
    n_boot: int = 1000


def config_key(cfg: BacktestConfig) -> str:
    """Return a stable, short, filesystem/DVC-matrix-safe key for ``cfg``.

    The key encodes scheme, window set, and cost-bps set compactly, then
    appends the remaining scalar knobs so distinct configs never collide.
    """
    windows_part = "-".join(str(w) for w in cfg.windows) or "none"
    costs_part = (
        "-".join(str(int(c)) if float(c).is_integer() else str(c) for c in cfg.cost_bps)
        or "none"
    )
    return (
        f"{cfg.scheme}_w{windows_part}_c{costs_part}"
        f"_rd{cfg.rebal_divisor}_ov{cfg.overlap_method}"
        f"_bl{cfg.block_length_scheme}_emb{cfg.embargo}"
        f"_pit{cfg.pit_policy}_ma{cfg.min_articles}"
        f"_s{cfg.seed}_nb{cfg.n_boot}"
    )


def trailing_schedule(
    n_obs: int, window: int, rebal_divisor: int = 4
) -> WindowSchedule:
    """Generate a fixed trailing-window rolling schedule.

    Reproduces ``pipeline.stages.papers.paper3.empirical.backtest`` exactly:
    ``rebal = max(1, window // rebal_divisor)``; one eval step for every
    ``t in range(window, n_obs)`` with estimation slice ``R[t-window:t]``,
    evaluated at ``R[t]``, rebalancing when ``(t - window) % rebal == 0``.
    """
    rebal = max(1, window // rebal_divisor)
    entries = tuple(
        ScheduleEntry(
            train_start=t - window,
            train_stop=t,
            eval_index=t,
            rebalance=((t - window) % rebal == 0),
        )
        for t in range(window, n_obs)
    )
    return WindowSchedule(
        window_type="trailing", window=window, rebal=rebal, entries=entries
    )


def anchored_schedule(
    n_obs: int, warmup: int, rebal_divisor: int = 4
) -> WindowSchedule:
    """Expanding-origin (anchored) schedule.

    Same eval cadence and rebalance-flag rule as :func:`trailing_schedule`
    (with ``window`` replaced by ``warmup``), but the training slice always
    starts at index 0: ``R[0:t]`` is the estimation slice at eval index
    ``t``, so the estimation window grows with each step.
    """
    rebal = max(1, warmup // rebal_divisor)
    entries = tuple(
        ScheduleEntry(
            train_start=0,
            train_stop=t,
            eval_index=t,
            rebalance=((t - warmup) % rebal == 0),
        )
        for t in range(warmup, n_obs)
    )
    return WindowSchedule(
        window_type="anchored", window=warmup, rebal=rebal, entries=entries
    )


def disjoint_epoch_schedule(
    epoch_start: int, epoch_stop: int, window: int, rebal_divisor: int = 4
) -> WindowSchedule:
    """Trailing rolling schedule confined to a sub-epoch ``[epoch_start, epoch_stop)``.

    Matches the former energy-robustness design: rolling estimation within a fixed
    evaluation epoch, calibration inputs otherwise held fixed. Indices are
    absolute into the full panel; equivalent to
    ``trailing_schedule(epoch_stop, window)`` restricted to
    ``eval_index in range(epoch_start + window, epoch_stop)``.
    """
    rebal = max(1, window // rebal_divisor)
    entries = tuple(
        ScheduleEntry(
            train_start=t - window,
            train_stop=t,
            eval_index=t,
            rebalance=((t - window) % rebal == 0),
        )
        for t in range(epoch_start + window, epoch_stop)
    )
    return WindowSchedule(
        window_type="disjoint", window=window, rebal=rebal, entries=entries
    )


def _all_finite(values: object) -> bool:
    """Return whether every scalar in an eager host-array value is finite."""
    try:
        return math.isfinite(float(typing.cast("float", values)))
    except (TypeError, ValueError):
        pass

    try:
        magnitude = abs(typing.cast("float", values))
        return math.isfinite(float(magnitude))
    except (TypeError, ValueError):
        pass

    try:
        iterable = typing.cast("typing.Iterable[object]", values)
        return all(_all_finite(value) for value in iterable)
    except TypeError:
        return False


def pit_guard[T](
    vintage_matrix: T | None,
    *,
    threshold_ok: bool,
    fallback_matrix: T,
) -> T:
    """Generalized F4 point-in-time fallback guard.

    Origin: ``pipeline.stages.papers.paper3.empirical.backtest`` (F4 guard,
    ``d2_use = D2 if pit_d2 is None or not all_finite(pit_d2) else pit_d2``).
    Thin early-vintage cells can be missing or contain non-finite entries (too
    few articles for a trustworthy pairwise distance); rather than propagate
    ``NaN`` into downstream estimators, fall back to a static baseline matrix.
    Extended here with an explicit ``threshold_ok`` flag (e.g. a
    minimum-article-count check) so callers can gate on richer PIT-availability
    policies than finiteness alone.
    """
    if vintage_matrix is None:
        return fallback_matrix
    if not _all_finite(vintage_matrix):
        return fallback_matrix
    if not threshold_ok:
        return fallback_matrix
    return vintage_matrix
