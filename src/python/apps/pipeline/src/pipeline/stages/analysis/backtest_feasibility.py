"""Backtest minimal-feasible-window feasibility assessment (additive analysis).

Answers the author's #1 backtest-workstream question — *what is the minimal
feasible estimation window given the data we already have* — as a NEW,
reproducible, light-compute analysis that changes NO published number. It
reproduces and extends the 2026-07-18 news-density constructibility cliff
(``.context/chat/2026-07-18_news-density-backtest-feasibility``) from the
non-overlapping-block proxy used there to TRUE trailing / expanding windows at
the actual rebalance dates the papers would trade on, and adds the
shrinkage-adjusted RETURNS view alongside the NEWS view.

For each ``window_type in {rolling, expanding}`` and each estimation window
``T`` (trading days) on a defensible grid, at every rebalance date it counts,
per asset:

* RETURNS: in-window available return observations, benchmarked against the
  covariance-conditioning thresholds ``T > N`` (singularity),
  ``T >= 5N`` (conditioning-comfortable) and ``T >= 10N`` (ample), with a
  ``shrinkage_required`` flag for any ``T < 5N`` cell.
* NEWS: in-window per-ticker article counts from the raw un-embedded corpus,
  benchmarked against the ``>= 30`` (usable two-sample energy estimate) and
  ``>= 100`` (stability, near the ~128-article embedding design point)
  engineering thresholds.

The binding constraint is cross-sectional: the papers need the FULL
``n x n`` distance/covariance object at each rebalance, so a period is
*fully constructible* for a channel only when EVERY asset clears the
threshold. The headline output is the fraction of fully-constructible periods
per ``(window_type, T, channel, threshold)`` — the constructibility cliff.

LITERATURE ANCHORS (decision brief 2026-07-18; references not yet staged in
any paper bibliography — flagged for author acquisition, do NOT @cite yet):

* Returns floor / default (D2): 252d floor and 504d default WITH shrinkage;
  their T/N ratios and the first all-asset clearing windows are derived from
  the configured universe and realized panel. Ledoit-Wolf
  (2003/2004a/2004b/2020); DeMiguel-Garlappi-Uppal (2009). Report BOTH rolling
  and expanding (Giacomini-White 2006; Rossi 2013).
* News floor (D3, a-priori prior): article-count-bound, ~3-year (756d)
  trailing or cross-ticker pooling; per-ticker text stats on < ~100 in-window
  articles are unreliable. The >=30 / >=100 thresholds are ENGINEERING
  thresholds validated empirically here (no primary citation gives a
  min-articles rule). On this dense corpus the empirical cliff REFINES the
  news floor downward from the a-priori 3-year prior to the derived
  recommended_floor_T_rolling (>=30) / recommended_default_T_rolling (>=100);
  see the news recommendation, not this prior, for the data-driven answer.

This is deliberately light compute (counts / aggregations over the configured
return panel and article corpus); the heavy Monte-Carlo size/power
justification of the final ``T`` (brief D6) is a separate harness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import ScalarFormatter
from numpy.typing import NDArray

from pipeline.figures._common import _C, _save, _style

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ReturnsThreshold(NamedTuple):
    """A covariance-conditioning threshold on in-window return observations.

    Attributes:
        name: Short key (``singularity`` / ``comfortable`` / ``ample``).
        label: Human/paper label (e.g. ``"T>N"``).
        min_count: Minimum per-asset observation count to clear the threshold
            (a cell clears when ``count >= min_count``).

    """

    name: str
    label: str
    min_count: int


class NewsThreshold(NamedTuple):
    """An article-density threshold on in-window per-ticker article counts.

    Attributes:
        name: Short key (``usable`` / ``stability``).
        label: Human label (e.g. ``"articles>=30"``).
        min_count: Minimum per-ticker article count to clear the threshold.

    """

    name: str
    label: str
    min_count: int


class FeasibilityConfig(NamedTuple):
    """Fully-hashable description of one feasibility assessment.

    Attributes:
        window_types: Window schemes to sweep (``rolling`` and/or
            ``expanding``).
        windows: Estimation-window lengths ``T`` (trading days).
        n_assets_nominal: Nominal universe size ``N`` used for the returns
            conditioning thresholds. The realized return-panel dimension is
            reported separately so any mismatch remains visible.
        returns_thresholds: Conditioning thresholds for the returns channel.
        news_thresholds: Density thresholds for the news channel.

    """

    window_types: tuple[str, ...]
    windows: tuple[int, ...]
    n_assets_nominal: int
    returns_thresholds: tuple[ReturnsThreshold, ...]
    news_thresholds: tuple[NewsThreshold, ...]


def _config_from_params(params: dict[str, object]) -> FeasibilityConfig:
    """Build a :class:`FeasibilityConfig` from the ``backtest_feasibility`` block."""
    block = _as_dict(params.get("backtest_feasibility", {}))

    window_types = tuple(
        str(w) for w in _as_list(block.get("window_types", ["rolling", "expanding"]))
    )
    windows = tuple(
        int(_as_scalar(t))
        for t in _as_list(block.get("windows", [63, 126, 252, 504, 756, 1260]))
    )
    n_assets = int(_as_scalar(block.get("n_assets_nominal", 100)))

    mult = _as_dict(block.get("returns_conditioning_multiples", {}))
    sing = int(_as_scalar(mult.get("singularity", 1)))
    comf = int(_as_scalar(mult.get("comfortable", 5)))
    ample = int(_as_scalar(mult.get("ample", 10)))
    returns_thresholds = (
        # T > N  <=>  count >= N + 1 (strict rank sufficiency for an N x N cov).
        ReturnsThreshold("singularity", f"T>{sing}N", sing * n_assets + 1),
        ReturnsThreshold("comfortable", f"T>={comf}N", comf * n_assets),
        ReturnsThreshold("ample", f"T>={ample}N", ample * n_assets),
    )

    news_counts = [
        int(_as_scalar(c)) for c in _as_list(block.get("news_thresholds", [30, 100]))
    ]
    names = ["usable", "stability", "extra"]
    news_thresholds = tuple(
        NewsThreshold(names[i] if i < len(names) else f"news{c}", f"articles>={c}", c)
        for i, c in enumerate(news_counts)
    )

    return FeasibilityConfig(
        window_types=window_types,
        windows=windows,
        n_assets_nominal=n_assets,
        returns_thresholds=returns_thresholds,
        news_thresholds=news_thresholds,
    )


def _as_list(obj: object) -> list[object]:
    """Coerce a params value to a list (single scalars become one-element lists)."""
    if isinstance(obj, list):
        return list(obj)
    return [obj]


def _as_dict(obj: object) -> dict[str, object]:
    """Coerce a params value to a ``dict[str, object]`` (empty for non-mappings)."""
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {}


def _as_scalar(obj: object) -> float:
    """Coerce a params value to a float scalar."""
    if isinstance(obj, (int, float)):
        return float(obj)
    return float(str(obj))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class ReturnCalendar(NamedTuple):
    """The aligned trading calendar and per-asset availability mask.

    Attributes:
        tickers: The ``n`` traded symbols (returns panel), sorted.
        dates: The ``(T_cal,)`` aligned trading-day ``DatetimeIndex``.
        finite: ``(T_cal, n)`` boolean mask, ``True`` where a return is
            observed (finite) for that (day, asset).

    """

    tickers: list[str]
    dates: pd.DatetimeIndex
    finite: BoolArray


def load_return_calendar(returns_dir: Path) -> ReturnCalendar:
    """Load the inner-joined return panel and its availability mask.

    Mirrors the returns-panel intersection used by the Paper-3 empirical
    driver (per-ticker ``<TICKER>.parquet`` with ``Date`` / ``return``),
    returning only the calendar and the per-asset finite mask needed for
    in-window observation counting.

    Args:
        returns_dir: Directory of per-ticker return parquets.

    Returns:
        A :class:`ReturnCalendar`.

    """
    tickers = sorted(p.stem for p in returns_dir.glob("*.parquet"))
    if not tickers:
        message = f"No return parquets found in {returns_dir}"
        raise FileNotFoundError(message)
    frames: list[pd.DataFrame] = []
    for t in tickers:
        df = pd.read_parquet(returns_dir / f"{t}.parquet")[["Date", "return"]]
        frames.append(df.rename(columns={"return": t}).set_index("Date"))
    panel = pd.concat(frames, axis=1).sort_index()
    dates = pd.DatetimeIndex(panel.index)
    finite: BoolArray = np.isfinite(panel.to_numpy(dtype=np.float64))
    return ReturnCalendar(tickers=tickers, dates=dates, finite=finite)


def load_article_dates(
    corpus_path: Path, tickers: list[str]
) -> dict[str, NDArray[np.datetime64]]:
    """Load sorted per-ticker article publication dates from the raw corpus.

    Args:
        corpus_path: Path to ``articles.parquet`` (columns ``symbol``,
            ``created_date``).
        tickers: Universe to restrict to (the traded symbols).

    Returns:
        Mapping ``ticker -> sorted np.datetime64[ns]`` array of article dates
        (empty array for a ticker with no articles).

    """
    df = pd.read_parquet(corpus_path, columns=["symbol", "created_date"])
    df = df[df["symbol"].isin(tickers)]
    created = pd.to_datetime(df["created_date"]).to_numpy(dtype="datetime64[ns]")
    symbols = df["symbol"].to_numpy()
    out: dict[str, NDArray[np.datetime64]] = {}
    for t in tickers:
        arr = np.sort(created[symbols == t])
        out[t] = arr
    return out


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


class RebalanceWindow(NamedTuple):
    """One rebalance window on the trading calendar.

    Attributes:
        train_start: Inclusive start index into the calendar of the estimation
            window (``t - T`` for rolling, ``0`` for expanding).
        rebal_index: The rebalance-date index ``t`` (exclusive end of the
            estimation window; weights are applied at/after this date).

    """

    train_start: int
    rebal_index: int


def generate_schedule(
    n_cal: int, window_type: str, window: int
) -> list[RebalanceWindow]:
    """Rebalance schedule reproducing the papers' trailing / anchored cadence.

    Rebalances every ``max(1, T // 4)`` trading days starting at ``t = T``
    (matching ``empirical.py`` / ``robust.py``'s ``rebal = max(1, win // 4)``
    and the ``for t in range(win, T)`` loop). ``rolling`` uses a fixed-width
    trailing window ``[t - T, t)``; ``expanding`` anchors the start at ``0``
    (all history up to ``t``), so ``T`` acts as the warm-up length. A
    ``rolling`` window with ``T >= n_cal`` yields an EMPTY schedule (the
    data-length ceiling: the panel cannot form a single such window).

    Args:
        n_cal: Number of trading days in the calendar.
        window_type: ``"rolling"`` or ``"expanding"``.
        window: Estimation-window length ``T`` (trading days).

    Returns:
        List of :class:`RebalanceWindow` (possibly empty).

    """
    if window_type not in ("rolling", "expanding"):
        message = f"window_type must be rolling/expanding, got {window_type!r}"
        raise ValueError(message)
    if window < 1:
        message = f"window must be >= 1, got {window}"
        raise ValueError(message)
    rebal = max(1, window // 4)
    out: list[RebalanceWindow] = []
    for t in range(window, n_cal):
        if (t - window) % rebal != 0:
            continue
        train_start = t - window if window_type == "rolling" else 0
        out.append(RebalanceWindow(train_start=train_start, rebal_index=t))
    return out


# ---------------------------------------------------------------------------
# Per-window counting
# ---------------------------------------------------------------------------


def count_returns_in_window(finite: BoolArray, win: RebalanceWindow) -> IntArray:
    """Per-asset count of available return observations in the window."""
    block = finite[win.train_start : win.rebal_index]
    return block.sum(axis=0).astype(np.int64)


def count_articles_in_window(
    article_dates: dict[str, NDArray[np.datetime64]],
    tickers: list[str],
    start: np.datetime64,
    end: np.datetime64,
) -> IntArray:
    """Per-ticker count of articles in the half-open calendar span ``[start, end)``."""
    counts = np.empty(len(tickers), dtype=np.int64)
    for i, t in enumerate(tickers):
        arr = article_dates[t]
        lo = int(np.searchsorted(arr, start, side="left"))
        hi = int(np.searchsorted(arr, end, side="left"))
        counts[i] = hi - lo
    return counts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class ChannelSummary(NamedTuple):
    """Aggregated constructibility statistics for one channel/threshold cell."""

    n_periods: int
    frac_fully_constructible: float
    mean_frac_assets_clearing: float
    median_worst_count: float
    p10_worst_count: float


def _summarize(counts_by_period: list[IntArray], min_count: int) -> ChannelSummary:
    """Aggregate per-period per-asset counts against a clearing threshold."""
    n_periods = len(counts_by_period)
    if n_periods == 0:
        return ChannelSummary(0, float("nan"), float("nan"), float("nan"), float("nan"))
    worst = np.array([int(c.min()) for c in counts_by_period], dtype=np.int64)
    frac_clear = np.array(
        [float((c >= min_count).mean()) for c in counts_by_period], dtype=np.float64
    )
    all_clear = worst >= min_count
    return ChannelSummary(
        n_periods=n_periods,
        frac_fully_constructible=float(all_clear.mean()),
        mean_frac_assets_clearing=float(frac_clear.mean()),
        median_worst_count=float(np.median(worst)),
        p10_worst_count=float(np.percentile(worst, 10)),
    )


def _binding_tickers(
    counts_by_period: list[IntArray],
    tickers: list[str],
    min_count: int,
    top_k: int = 8,
) -> list[dict[str, object]]:
    """Identify the tickers that most often fail (are the binding constraint)."""
    if not counts_by_period:
        return []
    stacked = np.vstack(counts_by_period)  # (n_periods, n_assets)
    below = (stacked < min_count).sum(axis=0).astype(np.int64)
    median_count = np.median(stacked, axis=0)
    order = np.argsort(-below)  # most failures first
    rows: list[dict[str, object]] = []
    for j in order[:top_k]:
        if int(below[j]) == 0:
            continue
        rows.append(
            {
                "ticker": tickers[j],
                "n_periods_below": int(below[j]),
                "frac_periods_below": float(below[j] / stacked.shape[0]),
                "median_count": float(median_count[j]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def assess_feasibility(
    calendar: ReturnCalendar,
    article_dates: dict[str, NDArray[np.datetime64]],
    config: FeasibilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the constructibility table and binding-ticker table.

    Args:
        calendar: The trading calendar + return availability mask.
        article_dates: Per-ticker sorted article dates.
        config: The feasibility configuration.

    Returns:
        ``(constructibility, binding)`` DataFrames. ``constructibility`` is
        long-format, one row per ``(window_type, T, channel, threshold)``;
        ``binding`` lists the thinnest tickers per ``(window_type, T,
        channel)`` at the primary usable threshold.

    """
    tickers = calendar.tickers
    dates = calendar.dates
    n_cal = len(dates)
    n_assets = len(tickers)

    con_rows: list[dict[str, object]] = []
    bind_rows: list[dict[str, object]] = []

    for wtype in config.window_types:
        for window in config.windows:
            schedule = generate_schedule(n_cal, wtype, window)

            # Per-window per-asset counts for each channel (computed once).
            ret_counts: list[IntArray] = []
            news_counts: list[IntArray] = []
            span_days: list[int] = []
            for win in schedule:
                ret_counts.append(count_returns_in_window(calendar.finite, win))
                start = np.datetime64(dates[win.train_start], "ns")
                end = np.datetime64(dates[win.rebal_index], "ns")
                news_counts.append(
                    count_articles_in_window(article_dates, tickers, start, end)
                )
                span_days.append(
                    int((dates[win.rebal_index] - dates[win.train_start]).days)
                )

            first_date = str(dates[schedule[0].rebal_index].date()) if schedule else ""
            last_date = str(dates[schedule[-1].rebal_index].date()) if schedule else ""
            median_span = float(np.median(span_days)) if span_days else float("nan")

            # RETURNS channel.
            for thr in config.returns_thresholds:
                s = _summarize(ret_counts, thr.min_count)
                con_rows.append(
                    {
                        "window_type": wtype,
                        "window": window,
                        "channel": "returns",
                        "threshold_name": thr.name,
                        "threshold_label": thr.label,
                        "threshold_count": thr.min_count,
                        "n_assets": n_assets,
                        "n_assets_nominal": config.n_assets_nominal,
                        "n_periods": s.n_periods,
                        "frac_fully_constructible": s.frac_fully_constructible,
                        "mean_frac_assets_clearing": s.mean_frac_assets_clearing,
                        "median_worst_count": s.median_worst_count,
                        "p10_worst_count": s.p10_worst_count,
                        "shrinkage_required": (5 * config.n_assets_nominal > window),
                        "first_rebalance": first_date,
                        "last_rebalance": last_date,
                        "median_window_calendar_days": median_span,
                    }
                )

            # NEWS channel.
            for nthr in config.news_thresholds:
                s = _summarize(news_counts, nthr.min_count)
                con_rows.append(
                    {
                        "window_type": wtype,
                        "window": window,
                        "channel": "news",
                        "threshold_name": nthr.name,
                        "threshold_label": nthr.label,
                        "threshold_count": nthr.min_count,
                        "n_assets": n_assets,
                        "n_assets_nominal": config.n_assets_nominal,
                        "n_periods": s.n_periods,
                        "frac_fully_constructible": s.frac_fully_constructible,
                        "mean_frac_assets_clearing": s.mean_frac_assets_clearing,
                        "median_worst_count": s.median_worst_count,
                        "p10_worst_count": s.p10_worst_count,
                        "shrinkage_required": False,
                        "first_rebalance": first_date,
                        "last_rebalance": last_date,
                        "median_window_calendar_days": median_span,
                    }
                )

            # Binding tickers at the primary usable thresholds.
            if config.returns_thresholds:
                bind_rows.extend(
                    {
                        "window_type": wtype,
                        "window": window,
                        "channel": "returns",
                        **row,
                    }
                    for row in _binding_tickers(
                        ret_counts, tickers, config.returns_thresholds[0].min_count
                    )
                )
            if config.news_thresholds:
                bind_rows.extend(
                    {
                        "window_type": wtype,
                        "window": window,
                        "channel": "news",
                        **row,
                    }
                    for row in _binding_tickers(
                        news_counts, tickers, config.news_thresholds[0].min_count
                    )
                )

    return pd.DataFrame(con_rows), pd.DataFrame(bind_rows)


def _recommendation(con: pd.DataFrame, config: FeasibilityConfig) -> dict[str, object]:
    """Derive the minimal-feasible-T recommendation per channel from the table."""

    def _min_feasible(
        channel: str, thr_name: str, wtype: str, level: float
    ) -> int | None:
        sub = con[
            (con["channel"] == channel)
            & (con["threshold_name"] == thr_name)
            & (con["window_type"] == wtype)
            & (con["n_periods"] > 0)
        ].sort_values("window")
        ok = sub[sub["frac_fully_constructible"] >= level]
        return int(ok["window"].iloc[0]) if len(ok) else None

    def _cell_value(
        channel: str, thr_name: str, wtype: str, window: int, column: str
    ) -> float | None:
        sub = con[
            (con["channel"] == channel)
            & (con["threshold_name"] == thr_name)
            & (con["window_type"] == wtype)
            & (con["window"] == window)
        ]
        if sub.empty or pd.isna(sub[column].iloc[0]):
            return None
        return float(sub[column].iloc[0])

    def _fmt_window(window: int | None) -> str:
        return f"{window}d" if window is not None else "not reached on the T-grid"

    def _fmt_fraction(fraction: float | None) -> str:
        return f"{fraction:.1%}" if fraction is not None else "not evaluated"

    def _fmt_threshold(threshold: int | None) -> str:
        return f">={threshold}" if threshold is not None else "unconfigured"

    n = config.n_assets_nominal
    if n < 1:
        message = f"n_assets_nominal must be positive, got {n}"
        raise ValueError(message)

    n_traded = int(con["n_assets"].iloc[0]) if len(con) else 0
    returns_floor = 252
    returns_default = 504
    returns_singularity = _min_feasible("returns", "singularity", "rolling", 1.0)
    returns_comfortable = _min_feasible("returns", "comfortable", "rolling", 1.0)
    default_comfortable_fraction = _cell_value(
        "returns",
        "comfortable",
        "rolling",
        returns_default,
        "frac_fully_constructible",
    )
    news_usable_rolling = _min_feasible("news", "usable", "rolling", 1.0)
    news_usable_expanding = _min_feasible("news", "usable", "expanding", 1.0)
    news_stability_rolling = _min_feasible("news", "stability", "rolling", 1.0)
    usable_threshold = (
        config.news_thresholds[0].min_count if config.news_thresholds else None
    )
    stability_threshold = (
        config.news_thresholds[-1].min_count if config.news_thresholds else None
    )
    joint_requirements = (returns_comfortable, news_stability_rolling)
    joint_candidates = [window for window in joint_requirements if window is not None]
    joint_window = (
        max(joint_candidates)
        if len(joint_candidates) == len(joint_requirements)
        else None
    )
    joint_periods_value = (
        _cell_value("returns", "comfortable", "rolling", joint_window, "n_periods")
        if joint_window is not None
        else None
    )
    joint_periods = (
        int(joint_periods_value) if joint_periods_value is not None else None
    )
    universe_note = (
        f"Returns and news feasibility use the same realized {n_traded}-asset "
        f"panel; conditioning thresholds use configured N={n}."
        if n_traded == n
        else (
            f"The realized panel contains {n_traded} assets while conditioning "
            f"thresholds use configured nominal N={n}; both counts are reported."
        )
    )

    return {
        "universe": {
            "n_assets_traded": n_traded,
            "n_assets_nominal": n,
            "note": universe_note,
        },
        "returns": {
            "singularity_min_T": n + 1,
            "conditioning_comfortable_min_T": 5 * n,
            "ample_min_T": 10 * n,
            "recommended_floor_T": returns_floor,
            "recommended_floor_T_over_N": round(returns_floor / n, 4),
            "recommended_default_T_rolling": returns_default,
            "recommended_default_T_over_N": round(returns_default / n, 4),
            "rationale": (
                f"{returns_floor}d floor (T/N={returns_floor / n:.2f}, shrinkage "
                f"REQUIRED under the configured T<5N rule); {returns_default}d "
                f"default (T/N={returns_default / n:.2f}). In the realized "
                f"{n_traded}-asset panel, the first rolling grid window where "
                "every rebalance period meets the >=5N observation threshold is "
                f"{_fmt_window(returns_comfortable)}; at the "
                f"{returns_default}d default, "
                f"{_fmt_fraction(default_comfortable_fraction)} of rolling "
                "periods are fully constructible. Brief D1/D2 "
                "(Ledoit-Wolf shrinkage; DeMiguel-Garlappi-Uppal 2009). Report "
                "rolling primary + expanding robustness (Giacomini-White 2006; "
                "Rossi 2013)."
            ),
            "first_T_all_periods_singularity_rolling": returns_singularity,
            "first_T_all_periods_conditioning_comfortable_rolling": (
                returns_comfortable
            ),
        },
        "news": {
            "usable_threshold": usable_threshold,
            "stability_threshold": stability_threshold,
            "first_T_all_assets_usable_rolling": news_usable_rolling,
            "first_T_all_assets_usable_expanding": news_usable_expanding,
            "first_T_all_assets_stability_rolling": news_stability_rolling,
            "recommended_floor_T_rolling": news_usable_rolling,
            "recommended_default_T_rolling": news_stability_rolling,
            "rationale": (
                "Article-count-bound and DATA-DERIVED (not the a-priori "
                "3-year prior): the binding constraint is the thinnest ticker "
                f"per window. On this {n_traded}-asset panel every ticker first "
                f"clears the {_fmt_threshold(usable_threshold)} usable bar at "
                f"{_fmt_window(news_usable_rolling)} and the "
                f"{_fmt_threshold(stability_threshold)} stability bar (near the "
                "~128-article embedding design point) at "
                f"{_fmt_window(news_stability_rolling)} "
                "in rolling windows. These are the first fully constructible "
                "windows on the configured grid. This REFINES Brief D3's "
                "conservative >=3-year / cross-ticker-pooling prior downward "
                "for this dense corpus; pooling/shrinkage and expanding windows "
                "extend feasibility to still-shorter windows. Thresholds are "
                "engineering bars validated here (no primary min-articles "
                "citation)."
            ),
        },
        "scissors": (
            "Feasibility improves jointly in T, but the two channels reach "
            "constructibility at very different windows relative to the "
            "returns-backtest horizon: all rolling periods first clear T>N at "
            f"{_fmt_window(returns_singularity)} and >=5N at "
            f"{_fmt_window(returns_comfortable)}; all assets first clear the "
            f"{_fmt_threshold(usable_threshold)} and "
            f"{_fmt_threshold(stability_threshold)} news bars at "
            f"{_fmt_window(news_usable_rolling)} and "
            f"{_fmt_window(news_stability_rolling)}, respectively. The first "
            "configured rolling window satisfying both the >=5N returns and "
            f"stable-news criteria is {_fmt_window(joint_window)}, leaving "
            f"{joint_periods if joint_periods is not None else 'no'} scheduled "
            "rebalance periods. Short return-backtest windows therefore cannot "
            "support a fully constructible per-window text object, which "
            "supports the static full-sample text matrix rather than treating it "
            "as a shortcut. (Reproduces+extends the 2026-07-18 constructibility "
            "cliff; the joint-window trade-off is flagged for author review.)"
        ),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_constructibility_cliff(con: pd.DataFrame, out_path: Path) -> None:
    """Render the constructibility cliff: fraction fully-constructible vs T."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    styles = {
        ("returns", "singularity"): (_C["blue"], "-", "returns T>N"),
        ("returns", "comfortable"): (_C["sky"], "--", "returns T>=5N"),
        ("news", "usable"): (_C["vermilion"], "-", "news >=30"),
        ("news", "stability"): (_C["orange"], "--", "news >=100"),
    }
    for ax, wtype in zip(axes, ("rolling", "expanding"), strict=False):
        for (channel, thr), (color, ls, label) in styles.items():
            sub = con[
                (con["window_type"] == wtype)
                & (con["channel"] == channel)
                & (con["threshold_name"] == thr)
                & (con["n_periods"] > 0)
            ].sort_values("window")
            if sub.empty:
                continue
            ax.plot(
                sub["window"].to_numpy(),
                sub["frac_fully_constructible"].to_numpy(),
                marker="o",
                markersize=3.5,
                color=color,
                linestyle=ls,
                label=label,
            )
        ax.set_title(f"{wtype} window")
        ax.set_xlabel("estimation window $T$ (trading days)")
        ax.set_xscale("log")
        ax.set_xticks(sorted(con["window"].unique().tolist()))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.axhline(1.0, color=_C["gray"], lw=0.6, ls=":")
        ax.grid(visible=True, alpha=0.25, lw=0.4)
    axes[0].set_ylabel("fraction of fully-\nconstructible periods")
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Backtest constructibility cliff: all-asset threshold per rebalance window",
        fontsize=10,
    )
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_backtest_feasibility(
    returns_dir: Path,
    corpus_path: Path,
    params_file: Path,
    output_dir: Path,
) -> None:
    """Run the backtest minimal-feasible-window feasibility assessment.

    Writes ``constructibility.parquet`` (the long-format table),
    ``binding_tickers.parquet`` (thinnest tickers per cell),
    ``summary.yaml`` (the minimal-feasible-T recommendation per channel), and
    ``constructibility_cliff.pgf`` (the figure) into ``output_dir``.

    Args:
        returns_dir: Directory of per-ticker return parquets.
        corpus_path: Path to the raw ``articles.parquet`` corpus.
        params_file: ``params.yaml`` (reads the ``backtest_feasibility`` block).
        output_dir: Output directory.

    """
    with params_file.open(encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    config = _config_from_params(params if isinstance(params, dict) else {})

    logger.info(
        "backtest-feasibility: %d window-types x %d windows over returns=%s corpus=%s",
        len(config.window_types),
        len(config.windows),
        returns_dir,
        corpus_path,
    )
    calendar = load_return_calendar(returns_dir)
    article_dates = load_article_dates(corpus_path, calendar.tickers)
    logger.info(
        "loaded %d traded tickers, %d trading days (%s -> %s), %d articles",
        len(calendar.tickers),
        len(calendar.dates),
        calendar.dates[0].date(),
        calendar.dates[-1].date(),
        sum(len(v) for v in article_dates.values()),
    )

    con, binding = assess_feasibility(calendar, article_dates, config)
    recommendation = _recommendation(con, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    con.to_parquet(output_dir / "constructibility.parquet", index=False)
    binding.to_parquet(output_dir / "binding_tickers.parquet", index=False)
    plot_constructibility_cliff(con, output_dir / "constructibility_cliff.pgf")

    summary: dict[str, object] = {
        "n_assets_traded": len(calendar.tickers),
        "n_assets_nominal": config.n_assets_nominal,
        "n_trading_days": len(calendar.dates),
        "calendar_start": str(calendar.dates[0].date()),
        "calendar_end": str(calendar.dates[-1].date()),
        "n_articles": int(sum(len(v) for v in article_dates.values())),
        "window_types": list(config.window_types),
        "windows": list(config.windows),
        "recommendation": recommendation,
        "cliff_rolling_news_usable": _cliff_slice(con, "rolling", "news", "usable"),
        "cliff_rolling_returns_comfortable": _cliff_slice(
            con, "rolling", "returns", "comfortable"
        ),
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    logger.info("backtest-feasibility: wrote outputs to %s", output_dir)


def _cliff_slice(
    con: pd.DataFrame, wtype: str, channel: str, thr_name: str
) -> dict[int, object]:
    """Compact ``{T: fraction-fully-constructible}`` slice for the summary."""
    sub = con[
        (con["window_type"] == wtype)
        & (con["channel"] == channel)
        & (con["threshold_name"] == thr_name)
    ].sort_values("window")
    out: dict[int, object] = {}
    for _, r in sub.iterrows():
        val = r["frac_fully_constructible"]
        out[int(r["window"])] = None if pd.isna(val) else round(float(val), 4)
    return out
