"""Compatibility facade for :mod:`jcor.inference.wolak`."""

from jcor.inference.wolak import (
    WolakResult,
    chi_bar_squared_sf,
    chi_bar_squared_weights_mc,
    wolak_statistic,
    wolak_test,
)

__all__ = [
    "WolakResult",
    "chi_bar_squared_sf",
    "chi_bar_squared_weights_mc",
    "wolak_statistic",
    "wolak_test",
]
