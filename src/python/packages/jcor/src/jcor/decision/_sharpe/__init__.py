"""Private step modules for the Sharpe decision layer (t46.8).

``jcor.decision.sharpe`` is the only supported import path -- it owns the
public surface (``sharpe_ratio``, the PSR/DSR family, the paired difference
test and the familywise multiple test). The modules here are an internal
organisation detail of that module; do NOT re-export them here, and do not
import them from outside ``sharpe.py``.

The split exists because the merged surface lands at ~490 lines, well past the
250-line knee; it follows the ``paper3/_empirical`` precedent rather than
leaving one oversized module for a later split.
"""
