"""Permutation-test internals — private sub-package of stage S7.

Created private from the outset: :mod:`~jcor.inference._permutation.common`
(113 lines) and :mod:`~jcor.inference._permutation.split_runner` (220 lines)
would push :mod:`jcor.inference.permutation` past the 250-line view knee if
concatenated, so t46.7 keeps them as siblings here instead. They also carry
deliberately generic names (``common``, ``split_runner``) that must not sit at
stage level, where a second migration would collide with them.

Nothing outside ``jcor.inference`` may import this package.
"""

from __future__ import annotations

__all__: list[str] = []
