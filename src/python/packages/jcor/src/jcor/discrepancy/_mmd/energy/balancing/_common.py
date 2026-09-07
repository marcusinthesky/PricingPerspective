"""Constants shared by the energy-kernel modules.

The shared substrate of the private MMD-energy balancing package: it imports no
sibling, so every other module may depend on it without creating a cycle.

``_MAX_DISTANCE_EXPONENT`` is deliberately **duplicated** in
:mod:`jcor.geometry._barycentre._common` rather than shared. Importing it there
would create a ``geometry -> discrepancy`` edge for a single float; the t46
stage fence permits that direction, but a one-constant dependency between two
stages is a worse trade than two literals that state the same closed interval.
"""

from __future__ import annotations

from typing import Final

#: Distance exponents live in the open interval ``(0, 2)``: at ``alpha = 2`` the
#: energy distance degenerates to the squared distance between means and loses
#: its characteristic property.
_MAX_DISTANCE_EXPONENT: Final = 2.0
