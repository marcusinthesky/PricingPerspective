import Architect
import «PricingPerspective»
import «EnergyStatistics»
import «WassersteinGeometry»

/-!
Publication export root. Project-owned annotations live beside their original declarations.
The external mathlib polarization identity is bound here because its source is upstream.
-/

attribute [blueprint "lem:polarization"
  (statement := /--
    For any $x,y$ in a real inner product space, $\langle x,y\rangle = \frac12(\lVert
    x\rVert^2+\lVert y\rVert^2-\lVert x-y\rVert^2)$.
  -/)]
  norm_sub_sq_real
