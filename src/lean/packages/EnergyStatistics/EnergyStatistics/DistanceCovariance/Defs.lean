import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Independence.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Matrix.Order
import Mathlib.LinearAlgebra.Matrix.Hadamard
import Mathlib.Analysis.SpecificLimits.Basic
import EnergyStatistics.Defs

set_option linter.style.longLine false

namespace EnergyStatistics


/-- The squared distance covariance.

    dCov²(X, Y) = E|X - X'||Y - Y'| + E|X - X'|·E|Y - Y'|
                - E|X - X'||Y - Y''| - E|X - X''||Y - Y'|

    where (X, Y), (X', Y'), (X'', Y'') are iid from the joint distribution π.

    This is the population distance covariance from Proposition 12.1,
    equivalently derived from the Brownian covariance representation
    in Theorem 15.1. The formula has three distinct terms:

    S₁ = E|X - X'||Y - Y'|         (cross-product of distances)
    S₂ = E|X - X'| · E|Y - Y'|     (product of marginal mean distances)
    S₃ = E|X - X'||Y - Y''|         (mixed term, coupling X-X' with Y-Y'')

    dCov²(X, Y) = S₁ + S₂ - 2·S₃

    Reference: Székely & Rizzo (2023), Proposition 12.1, eq. (12.7);
               Theorem 15.1, eq. (15.13). -/
noncomputable def distanceCovarianceSq
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : ℝ :=
  -- S₁: E|X - X'||Y - Y'| = ∫∫ dist(x,x')·dist(y,y') dπ(x,y) dπ(x',y')
  ∫ p, ∫ q, dist p.1 q.1 * dist p.2 q.2 ∂π.measure ∂π.measure
  -- S₂: E|X - X'| · E|Y - Y'|
  + (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)
    * (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure)
  -- -2·S₃: E|X - X'||Y - Y''| where (X,Y), (X',Y'), (X'',Y'') iid
  --   = ∫_p (∫_q dist(p₁,q₁) dπ(q)) · (∫_r dist(p₂,r₂) dπ(r)) dπ(p)
  - 2 * ∫ p, (∫ q, dist p.1 q.1 ∂π.measure)
             * (∫ r, dist p.2 r.2 ∂π.measure) ∂π.measure

/-- The squared distance variance of the first component.

    dVar²(X) = dCov²(X, X) = E|X-X'|² + (E|X-X'|)² - 2·E[|X-X'|·E|X-X''|]

    where X, X', X'' are iid from the first marginal of π.

    Reference: Székely & Rizzo (2023), Definition 12.2. -/
noncomputable def distanceVarianceSq
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : ℝ :=
  -- S₁: E|X - X'|²
  ∫ p, ∫ q, (dist p.1 q.1)^2 ∂π.measure ∂π.measure
  -- S₂: (E|X - X'|)²
  + (∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure)^2
  -- -2·S₃: E[|X-X'|·E|X-X''|] = ∫_p (E_{q}|X_p - X_q|)² dπ(p)
  - 2 * ∫ p, (∫ q, dist p.1 q.1 ∂π.measure)^2 ∂π.measure

/-- The squared distance variance of the second component.

    dVar²(Y) = dCov²(Y, Y)

    Reference: Székely & Rizzo (2023), Definition 12.2. -/
noncomputable def distanceVarianceSqSnd
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) : ℝ :=
  -- S₁: E|Y - Y'|²
  ∫ p, ∫ q, (dist p.2 q.2)^2 ∂π.measure ∂π.measure
  -- S₂: (E|Y - Y'|)²
  + (∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure)^2
  -- -2·S₃
  - 2 * ∫ p, (∫ q, dist p.2 q.2 ∂π.measure)^2 ∂π.measure

/-- Doubly-centered distance kernel for the first coordinate.

    The U-centered representation of distance covariance writes
    `dCov²(X,Y) = ∫ centeredDistFst π z · centeredDistSnd π z ∂(π⊗π)(z)`,
    which is nonneg when α has negative type (Lyons 2013 Thm 3.20).

    Reference: Székely & Rizzo (2023), Ch 12; Lyons (2013), §2. -/
noncomputable def centeredDistFst {α β : Type*}
    [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (z : (α × β) × (α × β)) : ℝ :=
  dist z.1.1 z.2.1
    - (∫ q, dist z.1.1 q.1 ∂π.measure) - (∫ q, dist z.2.1 q.1 ∂π.measure)
    + ∫ p, ∫ q, dist p.1 q.1 ∂π.measure ∂π.measure

/-- Doubly-centered distance kernel for the second coordinate.

    See `centeredDistFst` for the role in the U-centered representation.

    Reference: Székely & Rizzo (2023), Ch 12; Lyons (2013), §2. -/
noncomputable def centeredDistSnd {α β : Type*}
    [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β]
    (π : JointProbabilityMeasure α β) (z : (α × β) × (α × β)) : ℝ :=
  dist z.1.2 z.2.2
    - (∫ q, dist z.1.2 q.2 ∂π.measure) - (∫ q, dist z.2.2 q.2 ∂π.measure)
    + ∫ p, ∫ q, dist p.2 q.2 ∂π.measure ∂π.measure

end EnergyStatistics
