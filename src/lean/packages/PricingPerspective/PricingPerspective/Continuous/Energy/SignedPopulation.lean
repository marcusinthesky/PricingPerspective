import PricingPerspective.Continuous.CovarianceSpace.Interchange
import PricingPerspective.Continuous.Energy.Population

/-!
# Signed-exposure population correlation bound

This module composes the signed-measure residual/field interchange with the
abstract population correlation theorem.  It is the end-to-end formal version
of Paper 1, Corollary 2 at the measure-theoretic boundary.
-/

namespace PricingPerspective.ContinuousAPT

open MeasureTheory CovarianceSpace
open scoped InnerProductSpace

variable {Ω E : Type*} [MeasurableSpace Ω]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/--
**G2 (Signed-Exposure Total-Return Correlation Lower Bound).**

Let systematic exposures be signed field integrals.  Jordan-total-variation
integrability licenses moving each residual inner-product functional through
the integral.  Pointwise residual/field orthogonality therefore supplies all
four systematic/residual orthogonality hypotheses of
`total_return_correlation_lower_bound'`; only cross-residual orthogonality is
left as a separate assumption.
-/
theorem total_return_correlation_lower_bound_of_signed_exposures
    (F : Ω → E) (β_i β_j : AdmissibleExposure F) (ε_i ε_j : E)
    (L D_E : ℝ)
    (hF_i : Integrable F (β_i.val.1 : SignedMeasure Ω).totalVariation)
    (hF_j : Integrable F (β_j.val.1 : SignedMeasure Ω).totalVariation)
    (hL : 0 ≤ L) (hD : 0 ≤ D_E)
    (hLip : ‖fieldIntegral F β_i - fieldIntegral F β_j‖ ≤ L * D_E)
    (hβ_i : fieldIntegral F β_i ≠ 0) (hβ_j : fieldIntegral F β_j ≠ 0)
    (hε_i : ∀ ω, inner ℝ ε_i (F ω) = 0)
    (hε_j : ∀ ω, inner ℝ ε_j (F ω) = 0)
    (hee : inner ℝ ε_i ε_j = 0) :
    Real.sqrt
          ((‖fieldIntegral F β_i‖ ^ 2 / ‖fieldIntegral F β_i + ε_i‖ ^ 2) *
            (‖fieldIntegral F β_j‖ ^ 2 / ‖fieldIntegral F β_j + ε_j‖ ^ 2)) *
        ((‖fieldIntegral F β_i‖ ^ 2 + ‖fieldIntegral F β_j‖ ^ 2 -
              (L * D_E) ^ 2) /
          (2 * ‖fieldIntegral F β_i‖ * ‖fieldIntegral F β_j‖)) ≤
      inner ℝ (fieldIntegral F β_i + ε_i) (fieldIntegral F β_j + ε_j) /
        (‖fieldIntegral F β_i + ε_i‖ * ‖fieldIntegral F β_j + ε_j‖) := by
  have hii : inner ℝ (fieldIntegral F β_i) ε_i = 0 := by
    rw [real_inner_comm]
    exact residual_orthogonal_to_fieldIntegral F β_i hF_i ε_i hε_i
  have hjj : inner ℝ (fieldIntegral F β_j) ε_j = 0 := by
    rw [real_inner_comm]
    exact residual_orthogonal_to_fieldIntegral F β_j hF_j ε_j hε_j
  have hij : inner ℝ (fieldIntegral F β_i) ε_j = 0 := by
    rw [real_inner_comm]
    exact residual_orthogonal_to_fieldIntegral F β_i hF_i ε_j hε_j
  have hji : inner ℝ (fieldIntegral F β_j) ε_i = 0 := by
    rw [real_inner_comm]
    exact residual_orthogonal_to_fieldIntegral F β_j hF_j ε_i hε_i
  exact total_return_correlation_lower_bound'
    (fieldIntegral F β_i) (fieldIntegral F β_j) ε_i ε_j L D_E
    hL hD hLip hβ_i hβ_j hii hjj hij hji hee

end PricingPerspective.ContinuousAPT
