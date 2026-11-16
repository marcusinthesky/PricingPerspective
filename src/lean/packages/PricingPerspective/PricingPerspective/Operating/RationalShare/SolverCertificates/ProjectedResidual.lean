import PricingPerspective.Operating.VariationalInequality.ApproximateResponse
import PricingPerspective.Operating.RationalShare.FirmResponse.Response
import Mathlib.Tactic.NormNum

/-!
# Orthant stationarity residual certificates

The bound-constrained solver's stationarity residual is distinct from the natural projected-step
residual. At a zero coordinate it retains only a gradient component that points out of the
nonnegative orthant; at a positive coordinate it retains the full gradient. Its Euclidean norm
supplies the radial error in `ApproxSolvesVI` and hence a distance certificate.
-/

namespace PricingPerspective.Operating.RationalShare.SolverCertificates.ProjectedResidual

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.FirmResponse
  PricingPerspective.Operating.RationalShare.FirmResponse.Response

open scoped RealInnerProductSpace

variable {α : Type*} [Fintype α]

/-- Orthant stationarity residual at a feasible block.

At the boundary `x_a = 0`, a positive negative-gradient coordinate already satisfies the KKT
condition and is discarded. A negative coordinate is retained. At an interior coordinate the
full gradient is retained. Exact equality with zero is intentional: this is a real-arithmetic
definition, not a floating-point boundary heuristic. -/
noncomputable def stationarityResidual (F : Block α → Block α) (x : Block α) : Block α :=
  ofActivity fun a ↦ if x ((), a) = 0 then min (F x ((), a)) 0 else F x ((), a)

/-- Coordinate formula for the orthant stationarity residual. -/
@[simp] theorem stationarityResidual_apply (F : Block α → Block α) (x : Block α) (a : α) :
    stationarityResidual F x ((), a) =
      if x ((), a) = 0 then min (F x ((), a)) 0 else F x ((), a) := rfl

/-- Removing KKT-compatible boundary-gradient components can only lower the pairing with a
feasible displacement from `x`. -/
theorem stationarityResidual_inner_le (F : Block α → Block α) {x z : Block α}
    (hz : z ∈ orthant) :
    ⟪stationarityResidual F x, z - x⟫ ≤ ⟪F x, z - x⟫ := by
  rw [inner_eq_sum, inner_eq_sum, Fintype.sum_prod_type, Fintype.sum_prod_type]
  simp only [Fintype.sum_unique]
  apply Finset.sum_le_sum
  intro a _
  simp only [PiLp.sub_apply]
  change stationarityResidual F x ((), a) * (z ((), a) - x ((), a)) ≤
    F x ((), a) * (z ((), a) - x ((), a))
  rw [stationarityResidual_apply]
  by_cases hzero : x ((), a) = 0
  · rw [if_pos hzero, hzero, sub_zero]
    by_cases hgrad : F x ((), a) ≤ 0
    · rw [min_eq_left hgrad]
    · have hgrad' : 0 ≤ F x ((), a) := le_of_lt (lt_of_not_ge hgrad)
      rw [min_eq_right hgrad', zero_mul]
      exact mul_nonneg hgrad' (hz ((), a))
  · rw [if_neg hzero]

/-- A feasible block satisfies an approximate VI with error equal to the Euclidean norm of its
orthant stationarity residual. -/
theorem approxSolvesVI_of_stationarityResidual (F : Block α → Block α) {x : Block α}
    (hx : x ∈ orthant) :
    ApproxSolvesVI F orthant ‖stationarityResidual F x‖ x := by
  refine ⟨hx, fun z hz ↦ ?_⟩
  have hcauchy :=
    (abs_le.mp (abs_real_inner_le_norm (stationarityResidual F x) (z - x))).1
  simpa only [neg_mul] using hcauchy.trans (stationarityResidual_inner_le F hz)

/-- A componentwise stationarity tolerance bounds the Euclidean residual by the square root of
the number of activities times that tolerance. -/
theorem norm_stationarityResidual_le_sqrt_card_mul (F : Block α → Block α) (x : Block α)
    {tolerance : ℝ} (htolerance : 0 ≤ tolerance)
    (hcomponent : ∀ a, |stationarityResidual F x ((), a)| ≤ tolerance) :
    ‖stationarityResidual F x‖ ≤ Real.sqrt (Fintype.card α) * tolerance := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) htolerance)).mp
  calc
    ‖stationarityResidual F x‖ ^ 2 = ∑ p, stationarityResidual F x p ^ 2 :=
      norm_sq_eq_sum (stationarityResidual F x)
    _ = ∑ a, stationarityResidual F x ((), a) ^ 2 := by
      rw [Fintype.sum_prod_type]
      simp
    _ ≤ ∑ _a : α, tolerance ^ 2 := by
      apply Finset.sum_le_sum
      intro a _
      exact (sq_le_sq).2 (by simpa [abs_of_nonneg htolerance] using hcomponent a)
    _ = (Fintype.card α : ℝ) * tolerance ^ 2 := by simp
    _ = (Real.sqrt (Fintype.card α) * tolerance) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg _)]

/-- A componentwise infinity tolerance therefore supplies a radial approximate-VI error of
`sqrt(card α) * tolerance`. -/
theorem approxSolvesVI_of_componentwise_stationarityResidual (F : Block α → Block α)
    {x : Block α} {tolerance : ℝ} (hx : x ∈ orthant) (htolerance : 0 ≤ tolerance)
    (hcomponent : ∀ a, |stationarityResidual F x ((), a)| ≤ tolerance) :
    ApproxSolvesVI F orthant (Real.sqrt (Fintype.card α) * tolerance) x := by
  refine ⟨hx, fun z hz ↦ ?_⟩
  have hresidual := norm_stationarityResidual_le_sqrt_card_mul F x htolerance hcomponent
  have hnorm : 0 ≤ ‖z - x‖ := norm_nonneg _
  have hmul : -(Real.sqrt (Fintype.card α) * tolerance) * ‖z - x‖ ≤
      -‖stationarityResidual F x‖ * ‖z - x‖ := by
    exact mul_le_mul_of_nonneg_right (neg_le_neg hresidual) hnorm
  exact hmul.trans ((approxSolvesVI_of_stationarityResidual F hx).2 z hz)

/-- Dividing a natural projected-step residual by curvature is not a valid distance bound in
general. For `F(z) = 2z + 1` on the half-line, the point `z = 1` has natural residual one but
distance one from the exact solution zero, exceeding residual divided by curvature two. -/
theorem naturalResidual_div_curvature_counterexample :
    |(1 : ℝ) - max 0 (1 - (2 * 1 + 1))| / 2 < |(1 : ℝ) - 0| := by
  norm_num

end PricingPerspective.Operating.RationalShare.SolverCertificates.ProjectedResidual
