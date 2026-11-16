import PricingPerspective.Operating.VariationalInequality.Monotone

/-!
# Approximate variational responses

Numerical optimizers return feasible points with a finite stationarity residual rather than
kernel-level exact maximizers. This module records the corresponding approximate variational
inequality and derives deterministic distance and directional bounds from strong monotonicity.
The statements are over real normed spaces; they do not certify floating-point evaluation.
-/

namespace PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F G : E → E} {s : Set E} {epsilon epsilonF epsilonG mu : ℝ} {x y : E}

/-- `x` solves the variational inequality up to radial error `epsilon`.

The error is measured in the dual pairing: every feasible comparison `z` may violate the
exact VI by at most `epsilon * ‖z - x‖`. Applications obtain `epsilon` from a stationarity
residual; no sign restriction is built into the definition itself. -/
def ApproxSolvesVI (F : E → E) (s : Set E) (epsilon : ℝ) (x : E) : Prop :=
  x ∈ s ∧ ∀ z ∈ s, -epsilon * ‖z - x‖ ≤ ⟪F x, z - x⟫

/-- An exact VI solution is an approximate solution for every nonnegative error allowance. -/
theorem SolvesVI.approxSolvesVI (hx : SolvesVI F s x) (hepsilon : 0 ≤ epsilon) :
    ApproxSolvesVI F s epsilon x := by
  refine ⟨hx.1, fun z hz ↦ ?_⟩
  exact (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hepsilon) (norm_nonneg _)).trans
    (hx.2 z hz)

/-- Zero-error approximate solutions are exactly ordinary VI solutions. -/
theorem approxSolvesVI_zero_iff : ApproxSolvesVI F s 0 x ↔ SolvesVI F s x := by
  constructor
  · rintro ⟨hx, hvi⟩
    exact ⟨hx, fun z hz ↦ by simpa using hvi z hz⟩
  · intro hx
    exact hx.approxSolvesVI le_rfl

/-- Strong monotonicity turns approximate stationarity into a distance certificate from the
exact VI solution. The denominator is the same curvature modulus used for uniqueness. -/
theorem approxSolvesVI_dist_le (hmu : 0 < mu) (hepsilon : 0 ≤ epsilon)
    (hF : StronglyMonotoneOn F s mu) (hx : ApproxSolvesVI F s epsilon x)
    (hy : SolvesVI F s y) : ‖x - y‖ ≤ epsilon / mu := by
  have hxvi := hx.2 y hy.1
  have hyvi := hy.2 x hx.1
  have hmono := hF x hx.1 y hy.1
  rw [show y - x = -(x - y) by abel, inner_neg_right, norm_neg] at hxvi
  simp only [inner_sub_left] at hmono
  have hquad : mu * ‖x - y‖ ^ 2 ≤ epsilon * ‖x - y‖ := by
    linarith
  rcases eq_or_lt_of_le (norm_nonneg (x - y)) with hzero | hpos
  · rw [← hzero]
    exact div_nonneg hepsilon hmu.le
  · apply (le_div_iff₀ hmu).2
    nlinarith

/-- Two approximate VI responses obey the exact directional comparison with the sum of their
radial errors added to the curvature bound. The perturbation is evaluated at the old computed
response `x`, so no exact endpoint is substituted into a numerical quantity. -/
theorem approx_response_direction_le (hG : StronglyMonotoneOn G s mu)
    (hx : ApproxSolvesVI F s epsilonF x) (hy : ApproxSolvesVI G s epsilonG y) :
    ⟪G x - F x, y - x⟫ ≤
      -mu * ‖y - x‖ ^ 2 + (epsilonF + epsilonG) * ‖y - x‖ := by
  have hxvi := hx.2 y hy.1
  have hyvi := hy.2 x hx.1
  have hmono := hG y hy.1 x hx.1
  rw [show x - y = -(y - x) by abel, inner_neg_right, norm_neg] at hyvi
  simp only [inner_sub_left] at hmono ⊢
  linarith

/-- If curvature times the observed displacement strictly exceeds total response error, the
computed displacement has a strictly negative pairing with the incentive perturbation. -/
theorem approx_response_direction_lt (hG : StronglyMonotoneOn G s mu)
    (hx : ApproxSolvesVI F s epsilonF x) (hy : ApproxSolvesVI G s epsilonG y)
    (hxy : x ≠ y) (hmargin : epsilonF + epsilonG < mu * ‖y - x‖) :
    ⟪G x - F x, y - x⟫ < 0 := by
  have hbound := approx_response_direction_le hG hx hy
  have hnorm : 0 < ‖y - x‖ := norm_pos_iff.mpr (sub_ne_zero.mpr (Ne.symm hxy))
  have herror : (epsilonF + epsilonG) * ‖y - x‖ <
      mu * ‖y - x‖ * ‖y - x‖ := mul_lt_mul_of_pos_right hmargin hnorm
  nlinarith [hbound]

end PricingPerspective.Operating.VariationalInequality
