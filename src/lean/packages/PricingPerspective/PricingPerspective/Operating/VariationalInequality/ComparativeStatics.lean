import PricingPerspective.Operating.VariationalInequality.Proximal
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

/-!
# Directional comparative statics for rational-share responses

The finite marginal-revenue identity concerns fixed own appeal, not an observed trajectory.
The two-VI comparison concerns exact responses on the same feasible set. It signs a weighted
aggregate displacement; coordinatewise or normalized-composition signs need separate hypotheses.
-/

namespace PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

section Variational

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F G : E → E} {s : Set E} {mu : ℝ} {x y : E}

/-- Comparing exact VI solutions bounds the pairing with the incentive change at the old
solution. A nonnegative modulus makes this pairing nonpositive; no differentiability or
interiority is needed. -/
theorem response_direction_le (hG : StronglyMonotoneOn G s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI G s y) :
    ⟪G x - F x, y - x⟫ ≤ -mu * ‖y - x‖ ^ 2 := by
  have h1 := hx.2 y hy.1
  have h2 := hy.2 x hx.1
  have hm := hG y hy.1 x hx.1
  rw [show x - y = -(y - x) by abel, inner_neg_right] at h2
  simp only [inner_sub_left] at hm ⊢
  linarith

/-- A nonnegative monotonicity modulus signs the response against the incentive perturbation.
The definition `StronglyMonotoneOn` alone does not impose this modulus restriction. -/
theorem response_direction_nonpos (hmu : 0 ≤ mu) (hG : StronglyMonotoneOn G s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI G s y) : ⟪G x - F x, y - x⟫ ≤ 0 := by
  exact (response_direction_le hG hx hy).trans
    (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hmu) (sq_nonneg _))

/-- The dual endpoint comparison uses monotonicity of the old map and evaluates the incentive
change at the new response. These two versions must not silently exchange their hypotheses. -/
theorem response_direction_new_le (hF : StronglyMonotoneOn F s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI G s y) :
    ⟪G y - F y, y - x⟫ ≤ -mu * ‖y - x‖ ^ 2 := by
  have h1 := hx.2 y hy.1
  have h2 := hy.2 x hx.1
  have hm := hF y hy.1 x hx.1
  rw [show x - y = -(y - x) by abel, inner_neg_right] at h2
  simp only [inner_sub_left] at hm ⊢
  linarith

/-- Strong curvature bounds the response displacement by the incentive perturbation at the old
optimum. This is sensitivity of exact solutions, not convergence of an adjustment algorithm. -/
theorem response_norm_le (hmu : 0 < mu) (hG : StronglyMonotoneOn G s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI G s y) :
    ‖y - x‖ ≤ ‖G x - F x‖ / mu := by
  have hd := response_direction_le hG hx hy
  have hc := (abs_le.mp (abs_real_inner_le_norm (G x - F x) (y - x))).1
  rcases eq_or_lt_of_le (norm_nonneg (y - x)) with hz | hp
  · rw [← hz]
    exact div_nonneg (norm_nonneg _) hmu.le
  · apply (le_div_iff₀ hmu).2
    have : mu * ‖y - x‖ ^ 2 ≤ ‖G x - F x‖ * ‖y - x‖ := by linarith
    nlinarith

/-- In one dimension, an increased negative marginal payoff at the old exact optimum cannot
increase the new exact response. The feasible set may have boundary points. -/
theorem scalar_response_le {F G : ℝ → ℝ} {s : Set ℝ} {x y mu : ℝ}
    (hmu : 0 < mu) (hG : StronglyMonotoneOn G s mu)
    (hx : SolvesVI F s x) (hy : SolvesVI G s y) (hshift : F x ≤ G x) : y ≤ x := by
  have hd := response_direction_le hG hx hy
  simp only [real_inner_comm, Real.inner_apply, Real.norm_eq_abs, sq_abs] at hd
  by_contra h
  have hxy : 0 < y - x := sub_pos.mpr (lt_of_not_ge h)
  have hprod : 0 ≤ (G x - F x) * (y - x) := mul_nonneg (sub_nonneg.mpr hshift) hxy.le
  nlinarith [sq_pos_of_pos hxy]

/-- Strict response reduction on the nonnegative half-line needs an interior old optimum and a
strict incentive change. Boundary optima may remain zero despite a strict change. -/
theorem scalar_response_lt {F G : ℝ → ℝ} {x y mu : ℝ}
    (hmu : 0 < mu) (hG : StronglyMonotoneOn G (Set.Ici 0) mu)
    (hx : SolvesVI F (Set.Ici 0) x) (hy : SolvesVI G (Set.Ici 0) y)
    (hxpos : 0 < x) (hshift : F x < G x) : y < x := by
  have hle := scalar_response_le hmu hG hx hy hshift.le
  have hFx0 := hx.2 0 (by simp)
  have hFx1 := hx.2 (x + 1) (by simp only [Set.mem_Ici]; linarith)
  simp only [Real.inner_apply] at hFx0 hFx1
  have hFx : F x = 0 := by nlinarith
  have hne : y ≠ x := by
    intro heq
    subst y
    have hGy := hy.2 0 (by simp)
    simp only [Real.inner_apply] at hGy
    nlinarith
  exact lt_of_le_of_ne hle hne

end Variational

namespace Crowding

/-- Own marginal revenue at a demand node, with outside-plus-rival appeal `B`. -/
noncomputable def marginalRevenue (r A B : ℝ) : ℝ := r * B / (A + B) ^ 2

/-- Exact finite change at fixed own appeal. Unlike the cross derivative, this identity can
compare two distinct rival fields without integrating along an unspecified path. -/
theorem marginalRevenue_sub (r A B B' : ℝ) (hB : A + B ≠ 0) (hB' : A + B' ≠ 0) :
    marginalRevenue r A B' - marginalRevenue r A B =
      r * (B' - B) * (A ^ 2 - B * B') / ((A + B') ^ 2 * (A + B) ^ 2) := by
  unfold marginalRevenue
  field_simp
  ring

/-- Rival appeal weakly reduces marginal revenue when own appeal squared does not exceed the
product of the two rival fields. Zero revenue and unchanged rivals are permitted. -/
theorem marginalRevenue_le {r A B B' : ℝ} (hr : 0 ≤ r) (hA : 0 ≤ A)
    (hB : 0 < B) (hBB : B ≤ B') (hregion : A ^ 2 ≤ B * B') :
    marginalRevenue r A B' ≤ marginalRevenue r A B := by
  have hD : 0 < A + B := by linarith
  have hD' : 0 < A + B' := by linarith
  apply sub_nonpos.mp
  rw [marginalRevenue_sub r A B B' hD.ne' hD'.ne']
  exact div_nonpos_of_nonpos_of_nonneg
    (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hr (sub_nonneg.mpr hBB))
      (sub_nonpos.mpr hregion)) (by positivity)

/-- Strict marginal reduction needs positive revenue, a genuine rival increase and a strict
region condition; the generic game only assumes nonnegative revenue. -/
theorem marginalRevenue_lt {r A B B' : ℝ} (hr : 0 < r) (hA : 0 ≤ A)
    (hB : 0 < B) (hBB : B < B') (hregion : A ^ 2 < B * B') :
    marginalRevenue r A B' < marginalRevenue r A B := by
  have hD : 0 < A + B := by linarith
  have hD' : 0 < A + B' := by linarith
  apply sub_neg.mp
  rw [marginalRevenue_sub r A B B' hD.ne' hD'.ne']
  exact div_neg_of_neg_of_pos
    (mul_neg_of_pos_of_neg (mul_pos hr (sub_pos.mpr hBB)) (sub_neg.mpr hregion))
    (by positivity)

/-- The familiar rival-dominant condition is sufficient for the finite comparison, not only
for the sign of an infinitesimal cross derivative. -/
theorem marginalRevenue_le_of_dominant {r A B B' : ℝ} (hr : 0 ≤ r) (hA : 0 ≤ A)
    (hB : 0 < B) (hBB : B ≤ B') (hAB : A ≤ B) :
    marginalRevenue r A B' ≤ marginalRevenue r A B := by
  apply marginalRevenue_le hr hA hB hBB
  nlinarith

/-- Aggregation through nonnegative activity loadings preserves the nodewise marginal sign.
This signs incentives at fixed own capacity, not the inverse of a coupled response Hessian. -/
theorem weighted_marginal_le {κ : Type*} [Fintype κ] (r A B B' k : κ → ℝ)
    (hr : ∀ l, 0 ≤ r l) (hA : ∀ l, 0 ≤ A l) (hB : ∀ l, 0 < B l)
    (hBB : ∀ l, B l ≤ B' l) (hregion : ∀ l, A l ^ 2 ≤ B l * B' l)
    (hk : ∀ l, 0 ≤ k l) :
    (∑ l, k l * marginalRevenue (r l) (A l) (B' l)) ≤
      ∑ l, k l * marginalRevenue (r l) (A l) (B l) := by
  exact Finset.sum_le_sum fun l _ ↦ mul_le_mul_of_nonneg_left
    (marginalRevenue_le (hr l) (hA l) (hB l) (hBB l) (hregion l)) (hk l)

/-- Algebraic comparison of normalized fractions with positive denominators. To interpret the
arguments as masses, downstream callers additionally supply nonnegative components. -/
theorem normalized_share_le_iff {m u m' u' : ℝ} (ht : 0 < m + u) (ht' : 0 < m' + u') :
    m' / (m' + u') ≤ m / (m + u) ↔ m' * u ≤ m * u' := by
  rw [div_le_div_iff₀ ht' ht]
  constructor <;> intro h <;> nlinarith

/-- Decreasing crowded capacity reduces its normalized share when complementary capacity is
unchanged and positive. No fixed-total-capacity constraint is added to the orthant model. -/
theorem normalized_share_le {m m' u : ℝ} (hm : 0 ≤ m) (hm' : 0 ≤ m')
    (hu : 0 < u) (hmm : m' ≤ m) : m' / (m' + u) ≤ m / (m + u) := by
  apply (normalized_share_le_iff (add_pos_of_nonneg_of_pos hm hu)
    (add_pos_of_nonneg_of_pos hm' hu)).2
  exact mul_le_mul_of_nonneg_right hmm hu.le

/-- A strict raw reduction gives a strict normalized reduction under positive unchanged
complementary capacity. -/
theorem normalized_share_lt {m m' u : ℝ} (hm : 0 ≤ m) (hm' : 0 ≤ m')
    (hu : 0 < u) (hmm : m' < m) : m' / (m' + u) < m / (m + u) := by
  apply (div_lt_div_iff₀ (add_pos_of_nonneg_of_pos hm' hu)
    (add_pos_of_nonneg_of_pos hm hu)).2
  nlinarith

/-- Algebraic comparison of a frozen two-region score. Economic callers supply probability
shares; the inequality itself holds for arbitrary ordered real inputs. The score is held fixed. -/
theorem crowding_score_le {p p' w₁ w₂ : ℝ} (hp : p' ≤ p) (hw : w₂ ≤ w₁) :
    p' * w₁ + (1 - p') * w₂ ≤ p * w₁ + (1 - p) * w₂ := by
  nlinarith [mul_nonneg (sub_nonneg.mpr hp) (sub_nonneg.mpr hw)]

/-- Raw crowded capacity can fall while its normalized share rises if the complement falls
faster. This guards against interpreting capacity comparative statics as composition motion. -/
theorem raw_reduction_not_normalized_reduction :
    (1 : ℝ) < 2 ∧ 2 / (2 + 2 : ℝ) < 1 / (1 + 1 / 2 : ℝ) := by norm_num

end Crowding
end PricingPerspective.Operating.VariationalInequality
