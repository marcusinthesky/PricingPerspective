import PricingPerspective.Operating.RationalShare.Reallocation.MovingOpportunity
import PricingPerspective.Operating.VariationalInequality.Proximal

/-!
# Anchored adjustment costs: reallocation and joint proximal contraction

The moving-opportunity restriction cancels a *stable* cost function across two dates. An
adjustment cost centred on the previous position is not stable: its anchor moves with the path.
This module records what survives.

**Reallocation.** With dated adjustment costs the two choice inequalities no longer cancel
completely. The full gain is bounded below by the change in the adjustment-cost differential
between the two positions. For quadratic anchors that correction is exactly
`-kappa * ⟪a₁ - a₀, q¹ - q⁰⟫`. It vanishes when the anchor is common to both dates, which
recovers `MovingOpportunity.fullGain_ge_neg_allowances`. It is negative when successive moves
share a direction, so a negative full gain is then compatible with optimizing choice.

**Joint proximal contraction.** Here every firm plays the stage game against *contemporaneous*
rivals, penalised by `kappa / 2 * ‖q - a‖ ^ 2` around the previous profile `a`. The stage
equilibrium is the joint proximal response `ProxSolvesVI F s kappa a`. Its factor is
`kappa / (kappa + nu)`, with no cross-sensitivity condition, because rival effects sit inside the
strongly monotone game map `F`. The firm-level reading of `Operating/VariationalInequality/Proximal.lean`, in which
firms respond to *lagged* rivals, is a different timing assumption with factor
`(c + L) / (c + mu)`. The two must not be conflated.

The penalty changes the path, never the destination (`proxSolvesVI_self_iff`). The contraction
is therefore a statement about convergence to the static equilibrium. It rules out critical
amplification in the convex game.
-/

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse.AnchoredAdjustment

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.Reallocation

open scoped RealInnerProductSpace
open MovingOpportunity

section Reallocation

variable {Position Opportunity Rival : Type*}

/-- **Anchored reallocation.** Suppose each choice is approximately optimal in its own dated
environment against a stable cost plus a dated adjustment cost. Then the full gain is bounded
below by the change in the adjustment-cost differential between the positions, less the two
payoff regrets.

With `oldAdjust = newAdjust` the correction is zero and the stable-cost restriction returns. -/
theorem fullGain_ge_adjustmentCorrection
    (revenue : Position → Opportunity → Rival → ℝ) (cost oldAdjust newAdjust : Position → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) (oldAllowance newAllowance : ℝ)
    (oldChoice :
      revenue newPosition oldOpportunity oldRival - cost newPosition - oldAdjust newPosition ≤
        revenue oldPosition oldOpportunity oldRival - cost oldPosition - oldAdjust oldPosition +
          oldAllowance)
    (newChoice :
      revenue oldPosition newOpportunity newRival - cost oldPosition - newAdjust oldPosition ≤
        revenue newPosition newOpportunity newRival - cost newPosition - newAdjust newPosition +
          newAllowance) :
    (newAdjust newPosition - newAdjust oldPosition) -
        (oldAdjust newPosition - oldAdjust oldPosition) - (oldAllowance + newAllowance) ≤
      fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival := by
  simp only [fullGain, relativeAdvantage]
  linarith

end Reallocation

section Quadratic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Quadratic adjustment cost around an anchor position. -/
noncomputable def anchorCost (kappa : ℝ) (anchor q : E) : ℝ :=
  kappa / 2 * ‖q - anchor‖ ^ 2

/-- **The quadratic-anchor correction.** The change across dates in the adjustment-cost
differential between two positions is `-kappa * ⟪a₁ - a₀, q¹ - q⁰⟫`. -/
theorem anchorCost_correction (kappa : ℝ) (oldAnchor newAnchor oldPosition newPosition : E) :
    (anchorCost kappa newAnchor newPosition - anchorCost kappa newAnchor oldPosition) -
        (anchorCost kappa oldAnchor newPosition - anchorCost kappa oldAnchor oldPosition) =
      -(kappa * ⟪newAnchor - oldAnchor, newPosition - oldPosition⟫) := by
  simp only [anchorCost, norm_sub_sq_real, inner_sub_left, inner_sub_right,
    real_inner_comm newAnchor, real_inner_comm oldAnchor]
  ring

omit [InnerProductSpace ℝ E] in
/-- A common anchor produces no correction. -/
theorem anchorCost_correction_self (kappa : ℝ) (anchor oldPosition newPosition : E) :
    (anchorCost kappa anchor newPosition - anchorCost kappa anchor oldPosition) -
        (anchorCost kappa anchor newPosition - anchorCost kappa anchor oldPosition) = 0 :=
  sub_self _

/-- **Anchored reallocation with quadratic anchors.** Approximate choices against a stable cost
plus `kappa / 2 * ‖q - aₜ‖ ^ 2` bound the full gain below by
`-kappa * ⟪a₁ - a₀, q¹ - q⁰⟫` less the two payoff regrets.

With sequential anchors `a₀ = q⁻¹` and `a₁ = q⁰`, the bound is negative whenever the two
successive moves point in the same direction. -/
theorem fullGain_ge_neg_anchor_correction {Opportunity Rival : Type*}
    (revenue : E → Opportunity → Rival → ℝ) (cost : E → ℝ) (kappa : ℝ)
    (oldAnchor newAnchor oldPosition newPosition : E)
    (oldOpportunity newOpportunity : Opportunity) (oldRival newRival : Rival)
    (oldAllowance newAllowance : ℝ)
    (oldChoice :
      revenue newPosition oldOpportunity oldRival - cost newPosition -
          anchorCost kappa oldAnchor newPosition ≤
        revenue oldPosition oldOpportunity oldRival - cost oldPosition -
          anchorCost kappa oldAnchor oldPosition + oldAllowance)
    (newChoice :
      revenue oldPosition newOpportunity newRival - cost oldPosition -
          anchorCost kappa newAnchor oldPosition ≤
        revenue newPosition newOpportunity newRival - cost newPosition -
          anchorCost kappa newAnchor newPosition + newAllowance) :
    -(kappa * ⟪newAnchor - oldAnchor, newPosition - oldPosition⟫) -
        (oldAllowance + newAllowance) ≤
      fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival := by
  have h := fullGain_ge_adjustmentCorrection revenue cost (anchorCost kappa oldAnchor)
    (anchorCost kappa newAnchor) oldPosition newPosition oldOpportunity newOpportunity oldRival
    newRival oldAllowance newAllowance oldChoice newChoice
  rwa [anchorCost_correction] at h

end Quadratic

section JointProximal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F : E → E} {s : Set E} {kappa nu : ℝ}

/-- **Joint proximal contraction.** Stage equilibria of the anchored game, with contemporaneous
rivals, move by at most `kappa / (kappa + nu)` times the change in their anchors.

This is `proximal_response_dist_le` with one game map on both sides and zero cross-sensitivity.
Strong monotonicity of the joint map already absorbs rival effects. -/
theorem jointProx_dist_le (hk : 0 ≤ kappa) (hF : StronglyMonotoneOn F s nu) {a b y z : E}
    (hy : ProxSolvesVI F s kappa a y) (hz : ProxSolvesVI F s kappa b z) :
    (kappa + nu) * ‖y - z‖ ≤ kappa * ‖a - b‖ := by
  have h := proximal_response_dist_le (g := F) (gt := F) (mu := nu) (L := 0) hk
    (norm_nonneg (a - b)) hy hz (hF y hy.1 z hz.1) (by simp) le_rfl
  simpa using h

/-- The contraction in factor form. -/
theorem jointProx_dist_le_factor (hk : 0 ≤ kappa) (hpos : 0 < kappa + nu)
    (hF : StronglyMonotoneOn F s nu) {a b y z : E}
    (hy : ProxSolvesVI F s kappa a y) (hz : ProxSolvesVI F s kappa b z) :
    ‖y - z‖ ≤ kappa / (kappa + nu) * ‖a - b‖ := by
  rw [div_mul_eq_mul_div, le_div_iff₀ hpos]
  linarith [jointProx_dist_le hk hF hy hz]

/-- **The stage equilibrium is unique.** Two stage responses to the same anchor coincide. -/
theorem jointProx_unique (hk : 0 ≤ kappa) (hpos : 0 < kappa + nu)
    (hF : StronglyMonotoneOn F s nu) {a y z : E}
    (hy : ProxSolvesVI F s kappa a y) (hz : ProxSolvesVI F s kappa a z) : y = z := by
  have h := jointProx_dist_le hk hF hy hz
  rw [sub_self, norm_zero, mul_zero] at h
  have hn : ‖y - z‖ ≤ 0 := by
    nlinarith [norm_nonneg (y - z)]
  exact sub_eq_zero.mp (norm_le_zero_iff.mp hn)

/-- **Geometric convergence to the static equilibrium.** Along a path of anchored stage
equilibria, the distance to the unpenalised equilibrium falls at least geometrically, at rate
`kappa / (kappa + nu)`. -/
theorem jointProx_iterate_dist_le (hk : 0 ≤ kappa) (hpos : 0 < kappa + nu)
    (hF : StronglyMonotoneOn F s nu) {qstar : E} (hstar : SolvesVI F s qstar) (q : ℕ → E)
    (hstep : ∀ t, ProxSolvesVI F s kappa (q t) (q (t + 1))) (t : ℕ) :
    ‖q t - qstar‖ ≤ (kappa / (kappa + nu)) ^ t * ‖q 0 - qstar‖ := by
  have hfix : ProxSolvesVI F s kappa qstar qstar := proxSolvesVI_self_iff.mpr hstar
  have hrho : 0 ≤ kappa / (kappa + nu) := div_nonneg hk hpos.le
  induction t with
  | zero => simp
  | succ t ih =>
    have h1 := jointProx_dist_le_factor hk hpos hF (hstep t) hfix
    calc ‖q (t + 1) - qstar‖ ≤ kappa / (kappa + nu) * ‖q t - qstar‖ := h1
      _ ≤ kappa / (kappa + nu) * ((kappa / (kappa + nu)) ^ t * ‖q 0 - qstar‖) :=
          mul_le_mul_of_nonneg_left ih hrho
      _ = (kappa / (kappa + nu)) ^ (t + 1) * ‖q 0 - qstar‖ := by ring

end JointProximal

end PricingPerspective.Operating.RationalShare.EquilibriumResponse.AnchoredAdjustment
