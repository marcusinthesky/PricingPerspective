import Architect
import PricingPerspective.Operating.Valuation.CashFlowResponse
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.EquilibriumDerivative

/-!
# How equilibrium free cash flow responds to opportunity values

Combining the derivative of the equilibrium map (`EquilibriumDerivative.lean`) with the envelope
split (`CashFlowResponse.lean`) gives the equilibrium cash-flow sensitivity. Under strict
complementarity, a firm's equilibrium free cash flow changes, to first order in
`δr = r - r₀`, by

    Σ_l δr_l X_il  +  rival value effect of the linear response χ δr.

The firm's own adjustment does not enter. The response is supported on the active set, where
the field vanishes. Rivals' adjustment enters through crowding.
-/

namespace PricingPerspective.Operating.Valuation.EquilibriumCashFlow

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare
  PricingPerspective.Operating.RationalShare.EquilibriumResponse
  PricingPerspective.Operating.Valuation.CashFlowResponse

open Filter Topology
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

section Payoff

variable (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α)

theorem differentiableAt_payoff (hx : ∀ l, N.denom x l ≠ 0) (i : ι) :
    DifferentiableAt ℝ (payoff N c H i) x := by
  unfold payoff
  simp only [Nodes.denom, Nodes.appeal]
  fun_prop (disch := simp_all [Nodes.denom, Nodes.appeal])

/-- The Fréchet derivative of cash flow is the own-adjustment plus rival value effect. -/
theorem fderiv_payoff_apply (hsymm : ∀ i a b, H i a b = H i b a)
    (hx : ∀ l, 0 < N.denom x l) (i : ι) (u : Profile ι α) :
    fderiv ℝ (payoff N c H i) x u =
      ownAdjustmentEffect N c H x u i + rivalValueEffect N x u i := by
  have hdiff := (differentiableAt_payoff N c H x (fun l ↦ (hx l).ne') i).hasFDerivAt
  have hline : HasDerivAt (fun t : ℝ ↦ x + t • u) u 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const u).const_add x
  have hdiff' : HasFDerivAt (payoff N c H i) (fderiv ℝ (payoff N c H i) x)
      (x + (0 : ℝ) • u) := by simpa using hdiff
  have hcomp : HasDerivAt (fun t : ℝ ↦ payoff N c H i (x + t • u))
      (fderiv ℝ (payoff N c H i) x u) 0 := by
    simpa [Function.comp_def] using hdiff'.comp_hasDerivAt (0 : ℝ) hline
  exact hcomp.unique (hasDerivAt_payoff_line N c H x hsymm hx i u)

theorem continuousAt_share (hx : ∀ l, N.denom x l ≠ 0) (i : ι) (l : κ) :
    ContinuousAt (fun z : Profile ι α ↦ N.share z i l) x := by
  simp only [Nodes.share, Nodes.denom, Nodes.appeal]
  fun_prop (disch := simp_all [Nodes.denom, Nodes.appeal])

end Payoff

variable [DecidableEq ι]

omit [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι] in
/-- A tolerance divided by one more than a nonnegative constant absorbs that constant. -/
theorem div_succ_mul_le {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : a / (b + 1) * b ≤ a := by
  rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
  nlinarith

/-- **Equilibrium cash-flow sensitivity.** Under strict complementarity, the equilibrium free
cash flow of firm `i` responds to opportunity values, to first order, by direct opportunity news
plus the rival value effect of the linear response. The firm's own adjustment contributes
nothing at first order. -/
@[blueprint "eq:equilibrium-cash-flow"
  (statement := /--
    Equilibrium free cash flow responds to opportunity values through direct news plus the value
    of rivals' equilibrium repositioning. Assumptions: Assumption 1 primitives and an
    equilibrium with strict complementarity.
  -/)]
theorem equilibriumCashFlow_sub_le (N : Nodes κ α) (c : ι → α → ℝ)
    {H : ι → α → α → ℝ} {mu : ℝ} (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) {x : Profile ι α} (hxN : IsNashOrthant N c H x)
    (hsc : ∀ p, 0 < x p ∨ 0 < pseudoGradient N c H x p) (i : ι) {ε : ℝ} (hε : 0 < ε) :
    ∃ d > 0, ∀ (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l), opportunityDist N.revenue r < d →
      ∀ y, IsNashOrthant (N.withRevenue r hr) c H y →
      ∀ u, SolvesRestricted (jac N H x) (restrictL fun p ↦ 0 < x p)
          (opportunityImpulse N x fun l ↦ r l - N.revenue l) u →
        |payoff (N.withRevenue r hr) c H i y - payoff N c H i x -
            ((∑ l, (r l - N.revenue l) * N.share x i l) + rivalValueEffect N x u i)| ≤
          ε * opportunityDist N.revenue r := by
  classical
  have hx0 : ∀ p, 0 ≤ x p := hxN.1
  have hDpos : ∀ l, 0 < N.denom x l := N.denom_pos hx0
  have hxcomp := complementarity_of_solvesVI
    (solvesVI_capacityOrthant_of_isNashOrthant N c hsymm hmu hH hc hxN)
  set Dπ := fderiv ℝ (payoff N c H i) x with hDπ
  set B := ‖Dπ‖ with hB
  have hB0 : 0 ≤ B := norm_nonneg _
  set M := Real.sqrt (opportunityModulusSq ι N) with hM
  have hM0 : 0 ≤ M := Real.sqrt_nonneg _
  set K := M / mu with hK
  have hK0 : 0 ≤ K := div_nonneg hM0 hmu.le
  -- Three tolerances, each a third of `ε`.
  set eta := ε / 3 / (K + 1) with heta
  have heta0 : 0 < eta := by positivity
  set epsχ := ε / 3 / (B + 1) with hepsχ
  have hepsχ0 : 0 < epsχ := by positivity
  set epsX := ε / 3 / (Real.sqrt (Fintype.card κ) + 1) with hepsX
  have hepsX0 : 0 < epsX := by positivity
  -- The equilibrium derivative.
  obtain ⟨d1, hd1, hresp⟩ :=
    equilibrium_sub_linearResponse_le N c hsymm hmu hH hc hxN hsc hepsχ0
  -- Fréchet expansion of cash flow.
  have hfd := (differentiableAt_payoff N c H x (fun l ↦ (hDpos l).ne') i).hasFDerivAt
  obtain ⟨d2, hd2, hball2⟩ := Metric.eventually_nhds_iff.mp (hfd.isLittleO.def heta0)
  -- Continuity of the shares.
  have hshare : ∀ l, ContinuousAt (fun z : Profile ι α ↦ N.share z i l) x :=
    fun l ↦ continuousAt_share N x (fun l ↦ (hDpos l).ne') i l
  obtain ⟨d3, hd3, hball3⟩ : ∃ d3 > 0, ∀ z : Profile ι α, dist z x < d3 →
      ∀ l, |N.share z i l - N.share x i l| < epsX := by
    have hev : ∀ᶠ z in 𝓝 x, ∀ l, |N.share z i l - N.share x i l| < epsX := by
      rw [Filter.eventually_all]
      intro l
      have := (hshare l).eventually (Metric.ball_mem_nhds (N.share x i l) hepsX0)
      filter_upwards [this] with z hz
      simpa [Real.dist_eq] using hz
    exact Metric.eventually_nhds_iff.mp hev
  set d := min d1 (min d2 d3 / (K + 1)) with hd
  have hdpos : 0 < d := lt_min hd1 (div_pos (lt_min hd2 hd3) (by positivity))
  refine ⟨d, hdpos, fun r hr hρ y hy u hu ↦ ?_⟩
  set δ : κ → ℝ := fun l ↦ r l - N.revenue l with hδ
  set ρ := opportunityDist N.revenue r with hρdef
  have hρ0 : 0 ≤ ρ := Real.sqrt_nonneg _
  -- Continuity of the equilibrium.
  have hcont : ‖y - x‖ ≤ K * ρ := by
    have h := orthantEquilibrium_dist_le_opportunityShift N r hr c hsymm hmu hH hc hxN hy
    calc ‖y - x‖ ≤ M * ρ / mu := h
      _ = K * ρ := by rw [hK]; ring
  have hKρ : K * ρ < min d2 d3 := by
    have h1 : ρ < min d2 d3 / (K + 1) := hρ.trans_le (min_le_right _ _)
    have h2 : (K + 1) * ρ < min d2 d3 := by
      rwa [lt_div_iff₀ (by positivity), mul_comm] at h1
    have h3 : (K + 1) * ρ = K * ρ + ρ := by ring
    linarith
  have hyx2 : dist y x < d2 := by
    rw [dist_eq_norm]; exact hcont.trans_lt (hKρ.trans_le (min_le_left _ _))
  have hyx3 : dist y x < d3 := by
    rw [dist_eq_norm]; exact hcont.trans_lt (hKρ.trans_le (min_le_right _ _))
  -- Decompose the cash-flow change.
  have hnews : payoff (N.withRevenue r hr) c H i y - payoff N c H i y =
      ∑ l, δ l * N.share y i l := payoff_withRevenue_sub N r hr c H i y
  have hown : ownAdjustmentEffect N c H x u i = 0 := by
    rw [ownAdjustmentEffect, neg_eq_zero]
    refine Finset.sum_eq_zero fun a _ ↦ ?_
    by_cases hp : 0 < x (i, a)
    · rw [(hxcomp (i, a)).2 hp, zero_mul]
    · have hu0 : u (i, a) = 0 := by
        have := congrArg (fun v : Profile ι α ↦ v (i, a)) hu.1
        simpa [hp] using this.symm
      rw [hu0, mul_zero]
  have hDu : Dπ u = rivalValueEffect N x u i := by
    rw [hDπ, fderiv_payoff_apply N c H x hsymm hDpos i u, hown, zero_add]
  -- Three error terms.
  have e1 : |payoff N c H i y - payoff N c H i x - Dπ (y - x)| ≤ eta * ‖y - x‖ := by
    have := hball2 hyx2
    rwa [← Real.norm_eq_abs]
  have e2 : |Dπ (y - x) - Dπ u| ≤ B * (epsχ * ρ) := by
    rw [← map_sub, ← Real.norm_eq_abs]
    exact (Dπ.le_opNorm _).trans
      (mul_le_mul_of_nonneg_left (hresp r hr (hρ.trans_le (min_le_left _ _)) y hy u hu) hB0)
  have e3 : |∑ l, δ l * N.share y i l - ∑ l, δ l * N.share x i l| ≤
      epsX * (Real.sqrt (Fintype.card κ) * ρ) := by
    rw [← Finset.sum_sub_distrib]
    have hterm : ∀ l, |δ l * N.share y i l - δ l * N.share x i l| ≤ |δ l| * epsX := by
      intro l
      rw [← mul_sub, abs_mul]
      exact mul_le_mul_of_nonneg_left (hball3 y hyx3 l).le (abs_nonneg _)
    have hsum : ∑ l, |δ l| ≤ Real.sqrt (Fintype.card κ) * ρ := by
      have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ ↦ (1 : ℝ))
        (fun l ↦ |δ l|)
      simp only [one_pow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one,
        one_mul, sq_abs] at hcs
      have hnn : 0 ≤ ∑ l, |δ l| := Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
      have hS : 0 ≤ ∑ l, δ l ^ 2 := Finset.sum_nonneg fun _ _ ↦ sq_nonneg _
      rw [hρdef, opportunityDist, ← Real.sqrt_mul (Nat.cast_nonneg _)]
      exact Real.le_sqrt_of_sq_le hcs
    calc |∑ l, (δ l * N.share y i l - δ l * N.share x i l)|
        ≤ ∑ l, |δ l * N.share y i l - δ l * N.share x i l| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ l, |δ l| * epsX := Finset.sum_le_sum fun l _ ↦ hterm l
      _ = epsX * ∑ l, |δ l| := by rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ ↦ by ring
      _ ≤ epsX * (Real.sqrt (Fintype.card κ) * ρ) := mul_le_mul_of_nonneg_left hsum hepsX0.le
  -- Assemble.
  have hsplit : payoff (N.withRevenue r hr) c H i y - payoff N c H i x -
      ((∑ l, δ l * N.share x i l) + rivalValueEffect N x u i) =
      (payoff N c H i y - payoff N c H i x - Dπ (y - x)) + (Dπ (y - x) - Dπ u) +
        (∑ l, δ l * N.share y i l - ∑ l, δ l * N.share x i l) := by
    rw [← hDu, ← hnews]; ring
  rw [hsplit]
  have hsq : 0 ≤ Real.sqrt (Fintype.card κ) := Real.sqrt_nonneg _
  calc |_ + _ + _| ≤ |payoff N c H i y - payoff N c H i x - Dπ (y - x)| +
        |Dπ (y - x) - Dπ u| + |∑ l, δ l * N.share y i l - ∑ l, δ l * N.share x i l| :=
        (abs_add_le _ _).trans (add_le_add_left (abs_add_le _ _) _)
    _ ≤ eta * (K * ρ) + B * (epsχ * ρ) + epsX * (Real.sqrt (Fintype.card κ) * ρ) := by
        have := mul_le_mul_of_nonneg_left hcont heta0.le
        linarith
    _ ≤ ε / 3 * ρ + ε / 3 * ρ + ε / 3 * ρ := by
        have hε3 : 0 ≤ ε / 3 := by positivity
        have t1 : eta * (K * ρ) ≤ ε / 3 * ρ := by
          rw [← mul_assoc]
          exact mul_le_mul_of_nonneg_right (div_succ_mul_le hε3 hK0) hρ0
        have t2 : B * (epsχ * ρ) ≤ ε / 3 * ρ := by
          rw [← mul_assoc, mul_comm B epsχ]
          exact mul_le_mul_of_nonneg_right (div_succ_mul_le hε3 hB0) hρ0
        have t3 : epsX * (Real.sqrt (Fintype.card κ) * ρ) ≤ ε / 3 * ρ := by
          rw [← mul_assoc]
          exact mul_le_mul_of_nonneg_right (div_succ_mul_le hε3 hsq) hρ0
        linarith
    _ = ε * ρ := by ring

end PricingPerspective.Operating.Valuation.EquilibriumCashFlow
