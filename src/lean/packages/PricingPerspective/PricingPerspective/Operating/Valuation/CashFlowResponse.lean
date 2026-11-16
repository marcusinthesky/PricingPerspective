import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.LinearDynamics
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.ResponseOperator
import Mathlib.Analysis.SpecificLimits.Normed

/-!
# Operating cash flow along the response: envelope split and discounted value

**Envelope split.** Along any direction `(u, δ)` of capacities and opportunity values, operating
cash flow `π_i` changes at rate

    Σ_l δ_l X_il  -  Σ_a F_ia u_ia  -  Σ_l r_l A_il / D_l² (U_l - u_il),

where `X_il = A_il / D_l` is the firm's share, `U_l` the total appeal change and `u_il` the firm's
own appeal change. The terms are direct opportunity news, the firm's own adjustment weighted by
its marginal incentive, and the value effect of rivals' movement.

* At a static equilibrium the field vanishes wherever capacity moves. The own term drops out,
  leaving **direct news plus the rival response**.
* At an anchored stage equilibrium the field equals `-kappa (q - a)` on the active set, so the
  own term is `kappa ⟪q - a, u⟫` and does *not* vanish. When the stage objective includes an
  adjustment cost that cash flow excludes, the envelope split fails.

**Discounted value.** For a linear state recursion `x_{h+1} = M x_h` with `‖β M‖ < 1`, the
discounted value `Σ β^h H M^h x` equals `H (1 - β M)⁻¹ x`: the valuation resolvent. For the
linearised anchored dynamics, which decay at `ρ = kappa / (mu + kappa)`, the discounted value of
any linear cash-flow read-out is summable and bounded by `‖w‖ ‖v₀‖ / (1 - β ρ)`.
-/

namespace PricingPerspective.Operating.Valuation.CashFlowResponse

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare
  PricingPerspective.Operating.RationalShare.EquilibriumResponse

open scoped RealInnerProductSpace

/-! ### Discounted value -/

section Valuation

variable {E G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- **The valuation resolvent.** If `‖β • M‖ < 1`, the discounted read-out of the state path
`M^h x` sums to `H ((1 - β • M)⁻¹ x)`. -/
theorem hasSum_discounted_readout (M : E →L[ℝ] E) (Hr : E →L[ℝ] G) {β : ℝ}
    (hβ : ‖β • M‖ < 1) (x : E) :
    HasSum (fun h : ℕ ↦ β ^ h • Hr ((M ^ h) x)) (Hr (Ring.inverse (1 - β • M) x)) := by
  have hgeom := hasSum_geom_series_inverse (β • M) hβ
  have happ := hgeom.mapL ((Hr.comp (ContinuousLinearMap.apply ℝ E x)))
  refine happ.congr_fun fun h ↦ ?_
  simp [smul_pow]

/-- The resolvent exists: `1 - β • M` is a unit. -/
theorem isUnit_one_sub_smul (M : E →L[ℝ] E) {β : ℝ} (hβ : ‖β • M‖ < 1) :
    IsUnit (1 - β • M) :=
  isUnit_one_sub_of_norm_lt_one hβ

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **Discounted value of a geometrically decaying path.** If `‖v h‖ ≤ ρ^h ‖v 0‖` and
`β ρ < 1`, the discounted read-out `Σ β^h ⟪w, v h⟫` is summable and bounded by
`‖w‖ ‖v 0‖ / (1 - β ρ)`. -/
theorem discounted_readout_bound {v : ℕ → F} {ρ β : ℝ} (hρ : 0 ≤ ρ) (hβ : 0 ≤ β)
    (hβρ : β * ρ < 1) (hv : ∀ h, ‖v h‖ ≤ ρ ^ h * ‖v 0‖) (w : F) :
    Summable (fun h : ℕ ↦ β ^ h * ⟪w, v h⟫) ∧
      |∑' h : ℕ, β ^ h * ⟪w, v h⟫| ≤ ‖w‖ * ‖v 0‖ / (1 - β * ρ) := by
  have hq0 : 0 ≤ β * ρ := mul_nonneg hβ hρ
  have hterm : ∀ h : ℕ, ‖β ^ h * ⟪w, v h⟫‖ ≤ ‖w‖ * ‖v 0‖ * (β * ρ) ^ h := by
    intro h
    rw [norm_mul, Real.norm_of_nonneg (pow_nonneg hβ h), Real.norm_eq_abs]
    calc β ^ h * |⟪w, v h⟫| ≤ β ^ h * (‖w‖ * ‖v h‖) :=
          mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) (pow_nonneg hβ h)
      _ ≤ β ^ h * (‖w‖ * (ρ ^ h * ‖v 0‖)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (hv h) (norm_nonneg _))
            (pow_nonneg hβ h)
      _ = ‖w‖ * ‖v 0‖ * (β * ρ) ^ h := by rw [mul_pow]; ring
  have hgeom : Summable (fun h : ℕ ↦ ‖w‖ * ‖v 0‖ * (β * ρ) ^ h) :=
    (summable_geometric_of_lt_one hq0 hβρ).mul_left _
  have hsum : Summable (fun h : ℕ ↦ β ^ h * ⟪w, v h⟫) := hgeom.of_norm_bounded hterm
  refine ⟨hsum, ?_⟩
  rw [← Real.norm_eq_abs]
  calc ‖∑' h : ℕ, β ^ h * ⟪w, v h⟫‖ ≤ ∑' h : ℕ, ‖w‖ * ‖v 0‖ * (β * ρ) ^ h :=
        tsum_of_norm_bounded (hgeom.hasSum) hterm
    _ = ‖w‖ * ‖v 0‖ / (1 - β * ρ) := by
        rw [tsum_mul_left, tsum_geometric_of_lt_one hq0 hβρ, div_eq_mul_inv]

end Valuation

/-! ### The envelope split -/

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]
variable (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α)

/-- The value effect of rivals' movement along `u`: `-Σ_l r_l A_il / D_l² (U_l - u_il)`. -/
noncomputable def rivalValueEffect (u : Profile ι α) (i : ι) : ℝ :=
  -∑ l, N.revenue l * N.appeal x i l / N.denom x l ^ 2 * (dirSum N u l - N.appeal u i l)

/-- The firm's own adjustment weighted by its marginal incentive: `-Σ_a F_ia u_ia`. -/
noncomputable def ownAdjustmentEffect (u : Profile ι α) (i : ι) : ℝ :=
  -∑ a, pseudoGradient N c H x (i, a) * u (i, a)

/-- **Cash flow along a capacity direction.** -/
theorem hasDerivAt_payoff_line (hsymm : ∀ i a b, H i a b = H i b a)
    (hx : ∀ l, 0 < N.denom x l) (i : ι) (u : Profile ι α) :
    HasDerivAt (fun t : ℝ ↦ payoff N c H i (x + t • u))
      (ownAdjustmentEffect N c H x u i + rivalValueEffect N x u i) 0 := by
  have reshape : (fun t : ℝ ↦ payoff N c H i (x + t • u)) = fun t : ℝ ↦
      (∑ l, N.revenue l * (N.appeal x i l + t * N.appeal u i l) /
          (N.denom x l + t * dirSum N u l))
        - (∑ a, c i a * (x (i, a) + t * u (i, a)))
        - (1 / 2) * ∑ a, ∑ b, H i a b * (x (i, a) + t * u (i, a)) * (x (i, b) + t * u (i, b)) := by
    funext t
    simp only [payoff, appeal_add_smul, denom_add_smul, PiLp.add_apply, PiLp.smul_apply,
      smul_eq_mul]
  have haff : ∀ (p q : ℝ), HasDerivAt (fun t : ℝ ↦ p + t * q) q 0 := fun p q ↦ by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const q).const_add p
  have hrev : HasDerivAt (fun t : ℝ ↦ ∑ l, N.revenue l * (N.appeal x i l + t * N.appeal u i l) /
        (N.denom x l + t * dirSum N u l))
      (∑ l, N.revenue l * (N.appeal u i l * N.denom x l - N.appeal x i l * dirSum N u l) /
        N.denom x l ^ 2) 0 := by
    refine HasDerivAt.fun_sum fun l _ ↦ ?_
    have hne : N.denom x l + 0 * dirSum N u l ≠ 0 := by simpa using (hx l).ne'
    refine (((haff _ _).const_mul (N.revenue l)).div (haff _ _) hne).congr_deriv ?_
    simp only [zero_mul, add_zero]
    ring
  have hlin : HasDerivAt (fun t : ℝ ↦ ∑ a, c i a * (x (i, a) + t * u (i, a)))
      (∑ a, c i a * u (i, a)) 0 :=
    HasDerivAt.fun_sum fun a _ ↦ (haff _ _).const_mul (c i a)
  have hquad : HasDerivAt (fun t : ℝ ↦ (1 / 2) * ∑ a, ∑ b,
        H i a b * (x (i, a) + t * u (i, a)) * (x (i, b) + t * u (i, b)))
      ((1 / 2) * ∑ a, ∑ b, H i a b * (u (i, a) * x (i, b) + x (i, a) * u (i, b))) 0 := by
    refine HasDerivAt.const_mul _ (HasDerivAt.fun_sum fun a _ ↦ HasDerivAt.fun_sum fun b _ ↦ ?_)
    refine (((haff _ _).const_mul (H i a b)).mul (haff _ _)).congr_deriv ?_
    simp only [zero_mul, add_zero]
    ring
  rw [reshape]
  refine ((hrev.sub hlin).sub hquad).congr_deriv ?_
  -- Symmetry turns the quadratic derivative into `Σ_a (H x)_a u_a`.
  have hsymsum : (1 / 2) * ∑ a, ∑ b, H i a b * (u (i, a) * x (i, b) + x (i, a) * u (i, b)) =
      ∑ a, (∑ b, H i a b * x (i, b)) * u (i, a) := by
    have hswap : ∑ a, ∑ b, H i a b * (x (i, a) * u (i, b)) =
        ∑ a, ∑ b, H i a b * (u (i, a) * x (i, b)) := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun a _ ↦ Finset.sum_congr rfl fun b _ ↦ by
        rw [hsymm i b a]; ring
    simp only [mul_add, Finset.sum_add_distrib, hswap, Finset.sum_mul]
    rw [← two_mul, ← mul_assoc]
    norm_num
    exact Finset.sum_congr rfl fun a _ ↦ Finset.sum_congr rfl fun b _ ↦ by ring
  -- The revenue part of the field, paired with `u`, is the own-appeal term.
  have hfield : ∑ a, (∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.kernel l a) *
      u (i, a) = ∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.appeal u i l := by
    simp only [Finset.sum_mul, Nodes.appeal, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun l _ ↦ Finset.sum_congr rfl fun a _ ↦ by ring
  have hown : ownAdjustmentEffect N c H x u i =
      -(∑ a, c i a * u (i, a)) - ∑ a, (∑ b, H i a b * x (i, b)) * u (i, a) +
        ∑ l, N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.appeal u i l := by
    rw [ownAdjustmentEffect, ← hfield]
    simp only [pseudoGradient_apply, sub_mul, add_mul, Finset.sum_sub_distrib,
      Finset.sum_add_distrib]
    ring
  rw [hsymsum, hown, rivalValueEffect]
  have hnode : ∀ l, N.revenue l * (N.appeal u i l * N.denom x l - N.appeal x i l * dirSum N u l) /
      N.denom x l ^ 2 = N.revenue l * N.rival x i l / N.denom x l ^ 2 * N.appeal u i l -
        N.revenue l * N.appeal x i l / N.denom x l ^ 2 * (dirSum N u l - N.appeal u i l) := by
    intro l
    rw [Nodes.rival]
    field_simp
    ring
  simp only [hnode, Finset.sum_sub_distrib]
  ring

/-- Cash flow along a joint direction `(u, δ)` of capacities and opportunity values. At revenue
`N.revenue + t δ`, cash flow is `payoff (x + t u) + t Σ_l δ_l X_il (x + t u)`
(`payoff_withRevenue_sub`). -/
noncomputable def cashFlowLine (u : Profile ι α) (δ : κ → ℝ) (i : ι) (t : ℝ) : ℝ :=
  payoff N c H i (x + t • u) + t * ∑ l, δ l * N.share (x + t • u) i l

/-- The joint line agrees with cash flow at the moved opportunity values. -/
theorem cashFlowLine_eq_payoff (u : Profile ι α) (δ : κ → ℝ) (i : ι) (t : ℝ)
    (hr : ∀ l, 0 ≤ N.revenue l + t * δ l) :
    cashFlowLine N c H x u δ i t =
      payoff (N.withRevenue (fun l ↦ N.revenue l + t * δ l) hr) c H i (x + t • u) := by
  rw [cashFlowLine, eq_comm, ← sub_eq_iff_eq_add', payoff_withRevenue_sub]
  simp only [add_sub_cancel_left, Finset.mul_sum]
  exact Finset.sum_congr rfl fun l _ ↦ by ring

/-- **The envelope split, in general.** Cash flow along `(u, δ)` changes at the rate of direct
opportunity news, plus the own-adjustment effect, plus the rival value effect. -/
@[blueprint "eq:envelope"
  (statement := /--
    Cash flow along a joint capacity and opportunity direction changes by direct news, the
    own-adjustment term, and the rival value effect. Assumptions: Symmetric cost curvature and
    positive share denominators.
  -/)]
theorem hasDerivAt_cashFlowLine (hsymm : ∀ i a b, H i a b = H i b a)
    (hx : ∀ l, 0 < N.denom x l) (u : Profile ι α) (δ : κ → ℝ) (i : ι) :
    HasDerivAt (cashFlowLine N c H x u δ i)
      ((∑ l, δ l * N.share x i l) + ownAdjustmentEffect N c H x u i +
        rivalValueEffect N x u i) 0 := by
  have hshare : HasDerivAt (fun t : ℝ ↦ ∑ l, δ l * N.share (x + t • u) i l)
      (∑ l, δ l * ((N.appeal u i l * N.denom x l - N.appeal x i l * dirSum N u l) /
        N.denom x l ^ 2)) 0 := by
    refine HasDerivAt.fun_sum fun l _ ↦ HasDerivAt.const_mul _ ?_
    have hfun : (fun t : ℝ ↦ N.share (x + t • u) i l) =
        fun t ↦ (N.appeal x i l + t * N.appeal u i l) / (N.denom x l + t * dirSum N u l) := by
      funext t; simp only [Nodes.share, appeal_add_smul, denom_add_smul]
    rw [hfun]
    have h1 : HasDerivAt (fun t : ℝ ↦ N.appeal x i l + t * N.appeal u i l) (N.appeal u i l) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (N.appeal u i l)).const_add (N.appeal x i l)
    have h2 : HasDerivAt (fun t : ℝ ↦ N.denom x l + t * dirSum N u l) (dirSum N u l) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (dirSum N u l)).const_add (N.denom x l)
    have hne : N.denom x l + 0 * dirSum N u l ≠ 0 := by simpa using (hx l).ne'
    refine (h1.div h2 hne).congr_deriv ?_
    simp only [zero_mul, add_zero]
  have hprod := (hasDerivAt_id (0 : ℝ)).mul hshare
  have hpay := hasDerivAt_payoff_line N c H x hsymm hx i u
  refine (hpay.add hprod).congr_deriv ?_
  simp only [id, zero_mul, add_zero, one_mul, Nodes.share, zero_smul]
  ring

/-- **Envelope split at a static equilibrium.** If the field vanishes wherever the direction moves
the firm's own capacity, the own term drops out, and cash flow changes by direct opportunity news
plus the value effect of rivals' movement. -/
@[blueprint "rem:envelope"
  (statement := /--
    At a static equilibrium the own term vanishes, leaving direct opportunity news plus the
    rival value effect. Assumptions: Symmetric cost curvature, positive share denominators, and
    a field that vanishes wherever the direction moves the firm's capacity.
  -/)]
theorem hasDerivAt_cashFlowLine_static (hsymm : ∀ i a b, H i a b = H i b a)
    (hx : ∀ l, 0 < N.denom x l) (u : Profile ι α) (δ : κ → ℝ) (i : ι)
    (hstat : ∀ a, u (i, a) ≠ 0 → pseudoGradient N c H x (i, a) = 0) :
    HasDerivAt (cashFlowLine N c H x u δ i)
      ((∑ l, δ l * N.share x i l) + rivalValueEffect N x u i) 0 := by
  have hown : ownAdjustmentEffect N c H x u i = 0 := by
    rw [ownAdjustmentEffect, neg_eq_zero]
    refine Finset.sum_eq_zero fun a _ ↦ ?_
    by_cases hu : u (i, a) = 0
    · rw [hu, mul_zero]
    · rw [hstat a hu, zero_mul]
  simpa [hown] using hasDerivAt_cashFlowLine N c H x hsymm hx u δ i

/-- **The envelope split fails under anchored adjustment.** At an anchored stage equilibrium the
field equals `-kappa (q - a)` wherever the direction moves the firm, so the own term is
`kappa ⟪q_i - a_i, u_i⟫`. It is not zero unless the firm sits at its anchor or moves orthogonally
to its displacement. -/
theorem hasDerivAt_cashFlowLine_anchored (hsymm : ∀ i a b, H i a b = H i b a)
    (hx : ∀ l, 0 < N.denom x l) (u : Profile ι α) (δ : κ → ℝ) (i : ι) {kappa : ℝ}
    (anchor : Profile ι α)
    (hstage : ∀ a, u (i, a) ≠ 0 →
      pseudoGradient N c H x (i, a) = -(kappa * (x (i, a) - anchor (i, a)))) :
    HasDerivAt (cashFlowLine N c H x u δ i)
      ((∑ l, δ l * N.share x i l) + kappa * ∑ a, (x (i, a) - anchor (i, a)) * u (i, a) +
        rivalValueEffect N x u i) 0 := by
  have hown : ownAdjustmentEffect N c H x u i =
      kappa * ∑ a, (x (i, a) - anchor (i, a)) * u (i, a) := by
    rw [ownAdjustmentEffect, Finset.mul_sum, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun a _ ↦ ?_
    by_cases hu : u (i, a) = 0
    · rw [hu]; ring
    · rw [hstage a hu]; ring
  simpa [hown] using hasDerivAt_cashFlowLine N c H x hsymm hx u δ i

end PricingPerspective.Operating.Valuation.CashFlowResponse
