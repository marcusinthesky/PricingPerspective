import PricingPerspective.Transmission.Envelope

set_option linter.style.longLine false

open Finset
open scoped InnerProductSpace

/-!
# Transfer from observed characteristics to latent exposures

The envelope results of `Envelope.lean` live on the *exposure* clouds, which are
latent. What the pipeline observes is the *characteristic* cloud `Cᵢ` — article
embeddings on the unit sphere. This module carries transport costs across the
map `T` under

  **Assumption (bi-Lipschitz common map).**
  `L⁻¹ d(ω, ω′) ≤ ‖T ω − T ω′‖ ≤ L d(ω, ω′)`,

and absorbs departures from an exact common map into a slack budget
`Bᵢ = T(ξᵢ) + δᵢ`, `‖δᵢ‖ ≤ τᵢ`.


## Why this is cheap here and expensive in the measure-theoretic setting

`T` acts on **support points**, so the pushforward of a coupling is the very same
weight matrix `w`. Both transfer directions are therefore termwise inequalities
on a fixed `π` — no pullback of couplings through `T⁻¹`, hence no Lusin–Souslin
measurable-inverse argument.  The corresponding `P₂(H)`-valued reverse theorem
is now `WassersteinGeometry.wassersteinDistance_le_antilipschitz_map`; this
finite-support layer remains independent of it.

Note where injectivity is actually used: only the **lower** Lipschitz bound is
needed for the reverse transfer, and it is what makes `T` injective
(`map_injective_of_lower_lipschitz`). Injectivity buys separation
(`Cᵢ ≠ Cⱼ ⟹ Pᵢ ≠ Pⱼ`); it does not by itself buy a useful covariance ceiling,
because the ceiling's width is set by `L` and the slack budget, not by mere
distinguishability.

## Main results

* `transportCost_le_of_upper_lipschitz` — `cost_T ≤ L² · cost_d`.
* `transportCost_ge_of_lower_lipschitz` — `L⁻² · cost_d ≤ cost_T`.
* `systCov_slack_bound` — slack perturbs systematic covariance by at most
  `τ' 𝔼‖Tξ‖ + τ 𝔼‖Tζ‖ + τ τ'`.

The metric-level slack statement `|W₂(P, Q) − W₂(Q_T, Q_T′)| ≤ τ + τ′` is the
`W₂` triangle inequality, already proved measure-theoretically as
`WassersteinGeometry.wasserstein_triangle`; it is not re-derived here.

## References

* Memo, Assumptions 1–2 and Proposition 1 (bi-Lipschitz transfer).
* Paper 1 `methodology.tex`, `ass:axiom1`/`ass:axiom2` — the distribution-level
  premise this module *derives* rather than assumes.
-/

namespace PricingPerspective.RandomExposure
variable {ι ι' : Type*} [Fintype ι] [Fintype ι']
variable {Ω : Type*} [MetricSpace Ω]
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
variable {p : ι → ℝ} {q : ι' → ℝ}

/-- Expected squared ground distance between characteristic clouds under a
    coupling. For `L²`-normalized embeddings this is the quantity the pipeline's
    exact balanced transport solver already returns. -/
noncomputable def groundCost (π : Coupling p q) (ξ : ι → Ω) (ζ : ι' → Ω) : ℝ :=
  ∑ a, ∑ b, π.w a b * dist (ξ a) (ζ b) ^ 2

/-! ### Bi-Lipschitz transfer -/

omit [InnerProductSpace ℝ H] in
/-- **Upper transfer.** An upper-Lipschitz common map sends characteristic
    transport cost to an upper bound on exposure transport cost, coupling by
    coupling.

    Combined with `systCov_eq_of_transportCost` this is the direction that yields
    a *floor* on systematic covariance — the conclusion Paper 1 already reports,
    now derived from a pointwise premise on `Ω` instead of an unfalsifiable
    Lipschitz condition on an operator over `𝒫₁(Ω)`. -/
theorem transportCost_le_of_upper_lipschitz
    (π : Coupling p q) (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H) {L : ℝ}
    (hup : ∀ x y, ‖T x - T y‖ ≤ L * dist x y) :
    transportCost π (fun a => T (ξ a)) (fun b => T (ζ b)) ≤ L ^ 2 * groundCost π ξ ζ := by
  unfold transportCost groundCost
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun b _ => ?_
  have hsq : ‖T (ξ a) - T (ζ b)‖ ^ 2 ≤ L ^ 2 * dist (ξ a) (ζ b) ^ 2 := by
    have h := hup (ξ a) (ζ b)
    have := pow_le_pow_left₀ (norm_nonneg _) h 2
    calc ‖T (ξ a) - T (ζ b)‖ ^ 2 ≤ (L * dist (ξ a) (ζ b)) ^ 2 := this
      _ = L ^ 2 * dist (ξ a) (ζ b) ^ 2 := by ring
  calc π.w a b * ‖T (ξ a) - T (ζ b)‖ ^ 2
      ≤ π.w a b * (L ^ 2 * dist (ξ a) (ζ b) ^ 2) :=
        mul_le_mul_of_nonneg_left hsq (π.nonneg a b)
    _ = L ^ 2 * (π.w a b * dist (ξ a) (ζ b) ^ 2) := by ring

omit [InnerProductSpace ℝ H] in
/-- **Lower transfer.** The lower-Lipschitz half of the bi-Lipschitz assumption
    bounds exposure transport cost from below.

    This is the direction that would deliver a covariance *ceiling*, and it is
    the direction energy distance and MMD cannot supply at all. Whether the
    resulting ceiling is informative is a separate, quantitative question settled
    by `Certificate.ceiling_trivial_of_slack_ge`. -/
theorem transportCost_ge_of_lower_lipschitz
    (π : Coupling p q) (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H) {L : ℝ} (hL : 0 < L)
    (hlo : ∀ x y, L⁻¹ * dist x y ≤ ‖T x - T y‖) :
    (L⁻¹) ^ 2 * groundCost π ξ ζ ≤ transportCost π (fun a => T (ξ a)) (fun b => T (ζ b)) := by
  unfold transportCost groundCost
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun b _ => ?_
  have hLinv : (0:ℝ) ≤ L⁻¹ := le_of_lt (inv_pos.mpr hL)
  have hsq : (L⁻¹) ^ 2 * dist (ξ a) (ζ b) ^ 2 ≤ ‖T (ξ a) - T (ζ b)‖ ^ 2 := by
    have h := hlo (ξ a) (ζ b)
    have hnn : (0:ℝ) ≤ L⁻¹ * dist (ξ a) (ζ b) := mul_nonneg hLinv dist_nonneg
    calc (L⁻¹) ^ 2 * dist (ξ a) (ζ b) ^ 2 = (L⁻¹ * dist (ξ a) (ζ b)) ^ 2 := by ring
      _ ≤ ‖T (ξ a) - T (ζ b)‖ ^ 2 := pow_le_pow_left₀ hnn h 2
  calc (L⁻¹) ^ 2 * (π.w a b * dist (ξ a) (ζ b) ^ 2)
      = π.w a b * ((L⁻¹) ^ 2 * dist (ξ a) (ζ b) ^ 2) := by ring
    _ ≤ π.w a b * ‖T (ξ a) - T (ζ b)‖ ^ 2 :=
        mul_le_mul_of_nonneg_left hsq (π.nonneg a b)

omit [InnerProductSpace ℝ H] in
/-- Optimized upper transport transfer through a common Lipschitz map. -/
theorem optimized_transportCost_le_of_upper_lipschitz
    (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H) {L : ℝ}
    (ground : FiniteOptimalCostCertificate (fun π : Coupling p q => groundCost π ξ ζ))
    (exposure :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q => transportCost π (fun a => T (ξ a)) (fun b => T (ζ b))))
    (hup : ∀ x y, ‖T x - T y‖ ≤ L * dist x y) :
    exposure.value ≤ L ^ 2 * ground.value := by
  calc
    exposure.value ≤
        transportCost ground.plan (fun a => T (ξ a)) (fun b => T (ζ b)) :=
      exposure.minimal ground.plan
    _ ≤ L ^ 2 * groundCost ground.plan ξ ζ :=
      transportCost_le_of_upper_lipschitz ground.plan ξ ζ T hup
    _ = L ^ 2 * ground.value := by rw [ground.plan_cost]

omit [InnerProductSpace ℝ H] in
/-- Optimized lower transport transfer through a common bi-Lipschitz map. -/
theorem optimized_transportCost_ge_of_lower_lipschitz
    (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H) {L : ℝ}
    (ground : FiniteOptimalCostCertificate (fun π : Coupling p q => groundCost π ξ ζ))
    (exposure :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q => transportCost π (fun a => T (ξ a)) (fun b => T (ζ b))))
    (hL : 0 < L)
    (hlo : ∀ x y, L⁻¹ * dist x y ≤ ‖T x - T y‖) :
    (L⁻¹) ^ 2 * ground.value ≤ exposure.value := by
  calc
    (L⁻¹) ^ 2 * ground.value ≤
        (L⁻¹) ^ 2 * groundCost exposure.plan ξ ζ := by
      exact mul_le_mul_of_nonneg_left (ground.minimal exposure.plan) (sq_nonneg _)
    _ ≤ transportCost exposure.plan (fun a => T (ξ a)) (fun b => T (ζ b)) :=
      transportCost_ge_of_lower_lipschitz exposure.plan ξ ζ T hL hlo
    _ = exposure.value := exposure.plan_cost


omit [InnerProductSpace ℝ H] in
/-- A positive lower-Lipschitz constant makes a map injective on a metric space. -/
theorem map_injective_of_lower_lipschitz
    (T : Ω → H) {L : ℝ} (hL : 0 < L)
    (hlo : ∀ x y, L⁻¹ * dist x y ≤ ‖T x - T y‖) :
    Function.Injective T := by
  intro x y hxy
  have hLinv : 0 < L⁻¹ := inv_pos.mpr hL
  have hdist : L⁻¹ * dist x y ≤ 0 := by
    have h := hlo x y
    rw [hxy, sub_self, norm_zero] at h
    exact h
  have hdist_nonpos : dist x y ≤ 0 := by
    by_contra hpos
    have hpos' : 0 < dist x y := lt_of_not_ge hpos
    have hprod : 0 < L⁻¹ * dist x y := mul_pos hLinv hpos'
    exact (not_lt_of_ge hdist) hprod
  have hzero : dist x y = 0 := le_antisymm hdist_nonpos dist_nonneg
  exact dist_eq_zero.mp hzero



/-! ### Stochastic slack -/

/-- **Slack perturbation bound on systematic covariance.**

    With `Bₐ = T(ξₐ) + δₐ`, `‖δₐ‖ ≤ τ`, and `B′_b = T(ζ_b) + δ′_b`, `‖δ′_b‖ ≤ τ′`,
    the systematic covariance moves by at most

      `τ′ · 𝔼‖T ξ‖ + τ · 𝔼‖T ζ‖ + τ τ′`

    (total coupling mass normalized to one).

    This is stated on `systCov` rather than on `W₂` deliberately: `systCov` is the
    object the papers report, and the bound then reads directly as *"slack must be
    small relative to exposure scale."* It is also the precise sense in which the
    model degrades continuously where a common-map assumption fails outright —
    leverage, hedging and media selection move `τ`, not the theorem's validity. -/
theorem systCov_slack_bound
    (π : Coupling p q) (Tξ : ι → H) (Tζ : ι' → H) (δ : ι → H) (δ' : ι' → H)
    {τ τ' : ℝ}
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ') (hτ0 : 0 ≤ τ) :
    |systCov π (fun a => Tξ a + δ a) (fun b => Tζ b + δ' b) - systCov π Tξ Tζ|
      ≤ ∑ a, ∑ b, π.w a b * (τ' * ‖Tξ a‖ + τ * ‖Tζ b‖ + τ * τ') := by
  have hnn : ∀ a b, 0 ≤ π.w a b := π.nonneg
  have expand : ∀ a b,
      ⟪Tξ a + δ a, Tζ b + δ' b⟫_ℝ - ⟪Tξ a, Tζ b⟫_ℝ
        = ⟪Tξ a, δ' b⟫_ℝ + ⟪δ a, Tζ b⟫_ℝ + ⟪δ a, δ' b⟫_ℝ := by
    intro a b
    rw [inner_add_left, inner_add_right, inner_add_right]
    ring
  have bound : ∀ a b,
      |⟪Tξ a + δ a, Tζ b + δ' b⟫_ℝ - ⟪Tξ a, Tζ b⟫_ℝ|
        ≤ τ' * ‖Tξ a‖ + τ * ‖Tζ b‖ + τ * τ' := by
    intro a b
    rw [expand a b]
    have h₁ : |⟪Tξ a, δ' b⟫_ℝ| ≤ ‖Tξ a‖ * τ' :=
      le_trans (abs_real_inner_le_norm _ _)
        (mul_le_mul_of_nonneg_left (hτ' b) (norm_nonneg _))
    have h₂ : |⟪δ a, Tζ b⟫_ℝ| ≤ τ * ‖Tζ b‖ :=
      le_trans (abs_real_inner_le_norm _ _)
        (mul_le_mul_of_nonneg_right (hτ a) (norm_nonneg _))
    have h₃ : |⟪δ a, δ' b⟫_ℝ| ≤ τ * τ' := by
      refine le_trans (abs_real_inner_le_norm _ _) ?_
      exact mul_le_mul (hτ a) (hτ' b) (norm_nonneg _) hτ0
    calc |⟪Tξ a, δ' b⟫_ℝ + ⟪δ a, Tζ b⟫_ℝ + ⟪δ a, δ' b⟫_ℝ|
        ≤ |⟪Tξ a, δ' b⟫_ℝ + ⟪δ a, Tζ b⟫_ℝ| + |⟪δ a, δ' b⟫_ℝ| := abs_add_le _ _
      _ ≤ (|⟪Tξ a, δ' b⟫_ℝ| + |⟪δ a, Tζ b⟫_ℝ|) + |⟪δ a, δ' b⟫_ℝ| := by
          linarith [abs_add_le (⟪Tξ a, δ' b⟫_ℝ) (⟪δ a, Tζ b⟫_ℝ)]
      _ ≤ (‖Tξ a‖ * τ' + τ * ‖Tζ b‖) + τ * τ' := by
          exact add_le_add (add_le_add h₁ h₂) h₃
      _ = τ' * ‖Tξ a‖ + τ * ‖Tζ b‖ + τ * τ' := by ring
  have diff : systCov π (fun a => Tξ a + δ a) (fun b => Tζ b + δ' b) - systCov π Tξ Tζ
      = ∑ a, ∑ b, π.w a b *
          (⟪Tξ a + δ a, Tζ b + δ' b⟫_ℝ - ⟪Tξ a, Tζ b⟫_ℝ) := by
    unfold systCov
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun b _ => by ring
  rw [diff]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  refine Finset.sum_le_sum fun a _ => ?_
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  refine Finset.sum_le_sum fun b _ => ?_
  rw [abs_mul, abs_of_nonneg (hnn a b)]
  exact mul_le_mul_of_nonneg_left (bound a b) (hnn a b)

/-- Mean exposure scale `𝔼‖Z‖` of a weighted cloud — the quantity the slack
    budget must be small relative to. -/
noncomputable def firstAbsMoment (p : ι → ℝ) (Z : ι → H) : ℝ :=
  ∑ a, p a * ‖Z a‖

/-- A bounded loading slack perturbs the marginal second moment. -/
theorem secondMoment_slack_bound
    (p : ι → ℝ) (Tξ : ι → H) (δ : ι → H) {τ : ℝ}
    (hp : ∀ a, 0 ≤ p a) (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ0 : 0 ≤ τ) :
    |secondMoment p (fun a => Tξ a + δ a) - secondMoment p Tξ|
      ≤ 2 * τ * firstAbsMoment p Tξ + τ ^ 2 := by
  have hpoint : ∀ a,
      |‖Tξ a + δ a‖ ^ 2 - ‖Tξ a‖ ^ 2|
        ≤ 2 * τ * ‖Tξ a‖ + τ ^ 2 := by
    intro a
    have habs : |⟪Tξ a, δ a⟫_ℝ| ≤ ‖Tξ a‖ * τ :=
      le_trans (abs_real_inner_le_norm _ _)
        (mul_le_mul_of_nonneg_left (hτ a) (norm_nonneg _))
    have hinner := (abs_le.mp habs)
    have hδsq : ‖δ a‖ ^ 2 ≤ τ ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) hτ0).mpr (hτ a)
    rw [norm_add_sq_real]
    rw [abs_le]
    constructor <;> nlinarith [hinner.1, hinner.2, hδsq]
  have hdiff :
      secondMoment p (fun a => Tξ a + δ a) - secondMoment p Tξ =
        ∑ a, p a * (‖Tξ a + δ a‖ ^ 2 - ‖Tξ a‖ ^ 2) := by
    unfold secondMoment
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun a _ => by ring
  rw [hdiff]
  calc
    |∑ a, p a * (‖Tξ a + δ a‖ ^ 2 - ‖Tξ a‖ ^ 2)|
        ≤ ∑ a, |p a * (‖Tξ a + δ a‖ ^ 2 - ‖Tξ a‖ ^ 2)| := by
          exact Finset.abs_sum_le_sum_abs
            (fun a => p a * (‖Tξ a + δ a‖ ^ 2 - ‖Tξ a‖ ^ 2)) Finset.univ
    _ ≤ ∑ a, p a * (2 * τ * ‖Tξ a‖ + τ ^ 2) := by
      refine Finset.sum_le_sum fun a _ => ?_
      rw [abs_mul, abs_of_nonneg (hp a)]
      exact mul_le_mul_of_nonneg_left (hpoint a) (hp a)
    _ = 2 * τ * firstAbsMoment p Tξ + τ ^ 2 := by
      unfold firstAbsMoment
      rw [show (∑ a, p a * (2 * τ * ‖Tξ a‖ + τ ^ 2)) =
        (∑ a, (2 * τ) * (p a * ‖Tξ a‖)) + ∑ a, p a * τ ^ 2 by
          simp_rw [mul_add]
          rw [Finset.sum_add_distrib]
          congr 1
          exact Finset.sum_congr rfl fun a _ => by ring
      ]
      rw [(Finset.mul_sum (Finset.univ : Finset ι)
        (fun a => p a * ‖Tξ a‖) (2 * τ)).symm, ← Finset.sum_mul, hmass]
      ring

omit [InnerProductSpace ℝ H] in
/-- Fixed-coupling squared-cost perturbation under bounded loading slack. -/
theorem transportCost_le_of_slack
    (π : Coupling p q) (U : ι → H) (V : ι' → H)
    (δ : ι → H) (δ' : ι' → H) {τ τ' D : ℝ}
    (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ')
    (hτ0 : 0 ≤ τ) (hτ'0 : 0 ≤ τ')
    (hD : ∀ a b, ‖U a - V b‖ ≤ D) :
    transportCost π (fun a => U a + δ a) (fun b => V b + δ' b)
      ≤ transportCost π U V + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
  have hs0 : 0 ≤ τ + τ' := add_nonneg hτ0 hτ'0
  have hpoint : ∀ a b,
      ‖(U a + δ a) - (V b + δ' b)‖ ^ 2
        ≤ ‖U a - V b‖ ^ 2 + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    intro a b
    have hδ : ‖δ a - δ' b‖ ≤ τ + τ' := by
      calc
        ‖δ a - δ' b‖ ≤ ‖δ a‖ + ‖δ' b‖ := norm_sub_le _ _
        _ ≤ τ + τ' := add_le_add (hτ a) (hτ' b)
    have hdecomp :
        (U a + δ a) - (V b + δ' b) = (U a - V b) + (δ a - δ' b) := by
      abel
    have hnorm :
        ‖(U a + δ a) - (V b + δ' b)‖ ≤ ‖U a - V b‖ + (τ + τ') := by
      rw [hdecomp]
      exact le_trans (norm_add_le _ _) (add_le_add_right hδ _)
    have hsquare :
        ‖(U a + δ a) - (V b + δ' b)‖ ^ 2
          ≤ (‖U a - V b‖ + (τ + τ')) ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) (add_nonneg (norm_nonneg _) hs0)).mpr hnorm
    have hcross :
        (‖U a - V b‖ + (τ + τ')) ^ 2
          ≤ ‖U a - V b‖ ^ 2 + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
      calc
        (‖U a - V b‖ + (τ + τ')) ^ 2 =
            ‖U a - V b‖ ^ 2 + 2 * (τ + τ') * ‖U a - V b‖
              + (τ + τ') ^ 2 := by ring
        _ ≤ ‖U a - V b‖ ^ 2 + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
          have hmul : 2 * (τ + τ') * ‖U a - V b‖ ≤
              2 * (τ + τ') * D :=
            mul_le_mul_of_nonneg_left (hD a b) (by positivity)
          linarith
    calc
      ‖(U a + δ a) - (V b + δ' b)‖ ^ 2
          ≤ (‖U a - V b‖ + (τ + τ')) ^ 2 := hsquare
      _ ≤ ‖U a - V b‖ ^ 2 + 2 * (τ + τ') * D + (τ + τ') ^ 2 := hcross
  have htotal : ∑ a, ∑ b, π.w a b = 1 := by
    calc
      ∑ a, ∑ b, π.w a b = ∑ a, p a := by
        exact Finset.sum_congr rfl fun a _ => π.marginal_fst a
      _ = 1 := hmass
  unfold transportCost
  calc
    ∑ a, ∑ b, π.w a b * ‖(U a + δ a) - (V b + δ' b)‖ ^ 2
        ≤ ∑ a, ∑ b, π.w a b *
            (‖U a - V b‖ ^ 2 + 2 * (τ + τ') * D + (τ + τ') ^ 2) := by
          refine Finset.sum_le_sum fun a _ => ?_
          refine Finset.sum_le_sum fun b _ => ?_
          exact mul_le_mul_of_nonneg_left (hpoint a b) (π.nonneg a b)
    _ = ∑ a, ∑ b, π.w a b * ‖U a - V b‖ ^ 2
          + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
          let K : ℝ := 2 * (τ + τ') * D + (τ + τ') ^ 2
          have hK : ∑ a, ∑ b, π.w a b * K = K := by
            calc
              ∑ a, ∑ b, π.w a b * K =
                  ∑ a, (∑ b, π.w a b) * K := by
                refine Finset.sum_congr rfl fun a _ => ?_
                exact (Finset.sum_mul (Finset.univ : Finset ι')
                  (fun b => π.w a b) K).symm
              _ = (∑ a, ∑ b, π.w a b) * K := by
                exact (Finset.sum_mul (Finset.univ : Finset ι)
                  (fun a => ∑ b, π.w a b) K).symm
              _ = K := by rw [htotal]; ring
          have hexpand : ∀ a b,
              π.w a b * (‖U a - V b‖ ^ 2 + K) =
                π.w a b * ‖U a - V b‖ ^ 2 + π.w a b * K := by
            intro a b
            ring
          have hsum_add :
              (∑ a, ∑ b, (π.w a b * ‖U a - V b‖ ^ 2 + π.w a b * K)) =
                (∑ a, ∑ b, π.w a b * ‖U a - V b‖ ^ 2) +
                  (∑ a, ∑ b, π.w a b * K) := by
            calc
              (∑ a, ∑ b, (π.w a b * ‖U a - V b‖ ^ 2 + π.w a b * K)) =
                  ∑ a, ((∑ b, π.w a b * ‖U a - V b‖ ^ 2) +
                    (∑ b, π.w a b * K)) := by
                refine Finset.sum_congr rfl fun a _ => ?_
                exact Finset.sum_add_distrib
              _ = (∑ a, ∑ b, π.w a b * ‖U a - V b‖ ^ 2) +
                    (∑ a, ∑ b, π.w a b * K) :=
                Finset.sum_add_distrib
          have hEq :
              (∑ a, ∑ b, (π.w a b * ‖U a - V b‖ ^ 2 + π.w a b * K)) =
                (∑ a, ∑ b, π.w a b * ‖U a - V b‖ ^ 2) + K := by
            rw [hsum_add, hK]
          simpa [K, add_assoc, mul_add] using hEq

omit [MetricSpace Ω] [InnerProductSpace ℝ H] in
/-- A cloud-level transport bound for one common characteristic-to-exposure map. -/
theorem transportCost_le_of_cloud_slack
    (C : RandomExposureCloud ι Ω H)
    (D : RandomExposureCloud ι' Ω H)
    (π : Coupling C.characteristic.weight D.characteristic.weight)
    (T : Ω → H) {B : ℝ}
    (hB : ∀ a b,
      ‖T (C.characteristic.point a) - T (D.characteristic.point b)‖ ≤ B) :
    transportCost π (C.loading T) (D.loading T)
      ≤ transportCost π
          (fun a => T (C.characteristic.point a))
          (fun b => T (D.characteristic.point b))
        + 2 * (C.radius + D.radius) * B
        + (C.radius + D.radius) ^ 2 := by
  change transportCost π
      (fun a => T (C.characteristic.point a) + C.slack a)
      (fun b => T (D.characteristic.point b) + D.slack b)
    ≤ transportCost π
        (fun a => T (C.characteristic.point a))
        (fun b => T (D.characteristic.point b))
      + 2 * (C.radius + D.radius) * B
      + (C.radius + D.radius) ^ 2
  exact transportCost_le_of_slack
    π
    (fun a => T (C.characteristic.point a))
    (fun b => T (D.characteristic.point b))
    C.slack D.slack
    C.characteristic.mass_one
    C.slack_bound D.slack_bound
    C.radius_nonneg D.radius_nonneg
    hB

omit [InnerProductSpace ℝ H] in
/-- Optimized squared-cost perturbation under one common slack budget. -/
theorem optimized_transportCost_abs_sub_of_slack
    (U : ι → H) (V : ι' → H) (δ : ι → H) (δ' : ι' → H)
    {τ τ' D : ℝ}
    (base :
      FiniteOptimalCostCertificate (fun π : Coupling p q => transportCost π U V))
    (exposure :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q =>
          transportCost π (fun a => U a + δ a) (fun b => V b + δ' b)))
    (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ')
    (hτ0 : 0 ≤ τ) (hτ'0 : 0 ≤ τ')
    (hD : ∀ a b, ‖U a - V b‖ ≤ D)
    (hD' : ∀ a b, ‖(U a + δ a) - (V b + δ' b)‖ ≤ D) :
    |exposure.value - base.value| ≤ 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
  have hupper := transportCost_le_of_slack base.plan U V δ δ' hmass hτ hτ'
    hτ0 hτ'0 hD
  have hupper_value :
      exposure.value ≤ base.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    calc
      exposure.value ≤
          transportCost base.plan (fun a => U a + δ a) (fun b => V b + δ' b) :=
        exposure.minimal base.plan
      _ ≤ transportCost base.plan U V + 2 * (τ + τ') * D + (τ + τ') ^ 2 := hupper
      _ = base.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
        rw [base.plan_cost]
  have hlower_raw := transportCost_le_of_slack exposure.plan
    (fun a => U a + δ a) (fun b => V b + δ' b)
    (fun a => -δ a) (fun b => -δ' b) hmass
    (fun a => by simpa using hτ a) (fun b => by simpa using hτ' b)
    hτ0 hτ'0 hD'
  have hlower :
      transportCost exposure.plan U V ≤
        transportCost exposure.plan (fun a => U a + δ a) (fun b => V b + δ' b)
          + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    simpa [add_assoc] using hlower_raw
  have hlower_value :
      base.value ≤ exposure.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    calc
      base.value ≤ transportCost exposure.plan U V := base.minimal exposure.plan
      _ ≤ transportCost exposure.plan (fun a => U a + δ a) (fun b => V b + δ' b)
          + 2 * (τ + τ') * D + (τ + τ') ^ 2 := hlower
      _ = exposure.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
        rw [exposure.plan_cost]
  rw [abs_le]
  constructor <;> linarith


omit [InnerProductSpace ℝ H] in
/-- Optimized upper transfer with bounded loading slack. -/
theorem optimized_transportCost_le_of_upper_lipschitz_slack
    (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H)
    (δ : ι → H) (δ' : ι' → H) {L τ τ' D : ℝ}
    (ground : FiniteOptimalCostCertificate (fun π : Coupling p q => groundCost π ξ ζ))
    (mapped :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q => transportCost π (fun a => T (ξ a)) (fun b => T (ζ b))))
    (exposure :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q =>
          transportCost π (fun a => T (ξ a) + δ a) (fun b => T (ζ b) + δ' b)))
    (hup : ∀ x y, ‖T x - T y‖ ≤ L * dist x y)
    (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ')
    (hτ0 : 0 ≤ τ) (hτ'0 : 0 ≤ τ')
    (hD : ∀ a b, ‖T (ξ a) - T (ζ b)‖ ≤ D)
    (hD' : ∀ a b,
      ‖(T (ξ a) + δ a) - (T (ζ b) + δ' b)‖ ≤ D) :
    exposure.value ≤ L ^ 2 * ground.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
  have hmap := optimized_transportCost_le_of_upper_lipschitz ξ ζ T ground mapped hup
  have hslack := optimized_transportCost_abs_sub_of_slack
    (fun a => T (ξ a)) (fun b => T (ζ b)) δ δ' mapped exposure
    hmass hτ hτ' hτ0 hτ'0 hD hD'
  have hclose :
      exposure.value ≤ mapped.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    have := le_trans (le_abs_self (exposure.value - mapped.value)) hslack
    linarith
  linarith

omit [InnerProductSpace ℝ H] in
/-- Optimized lower transfer with bounded loading slack. -/
theorem optimized_transportCost_ge_of_lower_lipschitz_slack
    (ξ : ι → Ω) (ζ : ι' → Ω) (T : Ω → H)
    (δ : ι → H) (δ' : ι' → H) {L τ τ' D : ℝ}
    (ground : FiniteOptimalCostCertificate (fun π : Coupling p q => groundCost π ξ ζ))
    (mapped :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q => transportCost π (fun a => T (ξ a)) (fun b => T (ζ b))))
    (exposure :
      FiniteOptimalCostCertificate
        (fun π : Coupling p q =>
          transportCost π (fun a => T (ξ a) + δ a) (fun b => T (ζ b) + δ' b)))
    (hL : 0 < L)
    (hlo : ∀ x y, L⁻¹ * dist x y ≤ ‖T x - T y‖)
    (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ')
    (hτ0 : 0 ≤ τ) (hτ'0 : 0 ≤ τ')
    (hD : ∀ a b, ‖T (ξ a) - T (ζ b)‖ ≤ D)
    (hD' : ∀ a b,
      ‖(T (ξ a) + δ a) - (T (ζ b) + δ' b)‖ ≤ D) :
    (L⁻¹) ^ 2 * ground.value - 2 * (τ + τ') * D - (τ + τ') ^ 2
      ≤ exposure.value := by
  have hmap := optimized_transportCost_ge_of_lower_lipschitz
    ξ ζ T ground mapped hL hlo
  have hslack := optimized_transportCost_abs_sub_of_slack
    (fun a => T (ξ a)) (fun b => T (ζ b)) δ δ' mapped exposure
    hmass hτ hτ' hτ0 hτ'0 hD hD'
  have hclose :
      mapped.value ≤ exposure.value + 2 * (τ + τ') * D + (τ + τ') ^ 2 := by
    have := le_trans (neg_le_abs (exposure.value - mapped.value)) hslack
    linarith
  linarith



/-- **Slack bound in closed form.** Under unit coupling mass, the perturbation of
    systematic covariance is at most `τ′ 𝔼‖Tξ‖ + τ 𝔼‖Tζ‖ + τ τ′`.

    Read as a design constraint: the slack budget enters *linearly in the exposure
    scale*, so a `τ` calibrated to absorb leverage and hedging — first-order
    effects — perturbs covariance by a first-order amount. Identifying `τ` is
    therefore not an implementation detail; it is the binding requirement for the
    model to say anything. -/
theorem systCov_slack_bound_closed
    (π : Coupling p q) (Tξ : ι → H) (Tζ : ι' → H) (δ : ι → H) (δ' : ι' → H)
    {τ τ' : ℝ}
    (hmass : ∑ a, p a = 1)
    (hτ : ∀ a, ‖δ a‖ ≤ τ) (hτ' : ∀ b, ‖δ' b‖ ≤ τ') (hτ0 : 0 ≤ τ) :
    |systCov π (fun a => Tξ a + δ a) (fun b => Tζ b + δ' b) - systCov π Tξ Tζ|
      ≤ τ' * firstAbsMoment p Tξ + τ * firstAbsMoment q Tζ + τ * τ' := by
  refine le_trans (systCov_slack_bound π Tξ Tζ δ δ' hτ hτ' hτ0) ?_
  have hsplit : ∀ a, ∑ b, π.w a b * (τ' * ‖Tξ a‖ + τ * ‖Tζ b‖ + τ * τ')
      = p a * (τ' * ‖Tξ a‖) + (∑ b, π.w a b * (τ * ‖Tζ b‖)) + p a * (τ * τ') := by
    intro a
    have expand : ∀ b ∈ (Finset.univ : Finset ι'),
        π.w a b * (τ' * ‖Tξ a‖ + τ * ‖Tζ b‖ + τ * τ')
          = (π.w a b * (τ' * ‖Tξ a‖) + π.w a b * (τ * ‖Tζ b‖)) + π.w a b * (τ * τ') := by
      intro b _; ring
    rw [Finset.sum_congr rfl expand, Finset.sum_add_distrib, Finset.sum_add_distrib,
      ← Finset.sum_mul, ← Finset.sum_mul, π.marginal_fst a]
  rw [Finset.sum_congr rfl fun a _ => hsplit a, Finset.sum_add_distrib,
    Finset.sum_add_distrib, ← Finset.sum_mul, hmass, one_mul]
  have hswap : ∑ a, ∑ b, π.w a b * (τ * ‖Tζ b‖) = τ * firstAbsMoment q Tζ := by
    rw [Finset.sum_comm]
    unfold firstAbsMoment
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [← Finset.sum_mul, π.marginal_snd b]; ring
  have hfirst : ∑ a, p a * (τ' * ‖Tξ a‖) = τ' * firstAbsMoment p Tξ := by
    unfold firstAbsMoment
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => by ring
  rw [hswap, hfirst]

end PricingPerspective.RandomExposure
