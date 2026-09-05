import PricingPerspective.Transmission.Defs

set_option linter.style.longLine false

open Finset
open scoped InnerProductSpace

/-!
# The systematic-covariance envelope

The single identity behind the whole random-exposure model: under any coupling,
expected squared transport cost and systematic covariance are the *same
quantity*, related by polarization,

  `𝔼_π ‖Z − Y‖² = v_P + v_Q − 2 κ_π`.

Minimizing transport cost is therefore literally maximizing systematic
covariance. Two consequences follow immediately and are the reason to prefer a
coupling-based distance over a test-function-based one:

* the covariance ceiling is **attained**, by an optimal transport plan; and
* the shortfall of any particular coupling from that ceiling is exactly half its
  **excess transport cost**, an object that is estimable from paired draws.

Contrast `Continuous/Energy/Core.lean`, where the analogous bound is derived from
a Lipschitz premise `‖βᵢ − βⱼ‖ ≤ L 𝓔` and carries an exogenous, unidentified `L`.
Here the ceiling instead comes with an attaining plan, so no exogenous constant
enters the envelope itself; `L` reappears only in `Transfer.lean`, where the
*observed* characteristic cost is carried across the map.

## Main results

* `transportCost_polarization` — the identity above.
* `systCov_eq_of_transportCost` — `κ_π = (v_P + v_Q − cost_π)/2`.
* `isGreatest_frechetClass` — the ceiling `κ̄` is the greatest admissible
  covariance, given a certified finite minimum transport cost.
* `isLeast_frechetClass` — the reflected problem gives the floor `κ̲`, given a
  certified finite minimum transport cost.
* `systCov_eq_ceiling_sub_excess` — `κ_π = κ̄ − Δ_π/2`, the gap identity.
* `systCov_subsingleton_eq` — deterministic loadings: the class is a singleton
  and the envelope collapses to the vector-model covariance.

## References

* Gelbrich (1990); Puccetti & Wang (2015); Panaretos & Zemel (2020), Ch. 2.
-/

namespace PricingPerspective.RandomExposure

variable {ι ι' : Type*} [Fintype ι] [Fintype ι']
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
variable {p : ι → ℝ} {q : ι' → ℝ}

/-! ### Polarization -/

/-- **Polarization identity for couplings.**

    `𝔼_π ‖Z − Y‖² = v_P + v_Q − 2 κ_π`.

    This is the entire mathematical content of the covariance envelope. It holds
    for *every* coupling — no optimality, no assumption on the marginals, and no
    Lipschitz constant. The margins enter only to collapse the double sums. -/
theorem transportCost_polarization
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) :
    transportCost π Z Y
      = secondMoment p Z + secondMoment q Y - 2 * systCov π Z Y := by
  unfold transportCost secondMoment systCov
  have key : ∀ a, ∑ b, π.w a b * ‖Z a - Y b‖ ^ 2
      = p a * ‖Z a‖ ^ 2 + (∑ b, π.w a b * ‖Y b‖ ^ 2)
        - 2 * ∑ b, π.w a b * ⟪Z a, Y b⟫_ℝ := by
    intro a
    have expand : ∀ b ∈ (Finset.univ : Finset ι'),
        π.w a b * ‖Z a - Y b‖ ^ 2
          = (π.w a b * ‖Z a‖ ^ 2 + π.w a b * ‖Y b‖ ^ 2)
            - 2 * (π.w a b * ⟪Z a, Y b⟫_ℝ) := by
      intro b _
      rw [norm_sub_sq_real]; ring
    rw [Finset.sum_congr rfl expand, Finset.sum_sub_distrib, Finset.sum_add_distrib,
      ← Finset.sum_mul, π.marginal_fst a, ← Finset.mul_sum]
  rw [Finset.sum_congr rfl fun a _ => key a, Finset.sum_sub_distrib,
    Finset.sum_add_distrib, ← Finset.mul_sum]
  have swap : ∑ a, ∑ b, π.w a b * ‖Y b‖ ^ 2 = ∑ b, q b * ‖Y b‖ ^ 2 := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun b _ => by rw [← Finset.sum_mul, π.marginal_snd b]
  rw [swap]

/-- Solved form of `transportCost_polarization`: covariance is determined by the
    transport cost of the coupling that produced it. -/
theorem systCov_eq_of_transportCost
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) :
    systCov π Z Y
      = (secondMoment p Z + secondMoment q Y - transportCost π Z Y) / 2 := by
  rw [transportCost_polarization]; ring

/-! ### The Fréchet class -/

/-- The set of systematic covariances admissible given only the two marginal
    exposure laws. Marginals identify this **set**, not a point; that is the
    identification problem the model makes explicit rather than hiding. -/
def frechetClass (p : ι → ℝ) (q : ι' → ℝ) (Z : ι → H) (Y : ι' → H) : Set ℝ :=
  {c | ∃ π : Coupling p q, systCov π Z Y = c}

/-- Convex mixture of two couplings with the same margins. -/
def Coupling.mix
    (π π' : Coupling p q) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) : Coupling p q :=
  { w := fun a b => t * π.w a b + (1 - t) * π'.w a b
    nonneg := by
      intro a b
      exact add_nonneg
        (mul_nonneg ht0 (π.nonneg a b))
        (mul_nonneg (sub_nonneg.mpr ht1) (π'.nonneg a b))
    marginal_fst := by
      intro a
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
        π.marginal_fst, π'.marginal_fst]
      ring
    marginal_snd := by
      intro b
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
        π.marginal_snd, π'.marginal_snd]
      ring }

/-- Systematic covariance is affine under coupling mixtures. -/
theorem systCov_mix
    (π π' : Coupling p q) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (Z : ι → H) (Y : ι' → H) :
    systCov (π.mix π' t ht0 ht1) Z Y
      = t * systCov π Z Y + (1 - t) * systCov π' Z Y := by
  change
    (∑ a, ∑ b, (t * π.w a b + (1 - t) * π'.w a b) * ⟪Z a, Y b⟫_ℝ)
      = t * ∑ a, ∑ b, π.w a b * ⟪Z a, Y b⟫_ℝ
        + (1 - t) * ∑ a, ∑ b, π'.w a b * ⟪Z a, Y b⟫_ℝ
  simp only [add_mul, Finset.sum_add_distrib]
  have factor : ∀ (c : ℝ) (f : ι → ι' → ℝ),
      (∑ a, ∑ b, c * f a b) = c * ∑ a, ∑ b, f a b := by
    intro c f
    calc
      (∑ a, ∑ b, c * f a b) = ∑ a, c * ∑ b, f a b := by
        refine Finset.sum_congr rfl fun a _ => ?_
        exact (Finset.mul_sum (Finset.univ : Finset ι') (fun b => f a b) c).symm
      _ = c * ∑ a, ∑ b, f a b :=
        (Finset.mul_sum (Finset.univ : Finset ι) (fun a => ∑ b, f a b) c).symm
  simp_rw [mul_assoc]
  rw [factor, factor]


/-- **Sharp covariance ceiling.** Given a plan attaining the certified finite
    minimum transport cost, the value `κ̄ = (v_P + v_Q − optimalCost)/2` is the
    greatest admissible systematic covariance — and it is *attained*, not merely
    an upper bound.

    Existence of the plan is a hypothesis, not an assertion: in the finite
    setting it is discharged by any exact solver (network simplex / Hungarian),
    so the caller supplies the certificate the pipeline already computes. -/
theorem isGreatest_frechetClass
    (Z : ι → H) (Y : ι' → H) (optimalCost : ℝ) (π₀ : Coupling p q)
    (hopt : transportCost π₀ Z Y = optimalCost)
    (hmin : ∀ π : Coupling p q, optimalCost ≤ transportCost π Z Y) :
    IsGreatest (frechetClass p q Z Y)
      ((secondMoment p Z + secondMoment q Y - optimalCost) / 2) := by
  constructor
  · exact ⟨π₀, by rw [systCov_eq_of_transportCost, hopt]⟩
  · rintro c ⟨π, rfl⟩
    rw [systCov_eq_of_transportCost]
    have := hmin π
    linarith

/-! ### Reflection and the floor -/

/-- Reflecting the second cloud negates every admissible covariance. -/
theorem systCov_neg_right (π : Coupling p q) (Z : ι → H) (Y : ι' → H) :
    systCov π Z (fun b => -Y b) = -systCov π Z Y := by
  unfold systCov
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun b _ => by rw [inner_neg_right]; ring

omit [InnerProductSpace ℝ H] in
/-- Reflection preserves second moments. -/
theorem secondMoment_neg (q : ι' → ℝ) (Y : ι' → H) :
    secondMoment q (fun b => -Y b) = secondMoment q Y := by
  unfold secondMoment
  exact Finset.sum_congr rfl fun b _ => by rw [norm_neg]

/-- **Sharp covariance floor.** Applying the ceiling result to the reflected
    cloud `−Y` gives the least admissible systematic covariance,
    `κ̲ = (optimalReflectedCost − v_P − v_Q)/2`, where
    `optimalReflectedCost` is a certified finite minimum transport cost. -/
theorem isLeast_frechetClass
    (Z : ι → H) (Y : ι' → H) (optimalReflectedCost : ℝ) (π₀ : Coupling p q)
    (hopt : transportCost π₀ Z (fun b => -Y b) = optimalReflectedCost)
    (hmin : ∀ π : Coupling p q,
      optimalReflectedCost ≤ transportCost π Z (fun b => -Y b)) :
    IsLeast (frechetClass p q Z Y)
      ((optimalReflectedCost - secondMoment p Z - secondMoment q Y) / 2) := by
  constructor
  · refine ⟨π₀, ?_⟩
    have h := systCov_eq_of_transportCost π₀ Z (fun b => -Y b)
    rw [systCov_neg_right, secondMoment_neg, hopt] at h
    linarith
  · rintro c ⟨π, rfl⟩
    have h := systCov_eq_of_transportCost π Z (fun b => -Y b)
    rw [systCov_neg_right, secondMoment_neg] at h
    have := hmin π
    linarith

/-! ### The gap identity -/

/-- Excess transport cost of a coupling relative to a certified finite minimum
    transport cost, `Δ_π = 𝔼_π ‖Z − Y‖² − optimalCost`. -/
noncomputable def excess
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) (optimalCost : ℝ) : ℝ :=
  transportCost π Z Y - optimalCost

omit [InnerProductSpace ℝ H] in
/-- Excess is nonnegative whenever `optimalCost` lower-bounds the cost. -/
theorem excess_nonneg (π : Coupling p q) (Z : ι → H) (Y : ι' → H)
    {optimalCost : ℝ}
    (hmin : optimalCost ≤ transportCost π Z Y) :
    0 ≤ excess π Z Y optimalCost := by
  unfold excess; linarith

/-- **Gap identity.** `κ_π = κ̄ − Δ_π/2`.

    The realized covariance under any coupling equals the marginal-geometry
    ceiling minus half the coupling's excess transport cost. This is the model's
    testable content: `κ̄` is identified by the two marginals alone, while `Δ_π`
    requires paired draws — so the two are separately estimable and the residual
    `Δ̂` is a falsifiable diagnostic, not a free parameter. -/
theorem systCov_eq_ceiling_sub_excess
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) (optimalCost : ℝ) :
    systCov π Z Y
      = (secondMoment p Z + secondMoment q Y - optimalCost) / 2
        - excess π Z Y optimalCost / 2 := by
  rw [systCov_eq_of_transportCost]; unfold excess; ring

/-! ### Nesting: deterministic loadings -/

/-- **The vector/deterministic model is nested exactly.** When each cloud is a
    single point — `Cᵢ = δ_ω` and zero slack — the coupling is forced, every
    admissible covariance coincides, and the envelope has zero width.

    This is the `τ = 0`, `C = δ_ω` rung of the ladder: the random functional
    model degenerates to the classical fixed-loading factor model rather than
    merely approximating it. -/
theorem systCov_subsingleton_eq [Subsingleton ι] [Subsingleton ι'] [Nonempty ι] [Nonempty ι']
    (π π' : Coupling p q) (Z : ι → H) (Y : ι' → H) :
    systCov π Z Y = systCov π' Z Y := by
  obtain ⟨a₀⟩ := ‹Nonempty ι›
  obtain ⟨b₀⟩ := ‹Nonempty ι'›
  have hsum : ∀ (f : ι → ℝ), ∑ a, f a = f a₀ := fun f => Fintype.sum_subsingleton f a₀
  have hsum' : ∀ (f : ι' → ℝ), ∑ b, f b = f b₀ := fun f => Fintype.sum_subsingleton f b₀
  have hw : ∀ (σ : Coupling p q), σ.w a₀ b₀ = p a₀ := by
    intro σ
    have := σ.marginal_fst a₀
    rwa [hsum' fun b => σ.w a₀ b] at this
  unfold systCov
  rw [hsum, hsum', hsum, hsum', hw π, hw π']
/-- The covariance ceiling API with a finite optimal-cost certificate. -/
theorem isGreatest_of_transportCertificate
    (Z : ι → H) (Y : ι' → H)
    (cert : FiniteOptimalCostCertificate
      (fun π : Coupling p q => transportCost π Z Y)) :
    IsGreatest (frechetClass p q Z Y)
      ((secondMoment p Z + secondMoment q Y - cert.value) / 2) :=
  isGreatest_frechetClass Z Y cert.value cert.plan cert.plan_cost cert.minimal

omit [InnerProductSpace ℝ H] in
/-- Excess is nonnegative relative to a certified finite minimum transport cost. -/
theorem excess_nonneg_of_transportCertificate
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H)
    (cert : FiniteOptimalCostCertificate
      (fun π : Coupling p q => transportCost π Z Y)) :
    0 ≤ excess π Z Y cert.value := by
  exact excess_nonneg π Z Y (cert.minimal π)

/-- Every covariance between two attained endpoint values is attained. -/
theorem frechetClass_contains_Icc_of_lt
    (Z : ι → H) (Y : ι' → H) (πlo πhi : Coupling p q) (lo hi : ℝ)
    (hlo : systCov πlo Z Y = lo) (hhi : systCov πhi Z Y = hi)
    (hwidth : lo < hi) {c : ℝ} (hlo_c : lo ≤ c) (hc_hi : c ≤ hi) :
    c ∈ Set.Icc lo hi → c ∈ frechetClass p q Z Y := by
  intro _
  let t : ℝ := (hi - c) / (hi - lo)
  have hden : 0 < hi - lo := sub_pos.mpr hwidth
  have ht0 : 0 ≤ t := by
    dsimp [t]
    exact div_nonneg (sub_nonneg.mpr hc_hi) (le_of_lt hden)
  have ht1 : t ≤ 1 := by
    dsimp [t]
    apply (div_le_iff₀ hden).mpr
    linarith
  refine ⟨πlo.mix πhi t ht0 ht1, ?_⟩
  rw [systCov_mix, hlo, hhi]
  dsimp [t]
  field_simp [ne_of_gt hden]
  ring

/-- Deterministic singleton clouds have a singleton covariance class. -/
theorem frechetClass_eq_singleton_of_subsingleton
    [Subsingleton ι] [Subsingleton ι'] [Nonempty ι] [Nonempty ι']
    (π₀ : Coupling p q) (Z : ι → H) (Y : ι' → H) :
    frechetClass p q Z Y = {systCov π₀ Z Y} := by
  ext c
  constructor
  · rintro ⟨π, hπ⟩
    rw [Set.mem_singleton_iff]
    exact hπ.symm.trans (systCov_subsingleton_eq π π₀ Z Y)
  · intro hc
    rw [Set.mem_singleton_iff] at hc
    exact ⟨π₀, hc.symm⟩




/-- A covariance envelope with equal least and greatest endpoints is a singleton. -/
theorem frechetClass_eq_singleton_of_zero_width
    (S : Set ℝ) (lo hi : ℝ)
    (hlo : IsLeast S lo) (hhi : IsGreatest S hi) (hwidth : hi = lo) :
    S = {lo} := by
  ext c
  constructor
  · intro hc
    rw [Set.mem_singleton_iff]
    linarith [hlo.2 hc, hhi.2 hc, hwidth]
  · intro hc
    rw [Set.mem_singleton_iff] at hc
    simpa [hc] using hlo.1

end PricingPerspective.RandomExposure
