import WassersteinGeometry.Hilbert.Polarization

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry WassersteinGeometry.Hilbert

/-!
# The systematic-covariance envelope on `P₂(H)`

The measure-theoretic counterpart of `Transmission.Envelope`, which develops the same
envelope for finite clouds. Both rest on one identity — polarization, proved once in
`WassersteinGeometry.Hilbert.Polarization` — and both say that the systematic covariance
of two assets is pinned only up to the *coupling* of their loading laws, with quadratic
transport measuring exactly the residual freedom.

## The Fréchet class

`frechetClass P Q` is the set of systematic covariances attainable across couplings of `P`
and `Q`. Its endpoints are the economic content:

* the **ceiling** `κ̄ = (v_P + v_Q − W₂²) / 2`, attained by an optimal plan
  (`isGreatest_frechetClass`);
* the **floor** `κ̲ = (W₂²(P, Q̌) − v_P − v_Q) / 2` for the reflected law `Q̌ = (−I)_#Q`
  (`isLeast_frechetClass_of_reflection`);
* the **excess** `Δ_π = cost_π − W₂²`, with `κ_π = κ̄ − Δ_π/2`
  (`systCov_eq_ceiling_sub_excess`), so a coupling's covariance shortfall from the ceiling
  is exactly half its excess transport cost — an object estimable from paired draws.

## Cached integrability and certified optimality

**Integrability travels with the coupling.** `IntegrableCoupling` caches the three facts
polarization needs. `IntegrableCoupling.ofMeasure` proves that every raw coupling of two
`P₂` laws supplies those facts automatically, so this bundle is an ergonomic API rather
than a restriction of the Fréchet class.

**Optimality is certified, not proved.** `OptimalTransportCost` supplies a minimizing plan
as data. Existence of optimal couplings is a genuine theorem of optimal transport that this
workspace does not prove — `WassersteinGeometry.OptimalCoupling` makes the same choice. So
`isGreatest_frechetClass` reads: *given* an optimal plan, the ceiling is attained. It is not
a claim that one exists.

## What this does not say

The ceiling is an upper bound over couplings, not a prediction. Which coupling nature
selects is an economic question about the joint law of the loading processes, and marginals
cannot answer it. Small `W₂` raises the ceiling and so can exclude low covariance; large
`W₂` lowers it and excludes nothing.

## References

* Gelbrich (1990); Puccetti & Wang (2015); Panaretos & Zemel (2020), Ch. 2.
-/

namespace PricingPerspective.Transmission

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-! ### Couplings that carry their own integrability -/

/-- A coupling of two loading laws together with the integrability polarization needs.

    The three fields are exactly the hypotheses of
    `WassersteinGeometry.Hilbert.transportCost_eq_polarization`. Bundling them makes
    `systCov` well defined on the whole index set of the Fréchet class below. -/
structure IntegrableCoupling (P Q : WassersteinMeasure H) where
  /-- The underlying joint law on `H × H`. -/
  toMeasure : Measure (H × H)
  /-- Its marginals are `P` and `Q`. -/
  mem : toMeasure ∈ couplingSet P.measure Q.measure
  /-- The first marginal's second moment is integrable on the coupling. -/
  int_fst : Integrable (fun p : H × H => ‖p.1‖ ^ 2) toMeasure
  /-- The second marginal's second moment is integrable on the coupling. -/
  int_snd : Integrable (fun p : H × H => ‖p.2‖ ^ 2) toMeasure
  /-- The cross term is integrable, so `systCov` is defined. -/
  int_inner : Integrable (fun p : H × H => ⟪p.1, p.2⟫_ℝ) toMeasure

namespace IntegrableCoupling

variable {P Q : WassersteinMeasure H}

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
private lemma integrable_normSq (P : WassersteinMeasure H) :
    Integrable (fun x : H => ‖x‖ ^ 2) P.measure := by
  letI : IsProbabilityMeasure P.measure := ⟨P.is_probability⟩
  have hshift : Integrable (fun x : H => ‖x - default‖ ^ 2) P.measure := by
    apply (lintegral_ofReal_ne_top_iff_integrable
      (((continuous_norm.pow 2).comp (continuous_id.sub continuous_const)).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun x => sq_nonneg ‖x - default‖)).mp
    simpa only [Function.comp_apply, id_eq, edist_dist, dist_eq_norm,
      ← ENNReal.ofReal_pow (norm_nonneg _)] using P.second_moment_finite.ne
  have hmajorant :
      Integrable (fun x : H => 2 * ‖x - default‖ ^ 2 + 2 * ‖(default : H)‖ ^ 2)
        P.measure :=
    (hshift.const_mul 2).add (integrable_const _)
  refine Integrable.mono' hmajorant (aestronglyMeasurable_normSq _) ?_
  filter_upwards with x
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg ‖x‖)]
  have htri : ‖x‖ ≤ ‖x - default‖ + ‖(default : H)‖ := by
    calc
      ‖x‖ = ‖(x - default) + default‖ := by rw [sub_add_cancel]
      _ ≤ ‖x - default‖ + ‖(default : H)‖ := norm_add_le _ _
  nlinarith [sq_nonneg (‖x - default‖ - ‖(default : H)‖), norm_nonneg x]

/-- Package any coupling of two `P₂` laws as an integrable coupling.

    The marginal second-moment assumptions imply integrability of both coordinate squares;
    the real Cauchy–Schwarz inequality then supplies integrability of the cross term. Thus
    `IntegrableCoupling` carries convenient cached facts rather than an extra restriction on
    the Fréchet class. -/
noncomputable def ofMeasure (P Q : WassersteinMeasure H) (toMeasure : Measure (H × H))
    (mem : toMeasure ∈ couplingSet P.measure Q.measure) : IntegrableCoupling P Q where
  toMeasure := toMeasure
  mem := mem
  int_fst := by
    have hP := integrable_normSq P
    rw [← mem.1] at hP
    exact (integrable_map_measure (aestronglyMeasurable_normSq _)
      measurable_fst.aemeasurable).mp hP
  int_snd := by
    have hQ := integrable_normSq Q
    rw [← mem.2] at hQ
    exact (integrable_map_measure (aestronglyMeasurable_normSq _)
      measurable_snd.aemeasurable).mp hQ
  int_inner := by
    have hfst : Integrable (fun p : H × H => ‖p.1‖ ^ 2) toMeasure := by
      have hP := integrable_normSq P
      rw [← mem.1] at hP
      exact (integrable_map_measure (aestronglyMeasurable_normSq _)
        measurable_fst.aemeasurable).mp hP
    have hsnd : Integrable (fun p : H × H => ‖p.2‖ ^ 2) toMeasure := by
      have hQ := integrable_normSq Q
      rw [← mem.2] at hQ
      exact (integrable_map_measure (aestronglyMeasurable_normSq _)
        measurable_snd.aemeasurable).mp hQ
    refine Integrable.mono' (hfst.add hsnd)
      (by exact (continuous_inner (𝕜 := ℝ)).aestronglyMeasurable) ?_
    filter_upwards with p
    rw [Real.norm_eq_abs]
    calc
      |⟪p.1, p.2⟫_ℝ| ≤ ‖p.1‖ * ‖p.2‖ := abs_real_inner_le_norm _ _
      _ ≤ ‖p.1‖ ^ 2 + ‖p.2‖ ^ 2 := by
        nlinarith [sq_nonneg (‖p.1‖ - ‖p.2‖)]

/-- Convex mixture of two integrable couplings with the same marginals. -/
noncomputable def mix (π ρ : IntegrableCoupling P Q) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    IntegrableCoupling P Q where
  toMeasure := t.toNNReal • π.toMeasure + (1 - t).toNNReal • ρ.toMeasure
  mem := by
    constructor
    · rw [Measure.map_add _ _ measurable_fst, Measure.map_smul, Measure.map_smul,
        π.mem.1, ρ.mem.1, ← add_smul]
      have : t.toNNReal + (1 - t).toNNReal = 1 := by
        ext
        simp [Real.toNNReal_of_nonneg ht0, Real.toNNReal_of_nonneg (sub_nonneg.mpr ht1)]
      rw [this, one_smul]
    · rw [Measure.map_add _ _ measurable_snd, Measure.map_smul, Measure.map_smul,
        π.mem.2, ρ.mem.2, ← add_smul]
      have : t.toNNReal + (1 - t).toNNReal = 1 := by
        ext
        simp [Real.toNNReal_of_nonneg ht0, Real.toNNReal_of_nonneg (sub_nonneg.mpr ht1)]
      rw [this, one_smul]
  int_fst := integrable_add_measure.mpr
    ⟨π.int_fst.smul_measure_nnreal, ρ.int_fst.smul_measure_nnreal⟩
  int_snd := integrable_add_measure.mpr
    ⟨π.int_snd.smul_measure_nnreal, ρ.int_snd.smul_measure_nnreal⟩
  int_inner := integrable_add_measure.mpr
    ⟨π.int_inner.smul_measure_nnreal, ρ.int_inner.smul_measure_nnreal⟩

omit [BorelSpace H] [SecondCountableTopology H] in
/-- Systematic covariance is affine under convex mixtures of couplings. -/
theorem systCov_mix (π ρ : IntegrableCoupling P Q) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    systCov (π.mix ρ t ht0 ht1).toMeasure =
      t * systCov π.toMeasure + (1 - t) * systCov ρ.toMeasure := by
  unfold systCov mix
  rw [integral_add_measure π.int_inner.smul_measure_nnreal
      ρ.int_inner.smul_measure_nnreal,
    integral_smul_nnreal_measure, integral_smul_nnreal_measure]
  simp [Real.toNNReal_of_nonneg ht0, Real.toNNReal_of_nonneg (sub_nonneg.mpr ht1),
    NNReal.smul_def]

/-- Polarization, specialized to a coupling that carries its integrability. -/
theorem transportCost_eq (π : IntegrableCoupling P Q) :
    transportCost π.toMeasure = secondMoment P + secondMoment Q - 2 * systCov π.toMeasure :=
  transportCost_eq_polarization π.mem π.int_fst π.int_snd π.int_inner

/-- Systematic covariance solved out of polarization. -/
theorem systCov_eq (π : IntegrableCoupling P Q) :
    systCov π.toMeasure =
      (secondMoment P + secondMoment Q - transportCost π.toMeasure) / 2 :=
  systCov_eq_of_polarization π.mem π.int_fst π.int_snd π.int_inner

omit [BorelSpace H] [SecondCountableTopology H] in
/-- The transport cost of an integrable coupling is integrable. -/
theorem int_cost (π : IntegrableCoupling P Q) :
    Integrable (fun p : H × H => ‖p.1 - p.2‖ ^ 2) π.toMeasure :=
  transportCost_integrable_of π.int_fst π.int_snd π.int_inner

end IntegrableCoupling

/-! ### The Fréchet class of attainable covariances -/

/-- The set of systematic covariances attainable across couplings of `P` and `Q`.

    Two marginal loading laws do not determine systematic covariance. This is the exact set
    of values they leave open — the identification problem, written down. -/
def frechetClass (P Q : WassersteinMeasure H) : Set ℝ :=
  {κ | ∃ π : IntegrableCoupling P Q, systCov π.toMeasure = κ}

omit [BorelSpace H] [SecondCountableTopology H] in
lemma mem_frechetClass {P Q : WassersteinMeasure H} (π : IntegrableCoupling P Q) :
    systCov π.toMeasure ∈ frechetClass P Q := ⟨π, rfl⟩

/-! ### Certified optimal transport -/

/-- A certificate that `value` is the least transport cost over integrable couplings, with
    a plan attaining it.

    Existence is *not* proved here; the caller supplies the minimizer. That mirrors
    `WassersteinGeometry.OptimalCoupling` and the finite `FiniteOptimalCostCertificate`. -/
structure OptimalTransportCost (P Q : WassersteinMeasure H) where
  /-- A coupling attaining the certified minimum. -/
  plan : IntegrableCoupling P Q
  /-- The certified minimal transport cost. -/
  value : ℝ
  /-- The supplied plan attains it. -/
  plan_cost : transportCost plan.toMeasure = value
  /-- No integrable coupling does better. -/
  minimal : ∀ π : IntegrableCoupling P Q, value ≤ transportCost π.toMeasure

/-- The covariance ceiling determined by a certified optimal transport cost. -/
noncomputable def ceiling {P Q : WassersteinMeasure H} (cert : OptimalTransportCost P Q) : ℝ :=
  (secondMoment P + secondMoment Q - cert.value) / 2

/-- **The covariance ceiling is attained.**

    `κ̄ = (v_P + v_Q − W₂²)/2` is the greatest element of the Fréchet class: the optimal
    transport plan realizes it, and no coupling exceeds it.

    This is what quadratic transport buys over an unoptimized distance. Energy distance can
    bound covariance but cannot exhibit a coupling attaining the bound, because it optimizes
    over nothing. -/
theorem isGreatest_frechetClass {P Q : WassersteinMeasure H}
    (cert : OptimalTransportCost P Q) :
    IsGreatest (frechetClass P Q) (ceiling cert) := by
  constructor
  · -- The certified plan attains the ceiling.
    refine ⟨cert.plan, ?_⟩
    rw [cert.plan.systCov_eq, cert.plan_cost]
    rfl
  · -- No coupling exceeds it, because none undercuts the minimal cost.
    rintro κ ⟨π, rfl⟩
    rw [π.systCov_eq]
    have := cert.minimal π
    unfold ceiling
    linarith

/-! ### Excess transport cost -/

/-- The excess transport cost of a coupling over the certified optimum. -/
noncomputable def excess {P Q : WassersteinMeasure H}
    (cert : OptimalTransportCost P Q) (π : IntegrableCoupling P Q) : ℝ :=
  transportCost π.toMeasure - cert.value

omit [BorelSpace H] [SecondCountableTopology H] in
/-- Excess is nonnegative, by minimality of the certificate. -/
theorem excess_nonneg {P Q : WassersteinMeasure H}
    (cert : OptimalTransportCost P Q) (π : IntegrableCoupling P Q) :
    0 ≤ excess cert π :=
  sub_nonneg.mpr (cert.minimal π)

/-- **The gap identity.** `κ_π = κ̄ − Δ_π / 2`.

    A coupling's shortfall from the covariance ceiling is exactly half its excess transport
    cost. Unlike the ceiling itself, the excess is estimable from paired draws, which is
    what makes the envelope testable rather than merely true. -/
theorem systCov_eq_ceiling_sub_excess {P Q : WassersteinMeasure H}
    (cert : OptimalTransportCost P Q) (π : IntegrableCoupling P Q) :
    systCov π.toMeasure = ceiling cert - excess cert π / 2 := by
  rw [π.systCov_eq]
  unfold ceiling excess
  ring

/-! ### The floor, via the reflected law

The lower endpoint is the upper endpoint of the *reflected* problem. Replacing `Q` by
`Q̌ = (−I)_#Q` negates every attainable covariance, so the least attainable covariance
against `Q` is minus the greatest against `Q̌`. Economically: the most negatively two assets
can co-move is governed by how close one loading law is to the *reflection* of the other. -/

section Reflection

variable {P Q Q' : WassersteinMeasure H}

/-- Reflection of the second coordinate of a coupling. -/
def reflSnd : H × H → H × H := fun p => (p.1, -p.2)

omit [InnerProductSpace ℝ H] [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
@[simp] lemma reflSnd_apply (p : H × H) : reflSnd p = (p.1, -p.2) := rfl

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] [Inhabited H] in
lemma measurable_reflSnd : Measurable (reflSnd : H × H → H × H) :=
  measurable_fst.prodMk measurable_snd.neg

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- Reflecting the second marginal maps `Π(P, Q)` into `Π(P, Q̌)`. -/
lemma map_reflSnd_mem_couplingSet
    (hQ' : Q'.measure = Q.measure.map (fun y : H => -y))
    {π : Measure (H × H)} (hπ : π ∈ couplingSet P.measure Q.measure) :
    π.map reflSnd ∈ couplingSet P.measure Q'.measure := by
  constructor
  · rw [Measure.map_map measurable_fst measurable_reflSnd]
    have : (Prod.fst ∘ (reflSnd : H × H → H × H)) = Prod.fst := rfl
    rw [this]; exact hπ.1
  · rw [Measure.map_map measurable_snd measurable_reflSnd, hQ']
    have : (Prod.snd ∘ (reflSnd : H × H → H × H)) = (fun y : H => -y) ∘ Prod.snd := rfl
    rw [this, ← Measure.map_map measurable_neg measurable_snd, hπ.2]

omit [Inhabited H] in
/-- Reflection negates systematic covariance. -/
lemma systCov_map_reflSnd (π : Measure (H × H)) :
    systCov (π.map reflSnd) = - systCov π := by
  unfold systCov
  rw [integral_map measurable_reflSnd.aemeasurable
    (by exact (continuous_inner (𝕜 := ℝ)).aestronglyMeasurable)]
  simp [reflSnd, inner_neg_right, integral_neg]

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- Reflection preserves the second moment. -/
lemma secondMoment_reflect (hQ' : Q'.measure = Q.measure.map (fun y : H => -y)) :
    secondMoment Q' = secondMoment Q := by
  unfold secondMoment
  rw [hQ', integral_map measurable_neg.aemeasurable (aestronglyMeasurable_normSq _)]
  simp

/-- Reflection of an integrable coupling is an integrable coupling of the reflected law. -/
noncomputable def IntegrableCoupling.reflect
    (hQ' : Q'.measure = Q.measure.map (fun y : H => -y))
    (π : IntegrableCoupling P Q) : IntegrableCoupling P Q' where
  toMeasure := π.toMeasure.map reflSnd
  mem := map_reflSnd_mem_couplingSet hQ' π.mem
  int_fst := by
    rw [integrable_map_measure (aestronglyMeasurable_fst_normSq _)
      measurable_reflSnd.aemeasurable]
    exact π.int_fst
  int_snd := by
    rw [integrable_map_measure (aestronglyMeasurable_snd_normSq _)
      measurable_reflSnd.aemeasurable]
    simpa [Function.comp_def, reflSnd] using π.int_snd
  int_inner := by
    rw [integrable_map_measure
      (by exact (continuous_inner (𝕜 := ℝ)).aestronglyMeasurable)
      measurable_reflSnd.aemeasurable]
    simpa [Function.comp_def, reflSnd, inner_neg_right] using π.int_inner.neg

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- Reflecting twice is the identity on laws. -/
lemma measure_eq_map_neg_of_reflect
    (hQ' : Q'.measure = Q.measure.map (fun y : H => -y)) :
    Q.measure = Q'.measure.map (fun y : H => -y) := by
  rw [hQ', Measure.map_map measurable_neg measurable_neg]
  have : ((fun y : H => -y) ∘ fun y : H => -y) = id := by funext y; simp
  rw [this, Measure.map_id]

/-- **The covariance floor is attained.**

    `κ̲ = −(v_P + v_{Q̌} − W₂²(P, Q̌))/2` is the least element of the Fréchet class, given an
    optimal plan for the reflected problem.

    Together with `isGreatest_frechetClass` this brackets systematic covariance from both
    sides using only marginal loading laws: quadratic transport against `Q` caps co-movement
    from above, and quadratic transport against `Q̌` caps it from below. -/
theorem isLeast_frechetClass_of_reflection
    (hQ' : Q'.measure = Q.measure.map (fun y : H => -y))
    (cert : OptimalTransportCost P Q') :
    IsLeast (frechetClass P Q) (-(ceiling cert)) := by
  have hback : Q.measure = Q'.measure.map (fun y : H => -y) :=
    measure_eq_map_neg_of_reflect hQ'
  constructor
  · -- Reflect the certified optimal plan back to a coupling of `(P, Q)`.
    refine ⟨cert.plan.reflect hback, ?_⟩
    have hcov : systCov (cert.plan.reflect hback).toMeasure = - systCov cert.plan.toMeasure :=
      systCov_map_reflSnd _
    rw [hcov, cert.plan.systCov_eq, cert.plan_cost]
    rfl
  · -- Every attainable covariance is minus one attainable against the reflected law.
    rintro κ ⟨π, rfl⟩
    have hmem : systCov (π.reflect hQ').toMeasure ∈ frechetClass P Q' :=
      mem_frechetClass _
    have hle : systCov (π.reflect hQ').toMeasure ≤ ceiling cert :=
      (isGreatest_frechetClass cert).2 hmem
    have hcov : systCov (π.reflect hQ').toMeasure = - systCov π.toMeasure :=
      systCov_map_reflSnd _
    rw [hcov] at hle
    linarith

/-- **The complete Fréchet identification interval.**

    Given certified optimal plans for the direct and reflected quadratic-transport
    problems, the attainable systematic covariances are exactly
    `[-ceiling(reflected), ceiling(direct)]`. Every interior value is realized by a convex
    mixture of the two endpoint plans. The proof also covers a zero-width interval, where
    the two attained endpoints coincide and the Fréchet class is a singleton. -/
theorem frechetClass_eq_Icc_of_reflection
    (hQ' : Q'.measure = Q.measure.map (fun y : H => -y))
    (upperCert : OptimalTransportCost P Q)
    (lowerCert : OptimalTransportCost P Q') :
    frechetClass P Q = Set.Icc (-(ceiling lowerCert)) (ceiling upperCert) := by
  have hlo := isLeast_frechetClass_of_reflection hQ' lowerCert
  have hhi := isGreatest_frechetClass upperCert
  apply Set.Subset.antisymm
  · intro κ hκ
    exact ⟨hlo.2 hκ, hhi.2 hκ⟩
  · rintro κ ⟨hloκ, hκhi⟩
    by_cases hzero : -(ceiling lowerCert) = ceiling upperCert
    · have hκle : κ ≤ -(ceiling lowerCert) := by
        rw [hzero]
        exact hκhi
      have hκeq : κ = -(ceiling lowerCert) := le_antisymm hκle hloκ
      rw [hκeq]
      exact hlo.1
    · have hle : -(ceiling lowerCert) ≤ ceiling upperCert := hlo.2 hhi.1
      have hwidth : -(ceiling lowerCert) < ceiling upperCert := lt_of_le_of_ne hle hzero
      rcases hlo.1 with ⟨πlo, hπlo⟩
      rcases hhi.1 with ⟨πhi, hπhi⟩
      let t : ℝ := (ceiling upperCert - κ) /
        (ceiling upperCert - -(ceiling lowerCert))
      have hden : 0 < ceiling upperCert - -(ceiling lowerCert) := sub_pos.mpr hwidth
      have ht0 : 0 ≤ t := by
        dsimp [t]
        exact div_nonneg (sub_nonneg.mpr hκhi) (le_of_lt hden)
      have ht1 : t ≤ 1 := by
        dsimp [t]
        apply (div_le_iff₀ hden).mpr
        linarith
      refine ⟨πlo.mix πhi t ht0 ht1, ?_⟩
      rw [IntegrableCoupling.systCov_mix, hπlo, hπhi]
      dsimp [t]
      field_simp [ne_of_gt hden]
      ring

end Reflection

end PricingPerspective.Transmission
