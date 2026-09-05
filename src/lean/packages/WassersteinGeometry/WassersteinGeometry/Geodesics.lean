import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Probability.Kernel.Disintegration.StandardBorel
import Mathlib.Probability.Kernel.Composition.MeasureCompProd
import Mathlib.Probability.Kernel.MeasurableLIntegral
import Mathlib.MeasureTheory.Integral.MeanInequalities
import Mathlib.MeasureTheory.Constructions.BorelSpace.Metric
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal

set_option linter.style.longLine false
set_option linter.unusedVariables false
set_option maxHeartbeats 1000000

open scoped ENNReal MeasureTheory
open MeasureTheory ProbabilityTheory

/-!
# Wasserstein Geodesics

This module formalizes geodesics in Wasserstein space,
following Panaretos & Zemel (2020), "An Invitation to Statistics in Wasserstein Space".

## Main definitions

* `WassersteinMeasure`: The space P₂(Ω) of probability measures with finite second moment
* `couplingSet`: The set of couplings (transference plans) between two measures
* `WassersteinDistanceSq`: The squared 2-Wasserstein distance W₂²(μ, ν)
* `WassersteinDistance`: The 2-Wasserstein distance W₂(μ, ν)
* `Geodesic`: A constant-speed geodesic in Wasserstein space
* `OptimalCoupling`: A coupling achieving the Wasserstein distance

## Main theorems

* `wasserstein_nonneg`: W₂(μ, ν) ≥ 0
* `wasserstein_symm`: W₂(μ, ν) = W₂(ν, μ)
* `wasserstein_triangle`: W₂(μ, ν) ≤ W₂(μ, λ) + W₂(λ, ν)
* `KantorovichDualCertificate`: corrected measurable finite-cost dual contract

## References

* Panaretos, V. M., & Zemel, Y. (2020). An Invitation to Statistics in Wasserstein Space.
  Springer. Chapters 1, 2.
-/

namespace WassersteinGeometry

/-- A probability measure with finite second moment.

    The second moment condition ensures that the 2-Wasserstein distance is finite.
    The basepoint `default` from `[Inhabited Ω]` is used as the reference point;
    in a PseudoMetricSpace, finiteness is independent of the choice of basepoint
    (by the triangle inequality and the fact that μ is a probability measure).

    Reference: Panaretos & Zemel (2020), Chapter 2 §2.1. -/
structure WassersteinMeasure (Ω : Type*) [MeasurableSpace Ω] [PseudoMetricSpace Ω]
    [Inhabited Ω] where
  measure : MeasureTheory.Measure Ω
  is_probability : measure .univ = 1
  second_moment_finite : ∫⁻ x, (edist x default)^2 ∂measure < ⊤

/-- The set of couplings (transference plans) between μ and ν.

    A coupling π has μ as its first marginal and ν as its second marginal.
    This is the set Π(μ, ν) in the optimal transport literature.

    Reference: Panaretos & Zemel (2020), Chapter 1 §1.2. -/
def couplingSet {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : MeasureTheory.Measure Ω) : Set (MeasureTheory.Measure (Ω × Ω)) :=
  {π | π.map Prod.fst = μ ∧ π.map Prod.snd = ν}

/-- The squared 2-Wasserstein distance W₂²(μ, ν) as an extended nonnegative real.

    W₂²(μ, ν) = inf_{π ∈ Π(μ,ν)} ∫ ∥x - y∥² dπ(x, y)

    where Π(μ, ν) is the set of couplings with marginals μ and ν.

    Reference: Panaretos & Zemel (2020), Chapter 2 §2.1. -/
noncomputable def WassersteinDistanceSq
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    (μ ν : WassersteinMeasure Ω) : ℝ≥0∞ :=
  sInf {c : ℝ≥0∞ | ∃ π ∈ couplingSet μ.measure ν.measure,
    c = ∫⁻ p, (edist p.1 p.2)^2 ∂π}

/-- The 2-Wasserstein distance W₂(μ, ν).

    W₂(μ, ν) = (inf_{π ∈ Π(μ,ν)} ∫ ∥x - y∥² dπ(x, y))^{1/2}

    Reference: Panaretos & Zemel (2020), Chapter 2 §2.1. -/
noncomputable def WassersteinDistance
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    (μ ν : WassersteinMeasure Ω) : ℝ :=
  ENNReal.toReal (WassersteinDistanceSq μ ν) ^ (1/2 : ℝ)

/-- Wasserstein distance is nonnegative.

    W₂(μ, ν) ≥ 0

    Proof: W₂² is an infimum of nonnegative integrals, so ENNReal.toReal is ≥ 0,
    and raising a nonnegative number to the power 1/2 is nonnegative.

    Reference: Panaretos & Zemel (2020), Proposition 2.2. -/
theorem wasserstein_nonneg
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    [CompleteSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinMeasure Ω) :
    0 ≤ WassersteinDistance μ ν :=
  Real.rpow_nonneg ENNReal.toReal_nonneg _

/-- Wasserstein distance is symmetric.

    W₂(μ, ν) = W₂(ν, μ)

    Proof: The set of couplings Π(μ, ν) is symmetric: if π ∈ Π(μ, ν),
    then the pushforward of π under (x, y) ↦ (y, x) is in Π(ν, μ),
    and the cost function ∥x - y∥² is symmetric.

    Reference: Panaretos & Zemel (2020), Proposition 2.2. -/
theorem wasserstein_symm
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    [CompleteSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinMeasure Ω) :
    WassersteinDistance μ ν = WassersteinDistance ν μ := by
  -- The swap map is used as a `MeasurableEquiv` so that `lintegral_map_equiv` applies
  -- without any measurability hypothesis on the cost integrand: the ambient
  -- `MeasurableSpace Ω` is not assumed Borel, so `measurable_edist` /
  -- `continuous_edist.measurable` are NOT available here.
  have key : ∀ a b : WassersteinMeasure Ω,
      {c : ℝ≥0∞ | ∃ π ∈ couplingSet a.measure b.measure,
          c = ∫⁻ p, (edist p.1 p.2)^2 ∂π} ⊆
        {c : ℝ≥0∞ | ∃ π ∈ couplingSet b.measure a.measure,
          c = ∫⁻ p, (edist p.1 p.2)^2 ∂π} := by
    rintro a b c ⟨π, ⟨hfst, hsnd⟩, rfl⟩
    refine ⟨π.map (MeasurableEquiv.prodComm (α := Ω) (β := Ω)), ⟨?_, ?_⟩, ?_⟩
    · rw [MeasureTheory.Measure.map_map measurable_fst
        (MeasurableEquiv.prodComm (α := Ω) (β := Ω)).measurable]
      exact hsnd
    · rw [MeasureTheory.Measure.map_map measurable_snd
        (MeasurableEquiv.prodComm (α := Ω) (β := Ω)).measurable]
      exact hfst
    · rw [MeasureTheory.lintegral_map_equiv]
      exact MeasureTheory.lintegral_congr fun p => by
        show (edist p.1 p.2) ^ 2 = (edist p.2 p.1) ^ 2
        rw [edist_comm]
  have h_sq_symm : WassersteinDistanceSq μ ν = WassersteinDistanceSq ν μ := by
    unfold WassersteinDistanceSq
    exact le_antisymm (sInf_le_sInf (key ν μ)) (sInf_le_sInf (key μ ν))
  unfold WassersteinDistance
  rw [h_sq_symm]

/-- Every coupling of probability marginals is a probability measure.

    Proof: `π.map Prod.fst = μ`, so `π(Ω × Ω) = (π.map fst)(Ω) = μ(Ω) = 1`.

    Reference: Panaretos & Zemel (2020), Ch. 1 §1.2. -/
lemma coupling_isProbabilityMeasure
    {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : MeasureTheory.Measure Ω} (hμ : μ Set.univ = 1)
    {π : MeasureTheory.Measure (Ω × Ω)} (hπ : π ∈ couplingSet μ ν) :
    IsProbabilityMeasure π := by
  constructor
  have h1 : (π.map Prod.fst) Set.univ = μ Set.univ := by rw [hπ.1]
  rwa [MeasureTheory.Measure.map_apply measurable_fst MeasurableSet.univ,
      Set.preimage_univ, hμ] at h1

/-- The squared Wasserstein distance is finite (both marginals have finite second moment).

    Proof: The product coupling μ ⊗ ν is in Π(μ, ν). Its cost is bounded by
    4 · M_μ + 4 · M_ν where M_κ = ∫⁻ (edist x default)² ∂κ (second moment).
    Both M_μ, M_ν < ⊤ by `second_moment_finite`, so the sInf is < ⊤.

    Reference: Panaretos & Zemel (2020), Ch. 2 §2.1. -/
lemma wassersteinDistanceSq_lt_top
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinMeasure Ω) :
    WassersteinDistanceSq μ ν < ⊤ := by
  haveI : IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  have h_prod_mem : μ.measure.prod ν.measure ∈ couplingSet μ.measure ν.measure := by
    refine ⟨?_, ?_⟩
    · rw [MeasureTheory.Measure.map_fst_prod]; simp [ν.is_probability]
    · rw [MeasureTheory.Measure.map_snd_prod]; simp [μ.is_probability]
  set cost := ∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ.measure.prod ν.measure) with hcost
  have h_bound : cost < ⊤ := by
    have h_pointwise : ∀ p : Ω × Ω,
        (edist p.1 p.2) ^ 2 ≤ (4 : ℝ≥0∞) * (edist p.1 default) ^ 2 +
          (4 : ℝ≥0∞) * (edist default p.2) ^ 2 := by
      intro p
      set a := edist p.1 default
      set b := edist default p.2
      have h_tri : edist p.1 p.2 ≤ a + b := edist_triangle p.1 default p.2
      have h_max_sq : (max a b) ^ 2 ≤ a ^ 2 + b ^ 2 := by
        rcases le_total a b with h | h
        · rw [max_eq_right h, add_comm]; exact le_add_of_nonneg_right (by positivity)
        · rw [max_eq_left h]; exact le_add_of_nonneg_right (by positivity)
      calc (edist p.1 p.2) ^ 2 ≤ (a + b) ^ 2 := pow_le_pow_left' h_tri 2
        _ ≤ (2 * max a b) ^ 2 := by
            have h : a + b ≤ 2 * max a b := by
              rcases le_total a b with h | h
              · rw [max_eq_right h]
                calc a + b ≤ b + b := by simpa [add_comm] using add_le_add_right h b
                  _ = 2 * b := by ring
              · rw [max_eq_left h]
                calc a + b ≤ a + a := by simpa [add_comm] using add_le_add_left h a
                  _ = 2 * a := by ring
            exact pow_le_pow_left' h 2
        _ = (4 : ℝ≥0∞) * (max a b) ^ 2 := by ring
        _ ≤ (4 : ℝ≥0∞) * (a ^ 2 + b ^ 2) :=
            mul_le_mul_of_nonneg_left h_max_sq (by norm_num)
        _ = (4 : ℝ≥0∞) * a ^ 2 + (4 : ℝ≥0∞) * b ^ 2 := by ring
    have h_meas_sq1 : AEMeasurable (fun p : Ω × Ω => (4 : ℝ≥0∞) * (edist p.1 default) ^ 2)
        (μ.measure.prod ν.measure) :=
      (measurable_const.mul ((measurable_edist.comp
        (measurable_fst.prodMk measurable_const)).pow_const 2)).aemeasurable
    have h_lintegral_le : cost ≤ (4 : ℝ≥0∞) *
        (∫⁻ p, (edist p.1 default) ^ 2 ∂(μ.measure.prod ν.measure)) +
        (4 : ℝ≥0∞) * (∫⁻ p, (edist default p.2) ^ 2 ∂(μ.measure.prod ν.measure)) := by
      rw [hcost]
      calc ∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ.measure.prod ν.measure)
          ≤ ∫⁻ p, ((4 : ℝ≥0∞) * (edist p.1 default) ^ 2 + (4 : ℝ≥0∞) * (edist default p.2) ^ 2)
              ∂(μ.measure.prod ν.measure) := lintegral_mono h_pointwise
        _ = (∫⁻ p, (4 : ℝ≥0∞) * (edist p.1 default) ^ 2 ∂(μ.measure.prod ν.measure)) +
              (∫⁻ p, (4 : ℝ≥0∞) * (edist default p.2) ^ 2 ∂(μ.measure.prod ν.measure)) :=
            lintegral_add_left' h_meas_sq1 _
        _ = (4 : ℝ≥0∞) * (∫⁻ p, (edist p.1 default) ^ 2 ∂(μ.measure.prod ν.measure)) +
              (4 : ℝ≥0∞) * (∫⁻ p, (edist default p.2) ^ 2 ∂(μ.measure.prod ν.measure)) := by
            simp [MeasureTheory.lintegral_const_mul']
    have h_meas_fst : AEMeasurable (fun p : Ω × Ω => (edist p.1 default) ^ 2)
        (μ.measure.prod ν.measure) :=
      ((measurable_edist.comp (measurable_fst.prodMk measurable_const)).pow_const 2).aemeasurable
    have h_fst : (∫⁻ p, (edist p.1 default) ^ 2 ∂(μ.measure.prod ν.measure)) =
        ∫⁻ x, (edist x default) ^ 2 ∂μ.measure := by
      rw [MeasureTheory.lintegral_prod _ h_meas_fst]; simp [ν.is_probability]
    have h_snd : (∫⁻ p, (edist default p.2) ^ 2 ∂(μ.measure.prod ν.measure)) =
        ∫⁻ y, (edist y default) ^ 2 ∂ν.measure := by
      calc (∫⁻ p, (edist default p.2) ^ 2 ∂(μ.measure.prod ν.measure))
          = (∫⁻ p, (edist p.2 default) ^ 2 ∂(μ.measure.prod ν.measure)) := by
            refine lintegral_congr (fun p => ?_); rw [edist_comm default p.2]
        _ = ∫⁻ y, (edist y default) ^ 2 ∂ν.measure := by
            have h_meas_snd : AEMeasurable (fun p : Ω × Ω => (edist p.2 default) ^ 2)
                (μ.measure.prod ν.measure) :=
              ((measurable_edist.comp (measurable_snd.prodMk measurable_const)).pow_const 2).aemeasurable
            rw [MeasureTheory.lintegral_prod _ h_meas_snd]; simp [μ.is_probability]
    rw [h_fst, h_snd] at h_lintegral_le
    apply lt_of_le_of_lt h_lintegral_le
    have hMμ := μ.second_moment_finite
    have hMν := ν.second_moment_finite
    simp [hMμ, hMν, ENNReal.add_lt_top, ENNReal.mul_lt_top]
  unfold WassersteinDistanceSq
  exact lt_of_le_of_lt (sInf_le ⟨μ.measure.prod ν.measure, h_prod_mem, rfl⟩) h_bound

/-! ### Gluing lemma infrastructure for the triangle inequality

The triangle inequality for `W₂` is proved by the classical *gluing lemma*: given
couplings `π₁ ∈ Π(μ, κ)` and `π₂ ∈ Π(κ, ν)` sharing the middle marginal `κ`, one
disintegrates `π₂` over its first marginal (`Measure.condKernel`) and composes to build
a measure on `Ω × Ω × Ω` whose `(x, z)`-marginal is a coupling of `μ` and `ν`, then applies
Minkowski's inequality to the L² cost.

**Statement change (authorized).** The disintegration `Measure.condKernel` is only available
over a *standard Borel* space, so the triangle inequality (and the resulting `MetricSpace`
instance) carry the extra hypotheses `[OpensMeasurableSpace Ω]` (for measurability of `edist`)
and `[StandardBorelSpace Ω]`. Over a bare `MeasurableSpace` the gluing construction is
unavailable; standard Borel is the standard optimal-transport setting
(Panaretos & Zemel 2020, Ch. 1-2). `[Nonempty Ω]` is supplied by `[Inhabited Ω]`. -/

section GluingHelpers

variable {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
  [OpensMeasurableSpace Ω] [SecondCountableTopology Ω] [StandardBorelSpace Ω]

/-- The squared-distance cost integrand is measurable. -/
private lemma measurable_costFun :
    Measurable (fun p : Ω × Ω => (edist p.1 p.2) ^ 2) :=
  measurable_edist.pow_const 2

/-- Key disintegration identity: integrating `h(y, z)` against the glued measure
`π₁ ⊗ₘ (condKernel π₂ ∘ snd)` equals integrating `h` against `π₂`. -/
private lemma key_disint (κ ν : WassersteinMeasure Ω) {π₁ π₂ : MeasureTheory.Measure (Ω × Ω)}
    [IsProbabilityMeasure π₁] [IsProbabilityMeasure π₂]
    (hsnd1 : π₁.map Prod.snd = κ.measure)
    (hfst2 : π₂.fst = κ.measure)
    {h : Ω × Ω → ℝ≥0∞} (hh : Measurable h) :
    (∫⁻ q : (Ω × Ω) × Ω, h (q.1.2, q.2)
        ∂(π₁ ⊗ₘ (π₂.condKernel.comap Prod.snd measurable_snd)))
      = ∫⁻ p, h p ∂π₂ := by
  haveI : IsProbabilityMeasure κ.measure := ⟨κ.is_probability⟩
  set K := π₂.condKernel.comap Prod.snd measurable_snd with hK
  have hf_meas : Measurable (fun q : (Ω × Ω) × Ω => h (q.1.2, q.2)) :=
    hh.comp ((measurable_snd.comp measurable_fst).prodMk measurable_snd)
  have hG_meas : Measurable (fun y : Ω => ∫⁻ b, h (y, b) ∂(π₂.condKernel y)) :=
    Measurable.lintegral_kernel_prod_right (f := fun y b => h (y, b)) hh
  rw [Measure.lintegral_compProd hf_meas]
  have hinner : (fun a : Ω × Ω => ∫⁻ b, h (a.2, b) ∂(K a))
      = fun a : Ω × Ω => (fun y => ∫⁻ b, h (y, b) ∂(π₂.condKernel y)) a.2 := by
    funext a
    rw [hK, Kernel.comap_apply]
  rw [hinner]
  rw [← lintegral_map hG_meas measurable_snd, hsnd1]
  rw [← Measure.lintegral_compProd hh, ← hfst2, Measure.disintegrate π₂ π₂.condKernel]

/-- Gluing bound: `W₂(μ, ν)` is bounded by the sum of the square roots of the costs of two
couplings sharing the middle marginal. -/
private lemma glue_bound (μ κ ν : WassersteinMeasure Ω)
    {π₁ π₂ : MeasureTheory.Measure (Ω × Ω)}
    (h₁ : π₁ ∈ couplingSet μ.measure κ.measure)
    (h₂ : π₂ ∈ couplingSet κ.measure ν.measure) :
    (WassersteinDistanceSq μ ν) ^ (1/2 : ℝ) ≤
      (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₁) ^ (1/2 : ℝ)
        + (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₂) ^ (1/2 : ℝ) := by
  haveI : IsProbabilityMeasure μ.measure := ⟨μ.is_probability⟩
  haveI : IsProbabilityMeasure κ.measure := ⟨κ.is_probability⟩
  haveI : IsProbabilityMeasure ν.measure := ⟨ν.is_probability⟩
  haveI hp1 : IsProbabilityMeasure π₁ := coupling_isProbabilityMeasure μ.is_probability h₁
  haveI hp2 : IsProbabilityMeasure π₂ := coupling_isProbabilityMeasure κ.is_probability h₂
  have hsnd1 : π₁.map Prod.snd = κ.measure := h₁.2
  have hfst2 : π₂.fst = κ.measure := h₂.1
  set K := π₂.condKernel.comap Prod.snd measurable_snd with hK
  set γ : MeasureTheory.Measure ((Ω × Ω) × Ω) := π₁ ⊗ₘ K with hγ
  set φ : (Ω × Ω) × Ω → Ω × Ω := fun q => (q.1.1, q.2) with hφ
  have hφ_meas : Measurable φ :=
    (measurable_fst.comp measurable_fst).prodMk measurable_snd
  set π : MeasureTheory.Measure (Ω × Ω) := γ.map φ with hπ
  have hγ_fst : γ.map Prod.fst = π₁ := by
    rw [hγ]; exact Measure.fst_compProd π₁ K
  have hπ_fst : π.map Prod.fst = μ.measure := by
    rw [hπ, Measure.map_map measurable_fst hφ_meas]
    have : (Prod.fst ∘ φ) = (Prod.fst ∘ Prod.fst) := by funext q; rfl
    rw [this, ← Measure.map_map measurable_fst measurable_fst, hγ_fst]
    exact h₁.1
  have hψ_meas : Measurable (fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) :=
    (measurable_snd.comp measurable_fst).prodMk measurable_snd
  have hglue2 : γ.map (fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) = π₂ := by
    apply Measure.ext
    intro A hA
    rw [Measure.map_apply hψ_meas hA]
    have hk := key_disint κ ν hsnd1 hfst2 (h := A.indicator (fun _ => (1 : ℝ≥0∞)))
      (measurable_const.indicator hA)
    rw [← hK, ← hγ] at hk
    calc γ ((fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) ⁻¹' A)
        = ∫⁻ q, ((fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) ⁻¹' A).indicator
              (fun _ => (1 : ℝ≥0∞)) q ∂γ := (lintegral_indicator_one (hψ_meas hA)).symm
      _ = ∫⁻ q, A.indicator (fun _ => (1 : ℝ≥0∞)) (q.1.2, q.2) ∂γ := by
          apply lintegral_congr; intro q
          by_cases hq : (q.1.2, q.2) ∈ A
          · have hq' : q ∈ ((fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) ⁻¹' A) := hq
            rw [Set.indicator_of_mem hq', Set.indicator_of_mem hq]
          · have hq' : q ∉ ((fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) ⁻¹' A) := hq
            rw [Set.indicator_of_notMem hq', Set.indicator_of_notMem hq]
      _ = ∫⁻ p, A.indicator (fun _ => (1 : ℝ≥0∞)) p ∂π₂ := hk
      _ = π₂ A := lintegral_indicator_one hA
  have hπ_snd : π.map Prod.snd = ν.measure := by
    rw [hπ, Measure.map_map measurable_snd hφ_meas]
    have hcomp : (Prod.snd ∘ φ)
        = Prod.snd ∘ (fun q : (Ω × Ω) × Ω => (q.1.2, q.2)) := by funext q; rfl
    rw [hcomp, ← Measure.map_map measurable_snd hψ_meas, hglue2]
    exact h₂.2
  have hπ_mem : π ∈ couplingSet μ.measure ν.measure := ⟨hπ_fst, hπ_snd⟩
  have hcostXY : (∫⁻ q : (Ω × Ω) × Ω, (edist q.1.1 q.1.2) ^ 2 ∂γ)
      = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₁ := by
    rw [← hγ_fst]
    exact (lintegral_map measurable_costFun measurable_fst).symm
  have hcostYZ : (∫⁻ q : (Ω × Ω) × Ω, (edist q.1.2 q.2) ^ 2 ∂γ)
      = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₂ := by
    have hk := key_disint κ ν hsnd1 hfst2 measurable_costFun
    rw [← hK, ← hγ] at hk
    exact hk
  have hcostXZ : (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π)
      = ∫⁻ q : (Ω × Ω) × Ω, (edist q.1.1 q.2) ^ 2 ∂γ := by
    rw [hπ]
    exact lintegral_map measurable_costFun hφ_meas
  set fXY : (Ω × Ω) × Ω → ℝ≥0∞ := fun q => edist q.1.1 q.1.2 with hfXY
  set fYZ : (Ω × Ω) × Ω → ℝ≥0∞ := fun q => edist q.1.2 q.2 with hfYZ
  have hfXY_meas : AEMeasurable fXY γ :=
    (measurable_edist.comp
      ((measurable_fst.comp measurable_fst).prodMk (measurable_snd.comp measurable_fst))).aemeasurable
  have hfYZ_meas : AEMeasurable fYZ γ :=
    (measurable_edist.comp
      ((measurable_snd.comp measurable_fst).prodMk measurable_snd)).aemeasurable
  have hmink : (∫⁻ q, (fXY q + fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) ≤
      (∫⁻ q, (fXY q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ))
        + (∫⁻ q, (fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) :=
    ENNReal.lintegral_Lp_add_le hfXY_meas hfYZ_meas (by norm_num)
  have hpt : ∀ q : (Ω × Ω) × Ω, (edist q.1.1 q.2) ^ (2 : ℝ) ≤ (fXY q + fYZ q) ^ (2 : ℝ) := by
    intro q
    apply ENNReal.rpow_le_rpow _ (by norm_num)
    exact edist_triangle q.1.1 q.1.2 q.2
  have hXZmono : (∫⁻ q, (edist q.1.1 q.2) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) ≤
      (∫⁻ q, (fXY q + fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) :=
    ENNReal.rpow_le_rpow (lintegral_mono hpt) (by norm_num)
  have hrw2γ : ∀ (f : (Ω × Ω) × Ω → ℝ≥0∞),
      (∫⁻ q, (f q) ^ (2 : ℝ) ∂γ) = ∫⁻ q, (f q) ^ 2 ∂γ := by
    intro f; apply lintegral_congr; intro q; rw [ENNReal.rpow_two]
  have hcost_bound : (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π) ^ (1 / (2 : ℝ)) ≤
      (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₁) ^ (1 / (2 : ℝ))
        + (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₂) ^ (1 / (2 : ℝ)) := by
    have e1 : (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π) ^ (1 / (2 : ℝ))
        = (∫⁻ q, (edist q.1.1 q.2) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) := by
      rw [hcostXZ, hrw2γ (fun q => edist q.1.1 q.2)]
    have e2 : (∫⁻ q, (fXY q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ))
        = (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₁) ^ (1 / (2 : ℝ)) := by
      rw [hrw2γ fXY, hfXY]; rw [hcostXY]
    have e3 : (∫⁻ q, (fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ))
        = (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π₂) ^ (1 / (2 : ℝ)) := by
      rw [hrw2γ fYZ, hfYZ]; rw [hcostYZ]
    rw [e1]
    calc (∫⁻ q, (edist q.1.1 q.2) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ))
        ≤ (∫⁻ q, (fXY q + fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) := hXZmono
      _ ≤ (∫⁻ q, (fXY q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ))
            + (∫⁻ q, (fYZ q) ^ (2 : ℝ) ∂γ) ^ (1 / (2 : ℝ)) := hmink
      _ = _ := by rw [e2, e3]
  have hA_le : WassersteinDistanceSq μ ν ≤ ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π :=
    sInf_le ⟨π, hπ_mem, rfl⟩
  calc (WassersteinDistanceSq μ ν) ^ (1/2 : ℝ)
      ≤ (∫⁻ p, (edist p.1 p.2) ^ 2 ∂π) ^ (1/2 : ℝ) := ENNReal.rpow_le_rpow hA_le (by norm_num)
    _ ≤ _ := by rw [show (1/2 : ℝ) = 1 / (2 : ℝ) from rfl]; exact hcost_bound

/-- `rpow (1/2)` commutes with `sInf` on `ℝ≥0∞`. -/
private lemma rpow_half_sInf (S : Set ℝ≥0∞) :
    (sInf S) ^ (1/2 : ℝ) = ⨅ c ∈ S, c ^ (1/2 : ℝ) := by
  have hsq : ∀ x : ℝ≥0∞, (x ^ (1/2 : ℝ)) ^ (2 : ℝ) = x := fun x => by
    rw [← ENNReal.rpow_mul, show (1/2 : ℝ) * 2 = 1 by norm_num, ENNReal.rpow_one]
  have hsq2 : ∀ x : ℝ≥0∞, (x ^ (2 : ℝ)) ^ (1/2 : ℝ) = x := fun x => by
    rw [← ENNReal.rpow_mul, show (2 : ℝ) * (1/2) = 1 by norm_num, ENNReal.rpow_one]
  apply le_antisymm
  · exact le_iInf₂ fun c hc => ENNReal.rpow_le_rpow (sInf_le hc) (by norm_num)
  · have key : (⨅ c ∈ S, c ^ (1/2 : ℝ)) ^ (2 : ℝ) ≤ sInf S := by
      refine le_sInf fun c hc => ?_
      calc (⨅ c ∈ S, c ^ (1/2 : ℝ)) ^ (2 : ℝ)
          ≤ (c ^ (1/2 : ℝ)) ^ (2 : ℝ) := ENNReal.rpow_le_rpow (iInf₂_le c hc) (by norm_num)
        _ = c := hsq c
    calc (⨅ c ∈ S, c ^ (1/2 : ℝ))
        = ((⨅ c ∈ S, c ^ (1/2 : ℝ)) ^ (2 : ℝ)) ^ (1/2 : ℝ) := (hsq2 _).symm
      _ ≤ (sInf S) ^ (1/2 : ℝ) := ENNReal.rpow_le_rpow key (by norm_num)

/-- Triangle inequality for the square roots of the squared distances, in `ℝ≥0∞`. -/
private lemma wdsq_triangle_rpow (μ ν κ : WassersteinMeasure Ω) :
    (WassersteinDistanceSq μ ν) ^ (1/2 : ℝ) ≤
      (WassersteinDistanceSq μ κ) ^ (1/2 : ℝ) + (WassersteinDistanceSq κ ν) ^ (1/2 : ℝ) := by
  set A := WassersteinDistanceSq μ ν
  set S₁ := {c : ℝ≥0∞ | ∃ π ∈ couplingSet μ.measure κ.measure,
    c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS₁
  set S₂ := {c : ℝ≥0∞ | ∃ π ∈ couplingSet κ.measure ν.measure,
    c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} with hS₂
  have hBdef : WassersteinDistanceSq μ κ = sInf S₁ := rfl
  have hCdef : WassersteinDistanceSq κ ν = sInf S₂ := rfl
  have step1 : ∀ c₂ ∈ S₂, A ^ (1/2 : ℝ) ≤ (sInf S₁) ^ (1/2 : ℝ) + c₂ ^ (1/2 : ℝ) := by
    rintro c₂ ⟨π₂, hπ₂, rfl⟩
    rw [rpow_half_sInf S₁]
    simp_rw [ENNReal.iInf_add]
    refine le_iInf₂ (fun c₁ hc₁ => ?_)
    obtain ⟨π₁, hπ₁, rfl⟩ := hc₁
    exact glue_bound μ κ ν hπ₁ hπ₂
  rw [hBdef, hCdef, rpow_half_sInf S₂]
  simp_rw [ENNReal.add_iInf]
  exact le_iInf₂ (fun c₂ hc₂ => step1 c₂ hc₂)

/-- The diagonal coupling has zero cost, so `W₂(μ, μ) = 0`. -/
private lemma wasserstein_self_eq_zero (μ : WassersteinMeasure Ω) :
    WassersteinDistance μ μ = 0 := by
  have hdiag_meas : Measurable (fun x : Ω => (x, x)) := measurable_id.prodMk measurable_id
  have hmem : μ.measure.map (fun x : Ω => (x, x)) ∈ couplingSet μ.measure μ.measure := by
    constructor
    · rw [Measure.map_map measurable_fst hdiag_meas]
      have : (Prod.fst ∘ fun x : Ω => (x, x)) = id := by funext x; rfl
      rw [this, Measure.map_id]
    · rw [Measure.map_map measurable_snd hdiag_meas]
      have : (Prod.snd ∘ fun x : Ω => (x, x)) = id := by funext x; rfl
      rw [this, Measure.map_id]
  have hcost : (∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ.measure.map (fun x : Ω => (x, x)))) = 0 := by
    rw [lintegral_map measurable_costFun hdiag_meas]
    simp
  have hsq : WassersteinDistanceSq μ μ = 0 := by
    have hle : WassersteinDistanceSq μ μ ≤ 0 := by
      calc WassersteinDistanceSq μ μ
          ≤ ∫⁻ p, (edist p.1 p.2) ^ 2 ∂(μ.measure.map (fun x : Ω => (x, x))) :=
            sInf_le ⟨μ.measure.map (fun x : Ω => (x, x)), hmem, rfl⟩
        _ = 0 := hcost
    exact nonpos_iff_eq_zero.mp hle
  unfold WassersteinDistance
  rw [hsq]; simp

end GluingHelpers

/-- Wasserstein distance satisfies the triangle inequality.

    W₂(μ, ν) ≤ W₂(μ, κ) + W₂(κ, ν)

    Proof: Gluing lemma (`glue_bound`). Given couplings `π₁ ∈ Π(μ, κ)` and `π₂ ∈ Π(κ, ν)`,
    the disintegration of `π₂` over its first marginal composes with `π₁` to a measure on
    `Ω × Ω × Ω` whose `(x, z)`-pushforward couples `μ` and `ν`; Minkowski's inequality on the
    L² cost gives `W₂(μ,ν) ≤ (cost π₁)^{1/2} + (cost π₂)^{1/2}`. Taking infima over couplings
    (via `rpow_half_sInf`, `ENNReal.iInf_add`) yields the claim in `ℝ≥0∞`; finiteness of all
    three squared distances (`wassersteinDistanceSq_lt_top`) transfers it through `ENNReal.toReal`.

    See the `GluingHelpers` section docstring for the (authorized) standard-Borel hypotheses.

    Reference: Panaretos & Zemel (2020), Theorem 2.3. -/
theorem wasserstein_triangle
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    [CompleteSpace Ω] [SecondCountableTopology Ω]
    [OpensMeasurableSpace Ω] [StandardBorelSpace Ω]
    (μ ν κ : WassersteinMeasure Ω) :
    WassersteinDistance μ ν ≤ WassersteinDistance μ κ + WassersteinDistance κ ν := by
  have hAt : WassersteinDistanceSq μ ν ≠ ⊤ := (wassersteinDistanceSq_lt_top μ ν).ne
  have hBt : WassersteinDistanceSq μ κ ≠ ⊤ := (wassersteinDistanceSq_lt_top μ κ).ne
  have hCt : WassersteinDistanceSq κ ν ≠ ⊤ := (wassersteinDistanceSq_lt_top κ ν).ne
  have hAr : (WassersteinDistanceSq μ ν) ^ (1/2 : ℝ) ≠ ⊤ :=
    (ENNReal.rpow_lt_top_of_nonneg (by norm_num) hAt).ne
  have hBr : (WassersteinDistanceSq μ κ) ^ (1/2 : ℝ) ≠ ⊤ :=
    (ENNReal.rpow_lt_top_of_nonneg (by norm_num) hBt).ne
  have hCr : (WassersteinDistanceSq κ ν) ^ (1/2 : ℝ) ≠ ⊤ :=
    (ENNReal.rpow_lt_top_of_nonneg (by norm_num) hCt).ne
  have htri := wdsq_triangle_rpow μ ν κ
  unfold WassersteinDistance
  rw [ENNReal.toReal_rpow, ENNReal.toReal_rpow, ENNReal.toReal_rpow]
  rw [← ENNReal.toReal_add hBr hCr]
  exact (ENNReal.toReal_le_toReal hAr (by rw [ENNReal.add_ne_top]; exact ⟨hBr, hCr⟩)).mpr htri

/-- A coupling achieving the Wasserstein distance.

    An optimal coupling π ∈ Π(μ, ν) satisfies:
    * π has first marginal μ and second marginal ν
    * ∫ ∥x - y∥² dπ(x, y) = W₂²(μ, ν)

    Existence of optimal couplings follows from the direct method in the
    calculus of variations (tightness of couplings + lower semicontinuity of the cost).

    Reference: Panaretos & Zemel (2020), Theorem 1.2. -/
structure OptimalCoupling {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω]
    [Inhabited Ω] (μ ν : WassersteinMeasure Ω) where
  coupling : MeasureTheory.Measure (Ω × Ω)
  marginal_fst : coupling.map Prod.fst = μ.measure
  marginal_snd : coupling.map Prod.snd = ν.measure
  is_optimal : (∫⁻ p, (edist p.1 p.2)^2 ∂coupling) = WassersteinDistanceSq μ ν

/-- The Wasserstein space carries a pseudometric without an unproved separation claim.

The generic carrier retains the distance, symmetry, and triangle APIs used by the energy bridge.
Metric separation is a real-line/Borel theorem and is intentionally not asserted here.

Reference: Panaretos & Zemel (2020), Theorem 2.2. -/
noncomputable instance WassersteinSpace.instPseudoMetricSpace
    {Ω : Type*} [MeasurableSpace Ω] [MetricSpace Ω] [Inhabited Ω]
    [CompleteSpace Ω] [SecondCountableTopology Ω] [BorelSpace Ω] :
    PseudoMetricSpace (WassersteinMeasure Ω) where
  dist := WassersteinDistance
  dist_self := wasserstein_self_eq_zero
  dist_comm := wasserstein_symm
  dist_triangle := fun x y z ↦ wasserstein_triangle x z y

/-- A constant-speed geodesic in Wasserstein space.

    γ : [0, 1] → P₂(Ω) such that:
    * γ(0) = μ, γ(1) = ν
    * W₂(γ(s), γ(t)) = |t - s| · W₂(μ, ν) for all s, t ∈ [0, 1]

    Reference: Panaretos & Zemel (2020), Definition 2.11. -/
structure Geodesic {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω]
    [Inhabited Ω] [CompleteSpace Ω] [SecondCountableTopology Ω]
    (μ ν : WassersteinMeasure Ω) where
  path : ℝ → WassersteinMeasure Ω
  start : path 0 = μ
  end_ : path 1 = ν
  constant_speed : ∀ s t : ℝ, 0 ≤ s → s ≤ t → t ≤ 1 →
    WassersteinDistance (path s) (path t) = (t - s) * WassersteinDistance μ ν

/-- A measurable finite-cost Kantorovich dual certificate.

The generic Wasserstein layer exposes the corrected contract rather than the former unconditional
duality statement.  Potentials are measurable, their extended integrals are finite, and the
certificate records equality with the primal cost.  Real-line quantile constructions may provide
such a certificate without changing generic coupling definitions.

Reference: Villani (2003), Theorem 1.3; Panaretos & Zemel (2020), Chapter 2 §2.2. -/
structure KantorovichDualCertificate
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    (μ ν : WassersteinMeasure Ω) where
  potential₁ : Ω → ℝ≥0∞
  potential₂ : Ω → ℝ≥0∞
  measurable₁ : Measurable potential₁
  measurable₂ : Measurable potential₂
  admissible : ∀ x y, potential₁ x + potential₂ y ≤ (edist x y) ^ 2
  potential₁_integral_finite : (∫⁻ x, potential₁ x ∂μ.measure) < ⊤
  potential₂_integral_finite : (∫⁻ y, potential₂ y ∂ν.measure) < ⊤
  dual_value :
    ∫⁻ x, potential₁ x ∂μ.measure + ∫⁻ y, potential₂ y ∂ν.measure =
      WassersteinDistanceSq μ ν

/-- The existence proposition for a corrected finite-cost dual certificate. -/
def HasKantorovichDualCertificate
    {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
    (μ ν : WassersteinMeasure Ω) : Prop :=
  Nonempty (KantorovichDualCertificate μ ν)

end WassersteinGeometry
