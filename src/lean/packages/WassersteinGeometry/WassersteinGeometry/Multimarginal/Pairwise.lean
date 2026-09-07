import WassersteinGeometry.Multimarginal.Defs

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory Finset

/-!
# Pairwise certificates for multi-marginal dispersion

Every pair marginal of one coherent joint law is an admissible two-marginal
coupling.  Consequently the weighted sum of pairwise squared Wasserstein
distances is a computable lower certificate for the multi-marginal dispersion.
-/

namespace WassersteinGeometry.Multimarginal

variable {A Ω : Type*} [Fintype A]
  [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]

omit [Fintype A] in
/-- The `(a,b)` marginal of a coherent joint coupling couples the prescribed laws. -/
lemma pairMarginal_mem_couplingSet {P : A → WassersteinMeasure Ω}
    (J : JointCoupling P) (a b : A) :
    J.measure.map (fun x : A → Ω ↦ (x a, x b)) ∈
      couplingSet (P a).measure (P b).measure := by
  have hpair : Measurable fun x : A → Ω ↦ (x a, x b) :=
    (measurable_pi_apply a).prodMk (measurable_pi_apply b)
  constructor
  · rw [Measure.map_map measurable_fst hpair]
    change J.measure.map (fun x : A → Ω ↦ x a) = (P a).measure
    exact J.marginal a
  · rw [Measure.map_map measurable_snd hpair]
    change J.measure.map (fun x : A → Ω ↦ x b) = (P b).measure
    exact J.marginal b

/-- The squared Wasserstein distance of two marginals is no larger than their
coordinate-pair cost inside any coherent joint coupling. -/
theorem wassersteinDistanceSq_le_pairCost
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {P : A → WassersteinMeasure Ω} (J : JointCoupling P) (a b : A) :
    WassersteinDistanceSq (P a) (P b) ≤
      ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure := by
  unfold WassersteinDistanceSq
  have hpair : Measurable fun x : A → Ω ↦ (x a, x b) :=
    (measurable_pi_apply a).prodMk (measurable_pi_apply b)
  have hcost : Measurable fun p : Ω × Ω ↦ (edist p.1 p.2) ^ 2 :=
    measurable_edist.pow_const 2
  refine le_trans (sInf_le ⟨J.measure.map (fun x : A → Ω ↦ (x a, x b)),
    pairMarginal_mem_couplingSet J a b, rfl⟩) ?_
  rw [lintegral_map hcost hpair]

/-- The computable weighted pairwise Wasserstein certificate. -/
noncomputable def pairwiseCertificate
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) : ℝ≥0∞ :=
  ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) * WassersteinDistanceSq (P a) (P b)

/-- Every coherent joint plan costs at least the weighted pairwise certificate. -/
theorem pairwiseCertificate_le_weightedDispersionCost
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (q : ProbabilityWeight A) {P : A → WassersteinMeasure Ω} (J : JointCoupling P) :
    pairwiseCertificate q P ≤ weightedDispersionCost q J := by
  unfold pairwiseCertificate weightedDispersionCost
  refine Finset.sum_le_sum fun a _ ↦ ?_
  refine Finset.sum_le_sum fun b _ ↦ ?_
  gcongr
  exact wassersteinDistanceSq_le_pairCost J a b

/-- Pairwise squared Wasserstein distances lower-bound multi-marginal dispersion. -/
theorem pairwiseCertificate_le_wassersteinDispersionSq
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure Ω) :
    pairwiseCertificate q P ≤ wassersteinDispersionSq q P := by
  unfold wassersteinDispersionSq
  refine le_sInf ?_
  rintro c ⟨J, rfl⟩
  exact pairwiseCertificate_le_weightedDispersionCost q J

end WassersteinGeometry.Multimarginal
