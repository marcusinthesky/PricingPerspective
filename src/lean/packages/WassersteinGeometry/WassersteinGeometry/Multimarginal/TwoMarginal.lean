import WassersteinGeometry.Multimarginal.Pairwise

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory
open MeasureTheory Finset

/-!
# Two-marginal specialization of quadratic dispersion

For two coordinates, a coherent joint coupling is just an ordinary coupling.  This file
constructs that correspondence and proves that the multi-marginal dispersion reduces to the
product of the two weights and the squared Wasserstein distance.
-/

namespace WassersteinGeometry.Multimarginal

variable {Ω : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]

/-- Convert a pair into a function on the two-element index type. -/
def pairToFinTwo (p : Ω × Ω) : Fin 2 → Ω :=
  (MeasurableEquiv.finTwoArrow (α := Ω)).symm p

omit [PseudoMetricSpace Ω] [Inhabited Ω] in
/-- The conversion from pairs to two-coordinate functions is measurable. -/
lemma measurable_pairToFinTwo : Measurable (pairToFinTwo (Ω := Ω)) := by
  exact (MeasurableEquiv.finTwoArrow (α := Ω)).symm.measurable

/-- An ordinary coupling of the two marginals induces a coherent two-coordinate coupling. -/
noncomputable def jointCouplingOfPair
    (P : Fin 2 → WassersteinMeasure Ω) (π : Measure (Ω × Ω))
    (hπ : π ∈ couplingSet (P 0).measure (P 1).measure) : JointCoupling P where
  measure := π.map pairToFinTwo
  marginal := Fin.forall_fin_two.mpr ⟨by
    rw [Measure.map_map (measurable_pi_apply 0) measurable_pairToFinTwo]
    change π.map Prod.fst = (P 0).measure
    exact hπ.1, by
    rw [Measure.map_map (measurable_pi_apply 1) measurable_pairToFinTwo]
    change π.map Prod.snd = (P 1).measure
    exact hπ.2⟩

/-- The cost of the induced two-coordinate plan is the pair cost times the weight product. -/
theorem weightedDispersionCost_jointCouplingOfPair
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (q : ProbabilityWeight (Fin 2)) (P : Fin 2 → WassersteinMeasure Ω)
    (π : Measure (Ω × Ω)) (hπ : π ∈ couplingSet (P 0).measure (P 1).measure) :
    weightedDispersionCost q (jointCouplingOfPair P π hπ) =
      ENNReal.ofReal (q 0 * q 1) * ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π := by
  have hcost (a b : Fin 2) :
      Measurable (fun x : Fin 2 → Ω ↦ (edist (x a) (x b)) ^ 2) :=
    (measurable_edist.comp ((measurable_pi_apply a).prodMk
      (measurable_pi_apply b))).pow_const 2
  unfold weightedDispersionCost jointCouplingOfPair
  simp only [Fin.sum_univ_two]
  simp_rw [lintegral_map (hcost _ _) measurable_pairToFinTwo]
  simp only [pairToFinTwo, MeasurableEquiv.finTwoArrow_symm_apply, Fin.cons_zero,
    Fin.cons_one, edist_self]
  have hsym : (∫⁻ p, (edist p.2 p.1) ^ 2 ∂π) =
      ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π :=
    lintegral_congr fun p ↦ by rw [edist_comm]
  rw [hsym]
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), lintegral_zero, mul_zero,
    zero_add, add_zero, ← add_mul]
  congr 1
  have h01 : 0 ≤ q 0 * q 1 / 2 :=
    div_nonneg (mul_nonneg (q.nonneg 0) (q.nonneg 1)) (by norm_num)
  have h10 : 0 ≤ q 1 * q 0 / 2 :=
    div_nonneg (mul_nonneg (q.nonneg 1) (q.nonneg 0)) (by norm_num)
  rw [← ENNReal.ofReal_add h01 h10]
  congr 1
  ring

/-- Every coherent two-coordinate plan has the pair cost times the weight product as its cost. -/
theorem weightedDispersionCost_finTwo
    (q : ProbabilityWeight (Fin 2)) {P : Fin 2 → WassersteinMeasure Ω}
    (J : JointCoupling P) :
    weightedDispersionCost q J =
      ENNReal.ofReal (q 0 * q 1) *
        ∫⁻ x, (edist (x 0) (x 1)) ^ 2 ∂J.measure := by
  unfold weightedDispersionCost
  simp only [Fin.sum_univ_two, edist_self]
  have hsym : (∫⁻ x, (edist (x 1) (x 0)) ^ 2 ∂J.measure) =
      ∫⁻ x, (edist (x 0) (x 1)) ^ 2 ∂J.measure :=
    lintegral_congr fun x ↦ by rw [edist_comm]
  rw [hsym]
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), lintegral_zero, mul_zero,
    zero_add, add_zero, ← add_mul]
  congr 1
  have h01 : 0 ≤ q 0 * q 1 / 2 :=
    div_nonneg (mul_nonneg (q.nonneg 0) (q.nonneg 1)) (by norm_num)
  have h10 : 0 ≤ q 1 * q 0 / 2 :=
    div_nonneg (mul_nonneg (q.nonneg 1) (q.nonneg 0)) (by norm_num)
  rw [← ENNReal.ofReal_add h01 h10]
  congr 1
  ring

/-- The weighted two-marginal Wasserstein cost bounds the cost of every coherent plan. -/
theorem weightedWasserstein_le_weightedDispersionCost_finTwo
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (q : ProbabilityWeight (Fin 2)) {P : Fin 2 → WassersteinMeasure Ω}
    (J : JointCoupling P) :
    ENNReal.ofReal (q 0 * q 1) * WassersteinDistanceSq (P 0) (P 1) ≤
      weightedDispersionCost q J := by
  rw [weightedDispersionCost_finTwo]
  exact mul_le_mul_right (wassersteinDistanceSq_le_pairCost J 0 1) _

/-- For two marginals, multi-marginal dispersion is exactly
`q₀ q₁ W₂(P₀,P₁)²`; no optimal-coupling existence theorem is required. -/
theorem wassersteinDispersionSq_finTwo
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    (q : ProbabilityWeight (Fin 2)) (P : Fin 2 → WassersteinMeasure Ω) :
    wassersteinDispersionSq q P =
      ENNReal.ofReal (q 0 * q 1) * WassersteinDistanceSq (P 0) (P 1) := by
  let c : ℝ≥0∞ := ENNReal.ofReal (q 0 * q 1)
  let S : Set ℝ≥0∞ :=
    {r | ∃ π ∈ couplingSet (P 0).measure (P 1).measure,
      r = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π}
  have hW : WassersteinDistanceSq (P 0) (P 1) = sInf S := rfl
  apply le_antisymm
  · change wassersteinDispersionSq q P ≤ c * WassersteinDistanceSq (P 0) (P 1)
    have key : ∀ r ∈ S, wassersteinDispersionSq q P ≤ c * r := by
      rintro r ⟨π, hπ, rfl⟩
      unfold wassersteinDispersionSq
      refine sInf_le ⟨jointCouplingOfPair P π hπ, ?_⟩
      simpa [c] using (weightedDispersionCost_jointCouplingOfPair q P π hπ).symm
    haveI : IsProbabilityMeasure (P 0).measure := ⟨(P 0).is_probability⟩
    haveI : IsProbabilityMeasure (P 1).measure := ⟨(P 1).is_probability⟩
    have hprod : (P 0).measure.prod (P 1).measure ∈
        couplingSet (P 0).measure (P 1).measure :=
      ⟨by rw [Measure.map_fst_prod]; simp [(P 1).is_probability],
        by rw [Measure.map_snd_prod]; simp [(P 0).is_probability]⟩
    have hmemS :
        (∫⁻ p, (edist p.1 p.2) ^ 2 ∂((P 0).measure.prod (P 1).measure)) ∈ S :=
      ⟨_, hprod, rfl⟩
    rcases eq_or_ne c 0 with hc | hc
    · simpa [hc] using key _ hmemS
    · have htop : c ≠ ⊤ := by simp [c]
      calc
        wassersteinDispersionSq q P ≤ ⨅ r : S, c * (r : ℝ≥0∞) :=
          le_iInf fun r ↦ key r r.2
        _ = c * ⨅ r : S, (r : ℝ≥0∞) := (ENNReal.mul_iInf_of_ne hc htop).symm
        _ = c * sInf S := by rw [← sInf_eq_iInf']
        _ = c * WassersteinDistanceSq (P 0) (P 1) := by
          rw [← hW]
  · unfold wassersteinDispersionSq
    refine le_sInf ?_
    rintro r ⟨J, rfl⟩
    exact weightedWasserstein_le_weightedDispersionCost_finTwo q J

end WassersteinGeometry.Multimarginal
