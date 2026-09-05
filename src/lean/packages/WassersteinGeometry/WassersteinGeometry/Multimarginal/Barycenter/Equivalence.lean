import WassersteinGeometry.Gluing
import WassersteinGeometry.Multimarginal.Barycenter.Basic

set_option linter.style.longLine false

open scoped ENNReal InnerProductSpace MeasureTheory NNReal
open Finset MeasureTheory

/-!
# Unconditional Hilbert MMOT--barycenter equivalence

This file proves equality between the infimal weighted Frechet objective and finite
multi-marginal quadratic dispersion without assuming either infimum is attained.  The reverse
inequality uses epsilon-optimal pairwise couplings and finite common-first-marginal gluing.
-/

namespace WassersteinGeometry.Multimarginal

variable {A H : Type*} [Fintype A]
  [MeasurableSpace H] [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [CompleteSpace H] [SecondCountableTopology H] [BorelSpace H] [Inhabited H]

private lemma ProbabilityWeight.nonempty (q : ProbabilityWeight A) : Nonempty A := by
  rw [← Finset.univ_nonempty_iff]
  by_contra h
  have hmass := q.mass_one
  rw [Finset.not_nonempty_iff_eq_empty.mp h] at hmass
  simp at hmass

omit [CompleteSpace H] [Inhabited H] in
private lemma measurable_weightedHilbertMean (q : ProbabilityWeight A) :
    Measurable (weightedHilbertMean q : (A → H) → H) := by
  unfold weightedHilbertMean
  fun_prop

omit [MeasurableSpace H] [InnerProductSpace ℝ H] [CompleteSpace H]
  [SecondCountableTopology H] [BorelSpace H] [Inhabited H] in
private lemma edist_sq_eq_ofReal_norm_sq (x y : H) :
    (edist x y) ^ 2 = ENNReal.ofReal (‖x - y‖ ^ 2) := by
  rw [edist_dist, dist_eq_norm, ← ENNReal.ofReal_pow (norm_nonneg _)]

omit [MeasurableSpace H] [CompleteSpace H] [SecondCountableTopology H]
  [BorelSpace H] [Inhabited H] in
private lemma weighted_pairwise_eq_centered (q : ProbabilityWeight A) (z : A → H) :
    ∑ a, ∑ b, q a * q b / 2 * ‖z a - z b‖ ^ 2 =
      ∑ a, q a * ‖z a - weightedHilbertMean q z‖ ^ 2 := by
  have hvar := weighted_variance_decomposition q z 0
  have hpol := weighted_polarization q z
  simp only [sub_zero, weightedHilbertMean] at hvar ⊢
  calc
    ∑ a, ∑ b, q a * q b / 2 * ‖z a - z b‖ ^ 2 =
        (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * ‖z a - z b‖ ^ 2 := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro b _
      ring
    _ = ∑ a, q a * ‖z a - ∑ b, q b • z b‖ ^ 2 := by linarith

omit [MeasurableSpace H] [CompleteSpace H] [SecondCountableTopology H]
  [BorelSpace H] [Inhabited H] in
private lemma weighted_pairwiseENNReal_eq_centered (q : ProbabilityWeight A) (z : A → H) :
    ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) * (edist (z a) (z b)) ^ 2 =
      ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 := by
  have hpair_nonneg (a b : A) : 0 ≤ q a * q b / 2 :=
    div_nonneg (mul_nonneg (q.nonneg a) (q.nonneg b)) (by norm_num)
  calc
    ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) * (edist (z a) (z b)) ^ 2 =
        ENNReal.ofReal (∑ a, ∑ b, q a * q b / 2 * ‖z a - z b‖ ^ 2) := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      · apply Finset.sum_congr rfl
        intro a _
        rw [ENNReal.ofReal_sum_of_nonneg]
        · apply Finset.sum_congr rfl
          intro b _
          rw [edist_sq_eq_ofReal_norm_sq,
            ← ENNReal.ofReal_mul (hpair_nonneg a b)]
        · exact fun b _ ↦ mul_nonneg (hpair_nonneg a b) (sq_nonneg _)
      · intro a _
        exact Finset.sum_nonneg fun b _ ↦
          mul_nonneg (hpair_nonneg a b) (sq_nonneg _)
    _ = ENNReal.ofReal
        (∑ a, q a * ‖z a - weightedHilbertMean q z‖ ^ 2) := by
      rw [weighted_pairwise_eq_centered q z]
    _ = ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      · apply Finset.sum_congr rfl
        intro a _
        rw [edist_sq_eq_ofReal_norm_sq, ← ENNReal.ofReal_mul (q.nonneg a)]
      · exact fun a _ ↦ mul_nonneg (q.nonneg a) (sq_nonneg _)

omit [CompleteSpace H] [Inhabited H] in
private lemma measurable_centeredTerm (q : ProbabilityWeight A) (a : A) :
    Measurable (fun z : A → H ↦
      ENNReal.ofReal (q a) * (edist (z a) (weightedHilbertMean q z)) ^ 2) := by
  exact measurable_const.mul
    (((measurable_pi_apply a).edist (measurable_weightedHilbertMean q)).pow_const 2)

omit [CompleteSpace H] in
private lemma weightedDispersionCost_eq_lintegral_centered
    (q : ProbabilityWeight A) {P : A → WassersteinMeasure H} (J : JointCoupling P) :
    weightedDispersionCost q J =
      ∫⁻ z, ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 ∂J.measure := by
  unfold weightedDispersionCost
  calc
    ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
        ∫⁻ z, (edist (z a) (z b)) ^ 2 ∂J.measure =
        ∑ a, ∑ b, ∫⁻ z, ENNReal.ofReal (q a * q b / 2) *
          (edist (z a) (z b)) ^ 2 ∂J.measure := by
      apply Finset.sum_congr rfl
      intro a _
      apply Finset.sum_congr rfl
      intro b _
      rw [lintegral_const_mul]
      fun_prop
    _ = ∑ a, ∫⁻ z, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          (edist (z a) (z b)) ^ 2 ∂J.measure := by
      apply Finset.sum_congr rfl
      intro a _
      exact (lintegral_finsetSum Finset.univ fun b _ ↦ by fun_prop).symm
    _ = ∫⁻ z, ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          (edist (z a) (z b)) ^ 2 ∂J.measure := by
      exact (lintegral_finsetSum Finset.univ fun a _ ↦
        Finset.measurable_fun_sum _ fun b _ ↦ by fun_prop).symm
    _ = ∫⁻ z, ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 ∂J.measure := by
      apply lintegral_congr
      exact weighted_pairwiseENNReal_eq_centered q

omit [InnerProductSpace ℝ H] [CompleteSpace H] [SecondCountableTopology H]
  [BorelSpace H] in
private lemma joint_measure_univ (q : ProbabilityWeight A)
    {P : A → WassersteinMeasure H} (J : JointCoupling P) :
    J.measure Set.univ = 1 := by
  let a := Classical.choice q.nonempty
  have h := congrArg (fun μ : Measure H ↦ μ Set.univ) (J.marginal a)
  simpa [Measure.map_apply (measurable_pi_apply a) MeasurableSet.univ,
    (P a).is_probability] using h

private noncomputable def jointCenter (q : ProbabilityWeight A)
    {P : A → WassersteinMeasure H} (J : JointCoupling P) : WassersteinMeasure H where
  measure := J.measure.map (weightedHilbertMean q)
  is_probability := by
    rw [Measure.map_apply (measurable_weightedHilbertMean q) MeasurableSet.univ,
      Set.preimage_univ, joint_measure_univ q J]
  second_moment_finite := by
    classical
    have hcost : Measurable (fun x : H ↦ (edist x default) ^ 2) := by fun_prop
    rw [lintegral_map hcost (measurable_weightedHilbertMean q)]
    have hpoint (z : A → H) :
        (edist (weightedHilbertMean q z) default) ^ 2 ≤
          ∑ a, ENNReal.ofReal (q a) * (edist (z a) default) ^ 2 := by
      have hvar := weighted_variance_decomposition q z default
      have hreal : ‖weightedHilbertMean q z - default‖ ^ 2 ≤
          ∑ a, q a * ‖z a - default‖ ^ 2 := by
        simp only [weightedHilbertMean] at hvar ⊢
        have hc : 0 ≤ ∑ a, q a * ‖z a - ∑ b, q b • z b‖ ^ 2 :=
          Finset.sum_nonneg fun a _ ↦
            mul_nonneg (q.nonneg a) (sq_nonneg ‖z a - ∑ b, q b • z b‖)
        linarith
      rw [edist_sq_eq_ofReal_norm_sq]
      calc
        ENNReal.ofReal (‖weightedHilbertMean q z - default‖ ^ 2) ≤
            ENNReal.ofReal (∑ a, q a * ‖z a - default‖ ^ 2) :=
          ENNReal.ofReal_le_ofReal hreal
        _ = ∑ a, ENNReal.ofReal (q a) * (edist (z a) default) ^ 2 := by
          rw [ENNReal.ofReal_sum_of_nonneg]
          · apply Finset.sum_congr rfl
            intro a _
            rw [edist_sq_eq_ofReal_norm_sq,
              ← ENNReal.ofReal_mul (q.nonneg a)]
          · exact fun a _ ↦ mul_nonneg (q.nonneg a) (sq_nonneg _)
    refine lt_of_le_of_lt (lintegral_mono hpoint) ?_
    rw [lintegral_finsetSum]
    · have hcoord (a : A) : Measurable (fun z : A → H ↦ (edist (z a) default) ^ 2) :=
        (measurable_edist.comp ((measurable_pi_apply a).prodMk measurable_const)).pow_const 2
      have hmul (a : A) :
          ∫⁻ z, ENNReal.ofReal (q a) * (edist (z a) default) ^ 2 ∂J.measure =
            ENNReal.ofReal (q a) * ∫⁻ z, (edist (z a) default) ^ 2 ∂J.measure :=
        lintegral_const_mul _ (hcoord a)
      have hmarg (a : A) :
          ∫⁻ z, (edist (z a) default) ^ 2 ∂J.measure =
            ∫⁻ x, (edist x default) ^ 2 ∂(P a).measure := by
        rw [← J.marginal a, lintegral_map hcost (measurable_pi_apply a)]
      simp_rw [hmul, hmarg]
      induction (Finset.univ : Finset A) using Finset.induction_on with
      | empty => simp
      | @insert a s ha ih =>
          rw [Finset.sum_insert ha]
          exact ENNReal.add_lt_top.2 ⟨
            ENNReal.mul_lt_top ENNReal.ofReal_lt_top (P a).second_moment_finite, ih⟩
    · exact fun a _ ↦ by fun_prop

omit [CompleteSpace H] in
private lemma weightedFrechetCostENNReal_jointCenter_le
    (q : ProbabilityWeight A) {P : A → WassersteinMeasure H} (J : JointCoupling P) :
    weightedFrechetCostENNReal q P (jointCenter q J) ≤ weightedDispersionCost q J := by
  unfold weightedFrechetCostENNReal
  calc
    ∑ a, ENNReal.ofReal (q a) * WassersteinDistanceSq (jointCenter q J) (P a) ≤
        ∑ a, ENNReal.ofReal (q a) *
          ∫⁻ z, (edist (weightedHilbertMean q z) (z a)) ^ 2 ∂J.measure := by
      apply Finset.sum_le_sum
      intro a _
      gcongr
      unfold WassersteinDistanceSq
      let π : Measure (H × H) :=
        J.measure.map (fun z ↦ (weightedHilbertMean q z, z a))
      have hpair : Measurable (fun z : A → H ↦ (weightedHilbertMean q z, z a)) := by
        exact (measurable_weightedHilbertMean q).prodMk (measurable_pi_apply a)
      have hπ : π ∈ couplingSet (jointCenter q J).measure (P a).measure := by
        constructor
        · dsimp [π, jointCenter]
          rw [Measure.map_map measurable_fst hpair]
          rfl
        · dsimp [π]
          rw [Measure.map_map measurable_snd hpair]
          change J.measure.map (fun z ↦ z a) = (P a).measure
          exact J.marginal a
      calc
        sInf {c | ∃ π ∈ couplingSet (jointCenter q J).measure (P a).measure,
            c = ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π} ≤
            ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π := sInf_le ⟨π, hπ, rfl⟩
        _ = ∫⁻ z, (edist (weightedHilbertMean q z) (z a)) ^ 2 ∂J.measure := by
          dsimp [π]
          rw [lintegral_map (by fun_prop) hpair]
    _ = ∫⁻ z, ∑ a, ENNReal.ofReal (q a) *
          (edist (z a) (weightedHilbertMean q z)) ^ 2 ∂J.measure := by
      rw [lintegral_finsetSum]
      · apply Finset.sum_congr rfl
        intro a _
        rw [lintegral_const_mul]
        · congr 1
          apply lintegral_congr
          intro z
          rw [edist_comm]
        · exact ((measurable_pi_apply a).edist
            (measurable_weightedHilbertMean q)).pow_const 2
      · exact fun a _ ↦ measurable_centeredTerm q a
    _ = weightedDispersionCost q J :=
      (weightedDispersionCost_eq_lintegral_centered q J).symm

omit [MeasurableSpace H] [CompleteSpace H] [SecondCountableTopology H]
  [BorelSpace H] [Inhabited H] in
private lemma centeredENNReal_le_radius (q : ProbabilityWeight A) (z : A → H) (y : H) :
    ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 ≤
      ∑ a, ENNReal.ofReal (q a) * (edist (z a) y) ^ 2 := by
  have hvar := weighted_variance_decomposition q z y
  have hreal : ∑ a, q a * ‖z a - weightedHilbertMean q z‖ ^ 2 ≤
      ∑ a, q a * ‖z a - y‖ ^ 2 := by
    simp only [weightedHilbertMean] at hvar ⊢
    nlinarith [sq_nonneg ‖(∑ b, q b • z b) - y‖]
  calc
    ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2 =
        ENNReal.ofReal (∑ a, q a * ‖z a - weightedHilbertMean q z‖ ^ 2) := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      · apply Finset.sum_congr rfl
        intro a _
        rw [edist_sq_eq_ofReal_norm_sq, ← ENNReal.ofReal_mul (q.nonneg a)]
      · exact fun a _ ↦ mul_nonneg (q.nonneg a) (sq_nonneg _)
    _ ≤ ENNReal.ofReal (∑ a, q a * ‖z a - y‖ ^ 2) :=
      ENNReal.ofReal_le_ofReal hreal
    _ = ∑ a, ENNReal.ofReal (q a) * (edist (z a) y) ^ 2 := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      · apply Finset.sum_congr rfl
        intro a _
        rw [edist_sq_eq_ofReal_norm_sq, ← ENNReal.ofReal_mul (q.nonneg a)]
      · exact fun a _ ↦ mul_nonneg (q.nonneg a) (sq_nonneg _)

private lemma exists_joint_cost_le_frechet_add
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (Q : WassersteinMeasure H) (ε : ℝ≥0) (hε : 0 < ε) :
    ∃ J : JointCoupling P,
      weightedDispersionCost q J ≤ weightedFrechetCostENNReal q P Q + (ε : ℝ≥0∞) := by
  classical
  choose π hπ hcost using fun a ↦ exists_coupling_cost_lt_add Q (P a) ε hε
  letI : IsProbabilityMeasure Q.measure := ⟨Q.is_probability⟩
  letI : ∀ a, IsProbabilityMeasure (π a) := fun a ↦
    coupling_isProbabilityMeasure Q.is_probability (hπ a)
  have hfst (a : A) : (π a).fst = Q.measure := (hπ a).1
  obtain ⟨η, hη, hηmarg⟩ := Measure.exists_common_fst_fintype_gluing
    (A := A) (X := H) (Y := H) Q.measure π hfst
  letI : IsProbabilityMeasure η := hη
  let J : JointCoupling P :=
    { measure := η.map Prod.snd
      marginal := fun a ↦ by
        rw [Measure.map_map (measurable_pi_apply a) measurable_snd]
        calc
          η.map (fun p : H × (A → H) ↦ p.2 a) =
              (η.map (fun p ↦ (p.1, p.2 a))).map Prod.snd := by
            rw [Measure.map_map measurable_snd]
            · rfl
            · fun_prop
          _ = (π a).map Prod.snd := by rw [hηmarg a]
          _ = (P a).measure := (hπ a).2 }
  refine ⟨J, ?_⟩
  rw [weightedDispersionCost_eq_lintegral_centered]
  change (∫⁻ z, ∑ a, ENNReal.ofReal (q a) *
      (edist (z a) (weightedHilbertMean q z)) ^ 2 ∂η.map Prod.snd) ≤ _
  have hcentered : Measurable (fun z : A → H ↦
      ∑ a, ENNReal.ofReal (q a) *
        (edist (z a) (weightedHilbertMean q z)) ^ 2) :=
    Finset.measurable_fun_sum _ fun a _ ↦ measurable_centeredTerm q a
  rw [lintegral_map hcentered measurable_snd]
  refine (lintegral_mono (fun p : H × (A → H) ↦
    centeredENNReal_le_radius (A := A) (H := H) q p.2 p.1)).trans ?_
  calc
    ∫⁻ p, ∑ a, ENNReal.ofReal (q a) * (edist (p.2 a) p.1) ^ 2 ∂η =
        ∑ a, ENNReal.ofReal (q a) *
          ∫⁻ p, (edist (p.2 a) p.1) ^ 2 ∂η := by
      rw [lintegral_finsetSum]
      · apply Finset.sum_congr rfl
        intro a _
        rw [lintegral_const_mul]
        fun_prop
      · intro a _
        fun_prop
    _ = ∑ a, ENNReal.ofReal (q a) *
          ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π a := by
      apply Finset.sum_congr rfl
      intro a _
      congr 1
      rw [← hηmarg a, lintegral_map]
      · apply lintegral_congr
        intro p
        rw [edist_comm]
      · fun_prop
      · fun_prop
    _ ≤ ∑ a, ENNReal.ofReal (q a) *
          (WassersteinDistanceSq Q (P a) + (ε : ℝ≥0∞)) := by
      apply Finset.sum_le_sum
      intro a _
      gcongr
      exact (hcost a).le
    _ = weightedFrechetCostENNReal q P Q + (ε : ℝ≥0∞) := by
      have hq : ∑ a, ENNReal.ofReal (q a) = 1 := by
        rw [← ENNReal.ofReal_sum_of_nonneg]
        · simp [q.mass_one]
        · exact fun a _ ↦ q.nonneg a
      unfold weightedFrechetCostENNReal
      simp_rw [mul_add]
      rw [Finset.sum_add_distrib, ← Finset.sum_mul, hq, one_mul]

private lemma wassersteinDispersionSq_le_frechetENNReal
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (Q : WassersteinMeasure H) :
    wassersteinDispersionSq q P ≤ weightedFrechetCostENNReal q P Q := by
  apply ENNReal.le_of_forall_pos_le_add
  intro ε hε _
  obtain ⟨J, hJ⟩ := exists_joint_cost_le_frechet_add q P Q ε hε
  unfold wassersteinDispersionSq
  exact (sInf_le (s := {c : ℝ≥0∞ | ∃ J : JointCoupling P,
    c = weightedDispersionCost q J}) ⟨J, rfl⟩).trans hJ

/-- Extended-real unconditional Hilbert MMOT--barycenter equivalence.

The proof uses epsilon-optimal pairwise couplings and finite common-first-marginal gluing; it
does not assume an optimal joint plan or an attained barycenter.
-/
theorem ofReal_wassersteinBarycenterDispersionSq_eq_wassersteinDispersionSq_unconditional
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H) :
    ENNReal.ofReal (wassersteinBarycenterDispersionSq q P) =
      wassersteinDispersionSq q P := by
  apply le_antisymm
  · unfold wassersteinDispersionSq
    refine le_sInf ?_
    rintro c ⟨J, rfl⟩
    have hbdd : BddBelow (Set.range (weightedFrechetCost q P)) :=
      ⟨0, fun c ⟨Q, hQ⟩ ↦ hQ ▸ weightedFrechetCost_nonneg q P Q⟩
    have hcenter : wassersteinBarycenterDispersionSq q P ≤
        weightedFrechetCost q P (jointCenter q J) := by
      unfold wassersteinBarycenterDispersionSq
      exact csInf_le hbdd ⟨jointCenter q J, rfl⟩
    calc
      ENNReal.ofReal (wassersteinBarycenterDispersionSq q P) ≤
          ENNReal.ofReal (weightedFrechetCost q P (jointCenter q J)) :=
        ENNReal.ofReal_le_ofReal hcenter
      _ = weightedFrechetCostENNReal q P (jointCenter q J) :=
        (weightedFrechetCostENNReal_eq_ofReal q P (jointCenter q J)).symm
      _ ≤ weightedDispersionCost q J := weightedFrechetCostENNReal_jointCenter_le q J
  · have hQ (Q : WassersteinMeasure H) : wassersteinDispersionSq q P ≤
        ENNReal.ofReal (weightedFrechetCost q P Q) := by
      rw [← weightedFrechetCostENNReal_eq_ofReal]
      exact wassersteinDispersionSq_le_frechetENNReal q P Q
    let a := Classical.choice q.nonempty
    have hDtop : wassersteinDispersionSq q P ≠ ⊤ :=
      ne_top_of_le_ne_top ENNReal.ofReal_ne_top (hQ (P a))
    have hrange : (Set.range (weightedFrechetCost q P)).Nonempty :=
      ⟨weightedFrechetCost q P (P a), P a, rfl⟩
    have hreal : (wassersteinDispersionSq q P).toReal ≤
        wassersteinBarycenterDispersionSq q P := by
      unfold wassersteinBarycenterDispersionSq
      refine le_csInf hrange ?_
      rintro c ⟨Q, rfl⟩
      exact ENNReal.toReal_le_of_le_ofReal (weightedFrechetCost_nonneg q P Q) (hQ Q)
    rw [← ENNReal.ofReal_toReal hDtop]
    exact ENNReal.ofReal_le_ofReal hreal

/-- Real-valued unconditional Hilbert MMOT--barycenter equivalence.

No optimizer existence is used: equality holds at the level of infima.
-/
theorem wassersteinBarycenterDispersionSq_eq_toReal_wassersteinDispersionSq_unconditional
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H) :
    wassersteinBarycenterDispersionSq q P =
      (wassersteinDispersionSq q P).toReal := by
  have h := congrArg ENNReal.toReal
    (ofReal_wassersteinBarycenterDispersionSq_eq_wassersteinDispersionSq_unconditional q P)
  simpa [ENNReal.toReal_ofReal (wassersteinBarycenterDispersionSq_nonneg q P)] using h

/-- The square-root Frechet dispersion equals square-root MMOT dispersion unconditionally. -/
theorem wassersteinBarycenterDispersion_eq_wassersteinDispersion_unconditional
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H) :
    wassersteinBarycenterDispersion q P = wassersteinDispersion q P := by
  unfold wassersteinBarycenterDispersion wassersteinDispersion
  rw [wassersteinBarycenterDispersionSq_eq_toReal_wassersteinDispersionSq_unconditional q P,
    Real.sqrt_eq_rpow]

/-- Multi-marginal square-root dispersion is 1-Lipschitz in weighted product `W₂`.

This unconditional form assumes neither barycenter nor multi-marginal optimizer attainment.
-/
theorem wassersteinDispersion_stability_unconditional
    (q : ProbabilityWeight A) (P Q : A → WassersteinMeasure H) :
    |wassersteinDispersion q P - wassersteinDispersion q Q| ≤
      √(∑ a, q a * WassersteinDistance (P a) (Q a) ^ 2) := by
  rw [← wassersteinBarycenterDispersion_eq_wassersteinDispersion_unconditional q P,
    ← wassersteinBarycenterDispersion_eq_wassersteinDispersion_unconditional q Q]
  exact wassersteinBarycenterDispersion_stability q P Q

end WassersteinGeometry.Multimarginal
