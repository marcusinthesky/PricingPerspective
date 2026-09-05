import WassersteinGeometry.Contraction
import WassersteinGeometry.Multimarginal.Defs

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open MeasureTheory Finset

/-!
# Multi-marginal contraction and reverse transport

This module lifts the two-marginal measurable-inverse construction to finite
joint laws.  A coordinatewise pullback through a measurable embedding preserves
every prescribed marginal, and an antilipschitz map controls the full weighted
dispersion cost.

Optimal attainment remains certificate-first: the reverse infimum theorem
takes an `OptimalJointDispersion` for the image family rather than asserting a
new multi-marginal existence result.
-/

namespace WassersteinGeometry.Multimarginal

variable {A Ω Ω' : Type*} [Fintype A]
  [MeasurableSpace Ω] [PseudoMetricSpace Ω] [Inhabited Ω]
  [MeasurableSpace Ω'] [PseudoMetricSpace Ω'] [Inhabited Ω']

/-- Coordinatewise pushforward of a joint coupling preserves every prescribed
image marginal. -/
noncomputable def JointCoupling.pushforward
    {f : Ω → Ω'} (hm : Measurable f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (J : JointCoupling P) : JointCoupling P' where
  measure := J.measure.map (fun x a ↦ f (x a))
  marginal a := by
    have hF : Measurable fun x : A → Ω ↦ fun b ↦ f (x b) :=
      measurable_pi_lambda _ fun b ↦ hm.comp (measurable_pi_apply b)
    rw [Measure.map_map (measurable_pi_apply a) hF]
    change J.measure.map (f ∘ fun x : A → Ω ↦ x a) = (P' a).measure
    rw [← Measure.map_map hm (measurable_pi_apply a), J.marginal a, hP a]

omit [Fintype A] [Inhabited Ω] [MeasurableSpace Ω'] [Inhabited Ω'] in
/-- A coordinatewise `L`-Lipschitz map scales each coordinate-pair quadratic
cost by at most `L²`. -/
lemma lintegral_coord_edist_sq_pushforward_le
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f)
    (J : Measure (A → Ω)) (a b : A) :
    ∫⁻ x, (edist (f (x a)) (f (x b))) ^ 2 ∂J ≤
      (L : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J := by
  calc
    ∫⁻ x, (edist (f (x a)) (f (x b))) ^ 2 ∂J ≤
        ∫⁻ x, ((L : ℝ≥0∞) * edist (x a) (x b)) ^ 2 ∂J := by
      refine lintegral_mono fun x ↦ ?_
      exact pow_le_pow_left' (hf (x a) (x b)) 2
    _ = ∫⁻ x, (L : ℝ≥0∞) ^ 2 * (edist (x a) (x b)) ^ 2 ∂J := by
      simp_rw [mul_pow]
    _ = (L : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J :=
      lintegral_const_mul' _ _ (by simp)

/-- Coordinatewise Lipschitz pushforward scales a joint plan's weighted
quadratic dispersion cost by at most `L²`. -/
theorem weightedDispersionCost_pushforward_le
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hm : Measurable f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (J : JointCoupling P) :
    weightedDispersionCost q (J.pushforward hm hP) ≤
      (L : ℝ≥0∞) ^ 2 * weightedDispersionCost q J := by
  unfold weightedDispersionCost JointCoupling.pushforward
  have hF : Measurable fun x : A → Ω ↦ fun a ↦ f (x a) :=
    measurable_pi_lambda _ fun a ↦ hm.comp (measurable_pi_apply a)
  have hcost : ∀ a b, Measurable fun x : A → Ω' ↦ (edist (x a) (x b)) ^ 2 :=
    fun a b ↦ (measurable_edist.comp
      ((measurable_pi_apply a).prodMk (measurable_pi_apply b))).pow_const 2
  simp_rw [lintegral_map (hcost _ _) hF]
  calc
    ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
        ∫⁻ x, (edist (f (x a)) (f (x b))) ^ 2 ∂J.measure ≤
        ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          ((L : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) := by
      refine Finset.sum_le_sum fun a _ ↦ ?_
      refine Finset.sum_le_sum fun b _ ↦ ?_
      gcongr
      exact lintegral_coord_edist_sq_pushforward_le hf J.measure a b
    _ = (L : ℝ≥0∞) ^ 2 *
        ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure := by
      simp_rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      apply Finset.sum_congr rfl
      intro b _
      ac_rfl

/-- Multi-marginal quadratic dispersion contracts under a common Lipschitz
pushforward, conditional only on the repository's supplied optimal-plan
certificate. -/
theorem wassersteinDispersionSq_map_le_of_optimal
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hm : Measurable f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (cert : OptimalJointDispersion q P) :
    wassersteinDispersionSq q P' ≤
      (L : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P := by
  calc
    wassersteinDispersionSq q P' ≤
        weightedDispersionCost q (cert.plan.pushforward hm hP) := by
      unfold wassersteinDispersionSq
      exact sInf_le ⟨cert.plan.pushforward hm hP, rfl⟩
    _ ≤ (L : ℝ≥0∞) ^ 2 * weightedDispersionCost q cert.plan :=
      weightedDispersionCost_pushforward_le hf hm hP q cert.plan
    _ = (L : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P := by
      rw [cert.plan_cost, cert.value_eq_wassersteinDispersionSq]

/-- **Multi-marginal Lipschitz pushforward contraction.**

The result is stated directly at the two infima; no optimal joint plan is
required.  Independent joint couplings supply the only nonemptiness fact used
when the Lipschitz constant is zero.
-/
theorem wassersteinDispersionSq_map_le
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hm : Measurable f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) :
    wassersteinDispersionSq q P' ≤
      (L : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P := by
  unfold wassersteinDispersionSq
  set S : Set ℝ≥0∞ := {c | ∃ J : JointCoupling P, c = weightedDispersionCost q J}
  set S' : Set ℝ≥0∞ := {c | ∃ J : JointCoupling P', c = weightedDispersionCost q J}
  have key : ∀ c ∈ S, sInf S' ≤ (L : ℝ≥0∞) ^ 2 * c := by
    rintro c ⟨J, rfl⟩
    refine le_trans (sInf_le ?_) (weightedDispersionCost_pushforward_le hf hm hP q J)
    exact ⟨J.pushforward hm hP, rfl⟩
  have hmemS : weightedDispersionCost q (independentJointCoupling P) ∈ S :=
    ⟨independentJointCoupling P, rfl⟩
  rcases eq_or_ne L 0 with hL | hL
  · subst hL
    simpa using key _ hmemS
  · have h0 : ((L : ℝ≥0∞) ^ 2) ≠ 0 :=
      pow_ne_zero 2 (by exact_mod_cast hL)
    have htop : ((L : ℝ≥0∞) ^ 2) ≠ ⊤ := by
      simp [ENNReal.pow_eq_top_iff]
    calc
      sInf S' ≤ ⨅ c : S, (L : ℝ≥0∞) ^ 2 * (c : ℝ≥0∞) :=
        le_iInf fun c ↦ key c c.2
      _ = (L : ℝ≥0∞) ^ 2 * ⨅ c : S, (c : ℝ≥0∞) :=
        (ENNReal.mul_iInf_of_ne h0 htop).symm
      _ = (L : ℝ≥0∞) ^ 2 * sInf S := by rw [← sInf_eq_iInf']

/-- A finite squared-dispersion comparison by `K²` implies the corresponding
real square-root comparison by `K`. -/
lemma wassersteinDispersion_le_of_sq_le_mul_sq
    {K : ℝ≥0} {P : A → WassersteinMeasure Ω}
    {Q : A → WassersteinMeasure Ω'}
    {q : ProbabilityWeight A}
    (hfin : wassersteinDispersionSq q Q ≠ ⊤)
    (hsq : wassersteinDispersionSq q P ≤
      (K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q Q) :
    wassersteinDispersion q P ≤ (K : ℝ) * wassersteinDispersion q Q := by
  have hprod : ((K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q Q) ≠ ⊤ := by
    simp [ENNReal.mul_eq_top, hfin, ENNReal.pow_eq_top_iff]
  have hreal : (wassersteinDispersionSq q P).toReal ≤
      (K : ℝ) ^ 2 * (wassersteinDispersionSq q Q).toReal := by
    have := ENNReal.toReal_mono hprod hsq
    rwa [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.coe_toReal] at this
  have hKnn : (0 : ℝ) ≤ (K : ℝ) := K.coe_nonneg
  unfold wassersteinDispersion
  calc
    (wassersteinDispersionSq q P).toReal ^ (1 / 2 : ℝ) ≤
        ((K : ℝ) ^ 2 * (wassersteinDispersionSq q Q).toReal) ^ (1 / 2 : ℝ) :=
      Real.rpow_le_rpow ENNReal.toReal_nonneg hreal (by norm_num)
    _ = ((K : ℝ) ^ 2) ^ (1 / 2 : ℝ) *
        (wassersteinDispersionSq q Q).toReal ^ (1 / 2 : ℝ) :=
      Real.mul_rpow (by positivity) ENNReal.toReal_nonneg
    _ = (K : ℝ) *
        (wassersteinDispersionSq q Q).toReal ^ (1 / 2 : ℝ) := by
      congr 1
      rw [← Real.rpow_natCast (K : ℝ) 2, ← Real.rpow_mul hKnn]
      norm_num

/-- Real multi-marginal dispersion contracts under a common Lipschitz map when
the source optimum is certified finite. -/
theorem wassersteinDispersion_map_le
    [OpensMeasurableSpace Ω'] [SecondCountableTopology Ω']
    {L : ℝ≥0} {f : Ω → Ω'} (hf : LipschitzWith L f) (hm : Measurable f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (hfin : wassersteinDispersionSq q P ≠ ⊤) :
    wassersteinDispersion q P' ≤ (L : ℝ) * wassersteinDispersion q P :=
  wassersteinDispersion_le_of_sq_le_mul_sq hfin
    (wassersteinDispersionSq_map_le hf hm hP q)

/-- Coordinatewise pullback through a measurable embedding recovers a joint
coupling of the original marginal family. -/
noncomputable def JointCoupling.pullback
    {f : Ω → Ω'} (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (J : JointCoupling P') : JointCoupling P where
  measure := J.measure.map (fun x a ↦ hme.invFun (x a))
  marginal a := by
    have hg : Measurable hme.invFun := hme.measurable_invFun
    have hG : Measurable fun x : A → Ω' ↦ fun b ↦ hme.invFun (x b) :=
      measurable_pi_lambda _ fun b ↦ hg.comp (measurable_pi_apply b)
    rw [Measure.map_map (measurable_pi_apply a) hG]
    change J.measure.map (hme.invFun ∘ fun x : A → Ω' ↦ x a) = (P a).measure
    rw [← Measure.map_map hg (measurable_pi_apply a), J.marginal a, hP a,
      Measure.map_map hg hme.measurable]
    simp [Function.LeftInverse.id hme.leftInverse_invFun]

omit [Fintype A] in
/-- Pulling an image joint law through a measurable `K`-antilipschitz embedding
multiplies each coordinate-pair quadratic cost by at most `K²`. -/
lemma lintegral_coord_edist_sq_pullback_le
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (J : JointCoupling P') (a b : A) :
    ∫⁻ x, (edist (hme.invFun (x a)) (hme.invFun (x b))) ^ 2 ∂J.measure ≤
      (K : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure := by
  let g : Ω' → Ω := hme.invFun
  have ha : ∀ᵐ x ∂J.measure, x a ∈ Set.range f := by
    apply ae_of_ae_map (measurable_pi_apply a).aemeasurable
    rw [J.marginal a, hP a]
    exact ae_map_mem_range f hme.measurableSet_range (P a).measure
  have hb : ∀ᵐ x ∂J.measure, x b ∈ Set.range f := by
    apply ae_of_ae_map (measurable_pi_apply b).aemeasurable
    rw [J.marginal b, hP b]
    exact ae_map_mem_range f hme.measurableSet_range (P b).measure
  calc
    ∫⁻ x, (edist (g (x a)) (g (x b))) ^ 2 ∂J.measure ≤
        ∫⁻ x, ((K : ℝ≥0∞) * edist (x a) (x b)) ^ 2 ∂J.measure := by
      refine lintegral_mono_ae ?_
      filter_upwards [ha, hb] with x hxa hxb
      rcases hxa with ⟨y, hy⟩
      rcases hxb with ⟨z, hz⟩
      rw [← hy, ← hz]
      change (edist (hme.invFun (f y)) (hme.invFun (f z))) ^ 2 ≤ _
      rw [hme.leftInverse_invFun y, hme.leftInverse_invFun z]
      exact pow_le_pow_left' (hf y z) 2
    _ = ∫⁻ x, (K : ℝ≥0∞) ^ 2 * (edist (x a) (x b)) ^ 2 ∂J.measure := by
      simp_rw [mul_pow]
    _ = (K : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure :=
      lintegral_const_mul' _ _ (by simp)

/-- The coordinatewise inverse image of a joint law has no more than `K²`
times its weighted quadratic dispersion cost. -/
theorem weightedDispersionCost_pullback_le
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (J : JointCoupling P') :
    weightedDispersionCost q (J.pullback hme hP) ≤
      (K : ℝ≥0∞) ^ 2 * weightedDispersionCost q J := by
  unfold weightedDispersionCost JointCoupling.pullback
  have hg : Measurable hme.invFun := hme.measurable_invFun
  have hG : Measurable fun x : A → Ω' ↦ fun a ↦ hme.invFun (x a) :=
    measurable_pi_lambda _ fun a ↦ hg.comp (measurable_pi_apply a)
  have hcost : ∀ a b, Measurable fun x : A → Ω ↦ (edist (x a) (x b)) ^ 2 :=
    fun a b ↦ (measurable_edist.comp
      ((measurable_pi_apply a).prodMk (measurable_pi_apply b))).pow_const 2
  simp_rw [lintegral_map (hcost _ _) hG]
  calc
    ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
        ∫⁻ x, (edist (hme.invFun (x a)) (hme.invFun (x b))) ^ 2 ∂J.measure ≤
        ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          ((K : ℝ≥0∞) ^ 2 * ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) := by
      refine Finset.sum_le_sum fun a _ ↦ ?_
      refine Finset.sum_le_sum fun b _ ↦ ?_
      gcongr
      exact lintegral_coord_edist_sq_pullback_le hf hme hP J a b
    _ = (K : ℝ≥0∞) ^ 2 *
        ∑ a, ∑ b, ENNReal.ofReal (q a * q b / 2) *
          ∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure := by
      simp_rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      apply Finset.sum_congr rfl
      intro b _
      ac_rfl

/-- **Reverse multi-marginal transport under an antilipschitz measurable
embedding.**

Given an optimal image joint plan, coordinatewise measurable inversion proves
`Dq²(P) ≤ K² Dq²(f#P)` without an additional measure-level premise.
-/
theorem wassersteinDispersionSq_le_antilipschitz_map_of_optimal
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (cert : OptimalJointDispersion q P') :
    wassersteinDispersionSq q P ≤
      (K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P' := by
  calc
    wassersteinDispersionSq q P ≤
        weightedDispersionCost q (cert.plan.pullback hme hP) := by
      unfold wassersteinDispersionSq
      exact sInf_le ⟨cert.plan.pullback hme hP, rfl⟩
    _ ≤ (K : ℝ≥0∞) ^ 2 * weightedDispersionCost q cert.plan :=
      weightedDispersionCost_pullback_le hf hme hP q cert.plan
    _ = (K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P' := by
      rw [cert.plan_cost, cert.value_eq_wassersteinDispersionSq]

/-- **Reverse multi-marginal transport under an antilipschitz measurable
embedding.**

Coordinatewise measurable inversion is passed directly through both infima,
so no optimal joint-plan certificate is required.
-/
theorem wassersteinDispersionSq_le_antilipschitz_map
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) :
    wassersteinDispersionSq q P ≤
      (K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P' := by
  unfold wassersteinDispersionSq
  set S : Set ℝ≥0∞ := {c | ∃ J : JointCoupling P, c = weightedDispersionCost q J}
  set S' : Set ℝ≥0∞ := {c | ∃ J : JointCoupling P', c = weightedDispersionCost q J}
  have key : ∀ c ∈ S', sInf S ≤ (K : ℝ≥0∞) ^ 2 * c := by
    rintro c ⟨J, rfl⟩
    refine le_trans (sInf_le ?_) (weightedDispersionCost_pullback_le hf hme hP q J)
    exact ⟨J.pullback hme hP, rfl⟩
  have hmemS' : weightedDispersionCost q (independentJointCoupling P') ∈ S' :=
    ⟨independentJointCoupling P', rfl⟩
  rcases eq_or_ne K 0 with hK | hK
  · subst hK
    simpa using key _ hmemS'
  · have h0 : ((K : ℝ≥0∞) ^ 2) ≠ 0 :=
      pow_ne_zero 2 (by exact_mod_cast hK)
    have htop : ((K : ℝ≥0∞) ^ 2) ≠ ⊤ := by
      simp [ENNReal.pow_eq_top_iff]
    calc
      sInf S ≤ ⨅ c : S', (K : ℝ≥0∞) ^ 2 * (c : ℝ≥0∞) :=
        le_iInf fun c ↦ key c c.2
      _ = (K : ℝ≥0∞) ^ 2 * ⨅ c : S', (c : ℝ≥0∞) :=
        (ENNReal.mul_iInf_of_ne h0 htop).symm
      _ = (K : ℝ≥0∞) ^ 2 * sInf S' := by rw [← sInf_eq_iInf']

/-- Real reverse multi-marginal transport under an antilipschitz measurable
embedding, conditional on a finite certified image optimum. -/
theorem wassersteinDispersion_le_antilipschitz_map
    [OpensMeasurableSpace Ω] [SecondCountableTopology Ω]
    {K : ℝ≥0} {f : Ω → Ω'} (hf : AntilipschitzWith K f)
    (hme : MeasurableEmbedding f)
    {P : A → WassersteinMeasure Ω} {P' : A → WassersteinMeasure Ω'}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (hfin : wassersteinDispersionSq q P' ≠ ⊤) :
    wassersteinDispersion q P ≤ (K : ℝ) * wassersteinDispersion q P' :=
  wassersteinDispersion_le_of_sq_le_mul_sq hfin
    (wassersteinDispersionSq_le_antilipschitz_map hf hme hP q)

section StandardBorelSource

variable {S T : Type*}
  [MeasurableSpace S] [MetricSpace S] [Inhabited S]
  [OpensMeasurableSpace S] [SecondCountableTopology S] [StandardBorelSpace S]
  [MeasurableSpace T] [PseudoMetricSpace T] [Inhabited T]
  [MeasurableSpace.CountablySeparated T]

/-- On a standard Borel metric source, measurability and the injectivity from
`AntilipschitzWith` automatically supply the coordinatewise measurable inverse. -/
theorem wassersteinDispersionSq_le_antilipschitz_map_of_measurable
    {K : ℝ≥0} {f : S → T} (hf : AntilipschitzWith K f) (hm : Measurable f)
    {P : A → WassersteinMeasure S} {P' : A → WassersteinMeasure T}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) :
    wassersteinDispersionSq q P ≤
      (K : ℝ≥0∞) ^ 2 * wassersteinDispersionSq q P' :=
  wassersteinDispersionSq_le_antilipschitz_map hf
    (hm.measurableEmbedding hf.injective) hP q

/-- Real standard-Borel reverse dispersion bound derived from a measurable
pointwise antilipschitz carrier. -/
theorem wassersteinDispersion_le_antilipschitz_map_of_measurable
    {K : ℝ≥0} {f : S → T} (hf : AntilipschitzWith K f) (hm : Measurable f)
    {P : A → WassersteinMeasure S} {P' : A → WassersteinMeasure T}
    (hP : ∀ a, (P' a).measure = (P a).measure.map f)
    (q : ProbabilityWeight A) (hfin : wassersteinDispersionSq q P' ≠ ⊤) :
    wassersteinDispersion q P ≤ (K : ℝ) * wassersteinDispersion q P' :=
  wassersteinDispersion_le_antilipschitz_map hf
    (hm.measurableEmbedding hf.injective) hP q hfin

end StandardBorelSource

end WassersteinGeometry.Multimarginal
