import PricingPerspective.Transmission.Dispersion.PortfolioVariance
import WassersteinGeometry.Multimarginal.TwoMarginal

set_option linter.style.longLine false

open scoped ENNReal MeasureTheory InnerProductSpace
open MeasureTheory Finset
open WassersteinGeometry WassersteinGeometry.Hilbert
open WassersteinGeometry.Multimarginal

/-!
# Exact optimal systematic-variance envelope

Weighted Hilbert polarization turns minimization of coherent multi-marginal dispersion into
maximization of systematic portfolio variance.  Attainment is certificate-first: an
`OptimalJointDispersion` supplies the minimizing joint plan, without asserting a general
optimal-plan existence theorem.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A H : Type*} [Fintype A]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-- The realized real-valued pairwise dispersion of one coherent Hilbert-valued joint law. -/
noncomputable def realizedJointDispersion
    {P : A → WassersteinMeasure H} (q : ProbabilityWeight A) (J : JointCoupling P) : ℝ :=
  ∑ a, ∑ b, (q a * q b / 2) * ∫ x, ‖x a - x b‖ ^ 2 ∂J.measure

omit [Fintype A] [BorelSpace H] [SecondCountableTopology H] in
/-- Pairwise inner-product integrability implies coordinate-square integrability. -/
lemma integrable_coordinate_normSq
    {P : A → WassersteinMeasure H} (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure)
    (a : A) : Integrable (fun x : A → H ↦ ‖x a‖ ^ 2) J.measure :=
  (hint a a).congr (Filter.Eventually.of_forall fun x ↦ real_inner_self_eq_norm_sq (x a))

omit [Fintype A] [BorelSpace H] [SecondCountableTopology H] in
/-- Pairwise inner-product integrability implies integrability of every squared distance. -/
lemma integrable_coordinate_distSq
    {P : A → WassersteinMeasure H} (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure)
    (a b : A) : Integrable (fun x : A → H ↦ ‖x a - x b‖ ^ 2) J.measure := by
  have hleft := integrable_coordinate_normSq J hint a
  have hright := integrable_coordinate_normSq J hint b
  have hcross := (hint a b).const_mul 2
  have hrw : (fun x : A → H ↦ ‖x a - x b‖ ^ 2) =
      fun x ↦ ‖x a‖ ^ 2 - 2 * ⟪x a, x b⟫_ℝ + ‖x b‖ ^ 2 := by
    funext x
    exact norm_sub_sq_real (x a) (x b)
  rw [hrw]
  exact (hleft.sub hcross).add hright

omit [Fintype A] [BorelSpace H] [SecondCountableTopology H] in
/-- A coordinate-pair lower integral is the extended-real image of its real integral. -/
theorem lintegral_coordinate_edist_sq_eq_ofReal_integral
    {P : A → WassersteinMeasure H} (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure)
    (a b : A) :
    (∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) =
      ENNReal.ofReal (∫ x, ‖x a - x b‖ ^ 2 ∂J.measure) := by
  have hpt : ∀ x : A → H,
      (edist (x a) (x b)) ^ 2 = ENNReal.ofReal (‖x a - x b‖ ^ 2) := by
    intro x
    rw [edist_dist, dist_eq_norm, ← ENNReal.ofReal_pow (norm_nonneg _)]
  calc
    (∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) =
        ∫⁻ x, ENNReal.ofReal (‖x a - x b‖ ^ 2) ∂J.measure := by
      exact lintegral_congr hpt
    _ = ENNReal.ofReal (∫ x, ‖x a - x b‖ ^ 2 ∂J.measure) :=
      (ofReal_integral_eq_lintegral_ofReal (integrable_coordinate_distSq J hint a b)
        (Filter.Eventually.of_forall fun x ↦ by positivity)).symm

omit [BorelSpace H] [SecondCountableTopology H] in
/-- Integrability makes the extended-real cost of a coherent joint plan finite. -/
theorem weightedDispersionCost_ne_top
    {P : A → WassersteinMeasure H} (q : ProbabilityWeight A) (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    weightedDispersionCost q J ≠ ⊤ := by
  unfold weightedDispersionCost
  exact ENNReal.sum_ne_top.mpr fun a _ ↦ ENNReal.sum_ne_top.mpr fun b _ ↦
    ENNReal.mul_ne_top (by simp) (by
      rw [lintegral_coordinate_edist_sq_eq_ofReal_integral J hint]
      simp)

omit [BorelSpace H] [SecondCountableTopology H] in
/-- The extended-real multi-marginal cost converts to the realized real dispersion. -/
theorem weightedDispersionCost_toReal_eq_realizedJointDispersion
    {P : A → WassersteinMeasure H} (q : ProbabilityWeight A) (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    (weightedDispersionCost q J).toReal = realizedJointDispersion q J := by
  have hlin (a b : A) := lintegral_coordinate_edist_sq_eq_ofReal_integral J hint a b
  have hlin_top (a b : A) :
      (∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) ≠ ⊤ := by
    rw [hlin]
    simp
  have hterm_top (a b : A) :
      ENNReal.ofReal (q a * q b / 2) *
          (∫⁻ x, (edist (x a) (x b)) ^ 2 ∂J.measure) ≠ ⊤ :=
    ENNReal.mul_ne_top (by simp) (hlin_top a b)
  unfold weightedDispersionCost realizedJointDispersion
  rw [ENNReal.toReal_sum]
  · refine Finset.sum_congr rfl fun a _ ↦ ?_
    rw [ENNReal.toReal_sum]
    · refine Finset.sum_congr rfl fun b _ ↦ ?_
      have hcoef : 0 ≤ q a * q b / 2 :=
        div_nonneg (mul_nonneg (q.nonneg a) (q.nonneg b)) (by norm_num)
      have hint_nonneg : 0 ≤ ∫ x, ‖x a - x b‖ ^ 2 ∂J.measure :=
        integral_nonneg fun x ↦ by positivity
      rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal hcoef, hlin,
        ENNReal.toReal_ofReal hint_nonneg]
    · exact fun b _ ↦ hterm_top a b
  · intro a _
    exact ENNReal.sum_ne_top.mpr fun b _ ↦ hterm_top a b

omit [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] in
/-- Realized joint dispersion is nonnegative. -/
theorem realizedJointDispersion_nonneg
    {P : A → WassersteinMeasure H} (q : ProbabilityWeight A) (J : JointCoupling P) :
    0 ≤ realizedJointDispersion q J := by
  unfold realizedJointDispersion
  exact Finset.sum_nonneg fun a _ ↦ Finset.sum_nonneg fun b _ ↦
    mul_nonneg (div_nonneg (mul_nonneg (q.nonneg a) (q.nonneg b)) (by norm_num))
      (integral_nonneg fun x ↦ by positivity)

omit [SecondCountableTopology H] in
/-- Weighted polarization integrated under one coherent joint law. -/
theorem jointPortfolioVariance_eq_secondMoments_sub_realizedDispersion
    {P : A → WassersteinMeasure H} (q : ProbabilityWeight A) (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    PricingPerspective.Transmission.portfolioVariance J.measure q =
      (∑ a, q a * secondMoment (P a)) - realizedJointDispersion q J := by
  have hnorm (a : A) := integrable_coordinate_normSq J hint a
  have hpair (a b : A) := integrable_coordinate_distSq J hint a b
  have hmarg (a : A) : secondMoment (P a) = ∫ x, ‖x a‖ ^ 2 ∂J.measure := by
    unfold secondMoment
    rw [← J.marginal a, integral_map (measurable_pi_apply a).aemeasurable]
    exact aestronglyMeasurable_normSq _
  have hfirst : Integrable (fun x : A → H ↦ ∑ a, q a * ‖x a‖ ^ 2) J.measure :=
    integrable_finsetSum _ fun a _ ↦ (hnorm a).const_mul (q a)
  have hdouble : Integrable
      (fun x : A → H ↦ ∑ a, ∑ b, q a * q b * ‖x a - x b‖ ^ 2) J.measure :=
    integrable_finsetSum _ fun a _ ↦ integrable_finsetSum _ fun b _ ↦
      (hpair a b).const_mul (q a * q b)
  have hdisp : Integrable
      (fun x : A → H ↦
        (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * ‖x a - x b‖ ^ 2) J.measure :=
    hdouble.const_mul (1 / 2)
  unfold PricingPerspective.Transmission.portfolioVariance
  calc
    (∫ x, ‖PricingPerspective.Transmission.portfolioLoading q x‖ ^ 2 ∂J.measure) =
        ∫ x, ((∑ a, q a * ‖x a‖ ^ 2) -
          (1 / 2 : ℝ) * ∑ a, ∑ b, q a * q b * ‖x a - x b‖ ^ 2) ∂J.measure := by
      refine integral_congr_ae (Filter.Eventually.of_forall fun x ↦ ?_)
      exact weighted_polarization q x
    _ = (∫ x, ∑ a, q a * ‖x a‖ ^ 2 ∂J.measure) -
        ∫ x, (1 / 2 : ℝ) *
          ∑ a, ∑ b, q a * q b * ‖x a - x b‖ ^ 2 ∂J.measure := by
      rw [integral_sub hfirst hdisp]
    _ = (∑ a, q a * secondMoment (P a)) - realizedJointDispersion q J := by
      rw [integral_finsetSum _ (fun a _ ↦ (hnorm a).const_mul (q a))]
      simp_rw [integral_const_mul, ← hmarg]
      rw [integral_finsetSum _ (fun a _ ↦ integrable_finsetSum _ fun b _ ↦
          (hpair a b).const_mul (q a * q b))]
      simp_rw [integral_finsetSum _ (fun b _ ↦ (hpair _ b).const_mul _),
        integral_const_mul]
      unfold realizedJointDispersion
      congr 1
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun a _ ↦ ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun b _ ↦ ?_
      ring

/-- The systematic portfolio variance values attained by integrable coherent joint laws. -/
def attainableJointPortfolioVariances
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H) : Set ℝ :=
  {v | ∃ J : JointCoupling P,
    (∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) ∧
      v = PricingPerspective.Transmission.portfolioVariance J.measure q}

omit [SecondCountableTopology H] in
/-- A dispersion-minimizing certificate attains the exact systematic-variance envelope. -/
theorem optimalJointPortfolioVariance_eq_envelope
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (cert : OptimalJointDispersion q P)
    (hint : ∀ a b, Integrable
      (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) cert.plan.measure) :
    PricingPerspective.Transmission.portfolioVariance cert.plan.measure q =
      (∑ a, q a * secondMoment (P a)) - cert.value.toReal := by
  rw [jointPortfolioVariance_eq_secondMoments_sub_realizedDispersion q cert.plan hint]
  rw [← weightedDispersionCost_toReal_eq_realizedJointDispersion q cert.plan hint,
    cert.plan_cost]

omit [SecondCountableTopology H] in
/-- The attaining variance written directly with the multi-marginal dispersion infimum. -/
theorem optimalJointPortfolioVariance_eq_wassersteinDispersionEnvelope
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (cert : OptimalJointDispersion q P)
    (hint : ∀ a b, Integrable
      (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) cert.plan.measure) :
    PricingPerspective.Transmission.portfolioVariance cert.plan.measure q =
      (∑ a, q a * secondMoment (P a)) - (wassersteinDispersionSq q P).toReal := by
  rw [optimalJointPortfolioVariance_eq_envelope q P cert hint,
    cert.value_eq_wassersteinDispersionSq]

omit [SecondCountableTopology H] in
/-- No coherent joint law has larger systematic variance than a certified dispersion minimizer. -/
theorem jointPortfolioVariance_le_optimalEnvelope
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (cert : OptimalJointDispersion q P) (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    PricingPerspective.Transmission.portfolioVariance J.measure q ≤
      (∑ a, q a * secondMoment (P a)) - cert.value.toReal := by
  have hcost : cert.value.toReal ≤ (weightedDispersionCost q J).toReal :=
    ENNReal.toReal_mono (weightedDispersionCost_ne_top q J hint) (cert.minimal J)
  rw [jointPortfolioVariance_eq_secondMoments_sub_realizedDispersion q J hint,
    ← weightedDispersionCost_toReal_eq_realizedJointDispersion q J hint]
  linarith

omit [SecondCountableTopology H] in
/-- The certified envelope is the greatest attainable systematic portfolio variance. -/
theorem isGreatest_attainableJointPortfolioVariances
    (q : ProbabilityWeight A) (P : A → WassersteinMeasure H)
    (cert : OptimalJointDispersion q P)
    (hint : ∀ a b, Integrable
      (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) cert.plan.measure) :
    IsGreatest (attainableJointPortfolioVariances q P)
      ((∑ a, q a * secondMoment (P a)) - (wassersteinDispersionSq q P).toReal) := by
  constructor
  · exact ⟨cert.plan, hint,
      (optimalJointPortfolioVariance_eq_wassersteinDispersionEnvelope q P cert hint).symm⟩
  · rintro v ⟨J, hJ, rfl⟩
    simpa [cert.value_eq_wassersteinDispersionSq] using
      jointPortfolioVariance_le_optimalEnvelope q P cert J hJ

/-- **Two-asset Paper 1 bridge.** The optimal coherent two-asset portfolio variance subtracts
`q₀q₁ W₂²(P₀,P₁)`, exactly the two-marginal covariance-envelope correction. -/
theorem optimalTwoAssetPortfolioVariance_eq_paperOne
    (q : ProbabilityWeight (Fin 2)) (P : Fin 2 → WassersteinMeasure H)
    (cert : OptimalJointDispersion q P)
    (hint : ∀ a b, Integrable
      (fun x : Fin 2 → H ↦ ⟪x a, x b⟫_ℝ) cert.plan.measure) :
    PricingPerspective.Transmission.portfolioVariance cert.plan.measure q =
      q 0 * secondMoment (P 0) + q 1 * secondMoment (P 1) -
        q 0 * q 1 * (WassersteinDistanceSq (P 0) (P 1)).toReal := by
  rw [optimalJointPortfolioVariance_eq_wassersteinDispersionEnvelope q P cert hint,
    Fin.sum_univ_two, wassersteinDispersionSq_finTwo, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (mul_nonneg (q.nonneg 0) (q.nonneg 1))]

/-- Equal weights on the two-element asset index. -/
noncomputable def equalFinTwoWeight : ProbabilityWeight (Fin 2) where
  weight _ := 1 / 2
  nonneg _ := by norm_num
  mass_one := by simp

/-- The equal-weight bridge gives the familiar one-quarter Wasserstein correction. -/
theorem optimalEqualWeightTwoAssetVariance_eq_paperOne
    (P : Fin 2 → WassersteinMeasure H)
    (cert : OptimalJointDispersion equalFinTwoWeight P)
    (hint : ∀ a b, Integrable
      (fun x : Fin 2 → H ↦ ⟪x a, x b⟫_ℝ) cert.plan.measure) :
    PricingPerspective.Transmission.portfolioVariance cert.plan.measure equalFinTwoWeight =
      (secondMoment (P 0) + secondMoment (P 1)) / 2 -
        (1 / 4 : ℝ) * (WassersteinDistanceSq (P 0) (P 1)).toReal := by
  rw [optimalTwoAssetPortfolioVariance_eq_paperOne equalFinTwoWeight P cert hint]
  simp only [equalFinTwoWeight]
  ring

end PricingPerspective.Transmission.Dispersion
