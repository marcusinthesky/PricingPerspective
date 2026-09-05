import PricingPerspective.Transmission.Slack
import WassersteinGeometry.Multimarginal.Barycenter
import WassersteinGeometry.Multimarginal.Contraction

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open MeasureTheory Finset
open WassersteinGeometry WassersteinGeometry.Multimarginal

/-!
# Aggregate carrier dispersion with heterogeneous slack

This module combines three independently reusable facts: coordinatewise inversion of an
antilipschitz carrier, the Hilbert MMOT--barycenter bridge, and stability of the barycenter
radius under marginal Wasserstein perturbations.  The result is the aggregate lower bound

`D(P) >= K⁻¹ D(C) - sqrt (sum a, q a * tau a ^ 2)`.

The truncated version below records the nonnegativity of dispersion explicitly.  A second
theorem derives each marginal perturbation bound from a synchronous integrated quadratic
slack bound, so no pointwise slack assumption is needed.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A X H : Type*} [Fintype A]
  [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]
  [MeasurableSpace H] [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [CompleteSpace H] [SecondCountableTopology H] [BorelSpace H] [Inhabited H]

/-- The weighted root-mean-square radius of heterogeneous marginal slack bounds. -/
noncomputable def weightedSlackRadius
    (q : ProbabilityWeight A) (tau : A → ℝ≥0) : ℝ :=
  √(∑ a, q a * (tau a : ℝ) ^ 2)

/-- The economically meaningful nonnegative aggregate carrier lower bound. -/
noncomputable def aggregateCarrierFloor
    (K : ℝ≥0) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (tau : A → ℝ≥0) : ℝ :=
  max 0 ((K : ℝ)⁻¹ * wassersteinDispersion q C - weightedSlackRadius q tau)

omit [InnerProductSpace ℝ H] [BorelSpace H] in
/-- Coordinatewise Wasserstein slack is bounded by its supplied weighted RMS radius. -/
lemma weightedWassersteinRadius_le_weightedSlackRadius
    (q : ProbabilityWeight A) (P T : A → WassersteinMeasure H)
    (tau : A → ℝ≥0) (hslack : ∀ a, WassersteinDistance (P a) (T a) ≤ tau a) :
    √(∑ a, q a * WassersteinDistance (P a) (T a) ^ 2) ≤
      weightedSlackRadius q tau := by
  unfold weightedSlackRadius
  apply Real.sqrt_le_sqrt
  refine Finset.sum_le_sum fun a _ ↦ ?_
  apply mul_le_mul_of_nonneg_left _ (q.nonneg a)
  nlinarith [hslack a, wasserstein_nonneg (P a) (T a),
    show 0 ≤ (tau a : ℝ) by positivity]

/-- **Aggregate carrier floor with heterogeneous marginal slack.**

If the common carrier map is `K`-antilipschitz, the carrier tuple's dispersion cannot exceed
`K` times the transmitted tuple's dispersion.  Barycenter stability then allows each
observed exposure marginal to deviate from its transmitted law by `tau a`.  The resulting
lower bound is truncated at zero, which is always valid because dispersion is nonnegative.
-/
theorem aggregateCarrierFloor_le_of_antilipschitz
    {K : ℝ≥0} (hK : K ≠ 0) {t : X → H} (ht : AntilipschitzWith K t)
    (hme : MeasurableEmbedding t) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (T P : A → WassersteinMeasure H)
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (bridgeT : HilbertBarycenterBridge q T) (bridgeP : HilbertBarycenterBridge q P)
    (tau : A → ℝ≥0) (hslack : ∀ a, WassersteinDistance (P a) (T a) ≤ tau a) :
    aggregateCarrierFloor K q C tau ≤ wassersteinDispersion q P := by
  have hcarrier : wassersteinDispersion q C ≤
      (K : ℝ) * wassersteinDispersion q T :=
    wassersteinDispersion_le_antilipschitz_map ht hme hT q
      (wassersteinDispersionSq_ne_top_of_barycenterBridge q T bridgeT)
  have hstability := wassersteinDispersion_stability q P T bridgeP bridgeT
  have hradius := weightedWassersteinRadius_le_weightedSlackRadius q P T tau hslack
  have hTP : wassersteinDispersion q T ≤
      wassersteinDispersion q P + weightedSlackRadius q tau := by
    have hstability' : |wassersteinDispersion q T - wassersteinDispersion q P| ≤
        √(∑ a, q a * WassersteinDistance (P a) (T a) ^ 2) := by
      rw [abs_sub_comm]
      exact hstability
    have habs : wassersteinDispersion q T - wassersteinDispersion q P ≤
        √(∑ a, q a * WassersteinDistance (P a) (T a) ^ 2) :=
      (le_abs_self _).trans hstability'
    linarith
  have hKpos : 0 < (K : ℝ) := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hK)
  have hraw : (K : ℝ)⁻¹ * wassersteinDispersion q C -
      weightedSlackRadius q tau ≤ wassersteinDispersion q P := by
    have hscaled : (K : ℝ)⁻¹ * wassersteinDispersion q C ≤
        wassersteinDispersion q T := by
      calc
        (K : ℝ)⁻¹ * wassersteinDispersion q C ≤
            (K : ℝ)⁻¹ * ((K : ℝ) * wassersteinDispersion q T) :=
          mul_le_mul_of_nonneg_left hcarrier (inv_nonneg.mpr hKpos.le)
        _ = wassersteinDispersion q T := by field_simp
    linarith
  unfold aggregateCarrierFloor
  exact max_le (wassersteinDispersion_nonneg q P) hraw

/-- **Unconditional aggregate carrier floor with heterogeneous marginal slack.**

This bridge-free form obtains MMOT finiteness and stability from the unconditional Hilbert
MMOT--barycenter equivalence.  It does not assume that either infimum is attained.
-/
theorem aggregateCarrierFloor_le_of_antilipschitz_unconditional
    {K : ℝ≥0} (hK : K ≠ 0) {t : X → H} (ht : AntilipschitzWith K t)
    (hme : MeasurableEmbedding t) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (T P : A → WassersteinMeasure H)
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (tau : A → ℝ≥0) (hslack : ∀ a, WassersteinDistance (P a) (T a) ≤ tau a) :
    aggregateCarrierFloor K q C tau ≤ wassersteinDispersion q P := by
  have hTfinite : wassersteinDispersionSq q T ≠ ⊤ := by
    rw [← ofReal_wassersteinBarycenterDispersionSq_eq_wassersteinDispersionSq_unconditional]
    exact ENNReal.ofReal_ne_top
  have hcarrier : wassersteinDispersion q C ≤
      (K : ℝ) * wassersteinDispersion q T :=
    wassersteinDispersion_le_antilipschitz_map ht hme hT q hTfinite
  have hstability := wassersteinDispersion_stability_unconditional q P T
  have hradius := weightedWassersteinRadius_le_weightedSlackRadius q P T tau hslack
  have hTP : wassersteinDispersion q T ≤
      wassersteinDispersion q P + weightedSlackRadius q tau := by
    have hstability' : |wassersteinDispersion q T - wassersteinDispersion q P| ≤
        √(∑ a, q a * WassersteinDistance (P a) (T a) ^ 2) := by
      rw [abs_sub_comm]
      exact hstability
    have habs : wassersteinDispersion q T - wassersteinDispersion q P ≤
        √(∑ a, q a * WassersteinDistance (P a) (T a) ^ 2) :=
      (le_abs_self _).trans hstability'
    linarith
  have hKpos : 0 < (K : ℝ) := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr hK)
  have hraw : (K : ℝ)⁻¹ * wassersteinDispersion q C -
      weightedSlackRadius q tau ≤ wassersteinDispersion q P := by
    have hscaled : (K : ℝ)⁻¹ * wassersteinDispersion q C ≤
        wassersteinDispersion q T := by
      calc
        (K : ℝ)⁻¹ * wassersteinDispersion q C ≤
            (K : ℝ)⁻¹ * ((K : ℝ) * wassersteinDispersion q T) :=
          mul_le_mul_of_nonneg_left hcarrier (inv_nonneg.mpr hKpos.le)
        _ = wassersteinDispersion q T := by field_simp
    linarith
  unfold aggregateCarrierFloor
  exact max_le (wassersteinDispersion_nonneg q P) hraw

/-- The aggregate carrier floor when marginal slack is controlled in synchronous `L²`. -/
theorem aggregateCarrierFloor_le_of_rms_slack
    {K : ℝ≥0} (hK : K ≠ 0) {t : X → H} (ht : AntilipschitzWith K t)
    (hme : MeasurableEmbedding t) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (T P : A → WassersteinMeasure H)
    (u : A → X → H) (htm : Measurable t) (hu : ∀ a, Measurable (u a))
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (hP : ∀ a, (P a).measure = (C a).measure.map (u a))
    (bridgeT : HilbertBarycenterBridge q T) (bridgeP : HilbertBarycenterBridge q P)
    (tau : A → ℝ≥0)
    (hrms : ∀ a, ∫⁻ x, (edist (u a x) (t x)) ^ 2 ∂(C a).measure ≤
      (tau a : ℝ≥0∞) ^ 2) :
    aggregateCarrierFloor K q C tau ≤ wassersteinDispersion q P := by
  apply aggregateCarrierFloor_le_of_antilipschitz hK ht hme q C T P hT bridgeT bridgeP tau
  intro a
  exact wassersteinDistance_le_of_lintegral_edist_sq_le
    (C a) (hu a) htm (hrms a) (P a) (T a) (hP a) (hT a)

/-- Unconditional aggregate carrier floor under synchronous integrated `L²` slack. -/
theorem aggregateCarrierFloor_le_of_rms_slack_unconditional
    {K : ℝ≥0} (hK : K ≠ 0) {t : X → H} (ht : AntilipschitzWith K t)
    (hme : MeasurableEmbedding t) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (T P : A → WassersteinMeasure H)
    (u : A → X → H) (htm : Measurable t) (hu : ∀ a, Measurable (u a))
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (hP : ∀ a, (P a).measure = (C a).measure.map (u a))
    (tau : A → ℝ≥0)
    (hrms : ∀ a, ∫⁻ x, (edist (u a x) (t x)) ^ 2 ∂(C a).measure ≤
      (tau a : ℝ≥0∞) ^ 2) :
    aggregateCarrierFloor K q C tau ≤ wassersteinDispersion q P := by
  apply aggregateCarrierFloor_le_of_antilipschitz_unconditional
    hK ht hme q C T P hT tau
  intro a
  exact wassersteinDistance_le_of_lintegral_edist_sq_le
    (C a) (hu a) htm (hrms a) (P a) (T a) (hP a) (hT a)

end PricingPerspective.Transmission.Dispersion
