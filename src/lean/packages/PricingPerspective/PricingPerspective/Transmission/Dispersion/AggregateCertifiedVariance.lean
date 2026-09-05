import PricingPerspective.Transmission.Dispersion.AggregateCarrier
import PricingPerspective.Transmission.Dispersion.OptimalVariance

set_option linter.style.longLine false

open scoped ENNReal NNReal InnerProductSpace MeasureTheory
open MeasureTheory Finset
open WassersteinGeometry WassersteinGeometry.Hilbert
open WassersteinGeometry.Multimarginal

/-!
# Sharp information-certified portfolio variance

The pairwise certificate of `CertifiedVariance.lean` relaxes the multi-marginal dispersion
to a weighted sum of pairwise floors and subtracts each pair's slack radii separately.  This
module states the sharp form: the multi-marginal infimum caps systematic variance directly,
and the aggregate carrier floor bounds that infimum from below after subtracting one weighted
root-mean-square slack radius.

Both steps compose results proved elsewhere.  Attainment is *not* assumed: the envelope
bound uses only that the multi-marginal dispersion is an infimum over coherent joint
couplings, so no `OptimalJointDispersion` certificate is required.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A X H : Type*} [Fintype A]
  [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]
  [MeasurableSpace H] [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [CompleteSpace H] [SecondCountableTopology H] [BorelSpace H] [Inhabited H]

omit [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X] [OpensMeasurableSpace X]
  [SecondCountableTopology X] [CompleteSpace H] [SecondCountableTopology H] in
/-- **Attainment-free systematic-variance envelope.**

Every coherent joint law with integrable pairwise inner products has systematic portfolio
variance at most the weighted marginal second moments minus the multi-marginal dispersion
infimum.  Unlike `jointPortfolioVariance_le_optimalEnvelope`, this takes no
`OptimalJointDispersion`: the bound follows from `sInf_le` applied to the plan in hand.
-/
theorem jointPortfolioVariance_le_dispersionEnvelope
    (q : ProbabilityWeight A) {P : A → WassersteinMeasure H} (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    PricingPerspective.Transmission.portfolioVariance J.measure q ≤
      (∑ a, q a * secondMoment (P a)) - (wassersteinDispersionSq q P).toReal := by
  have hmem : wassersteinDispersionSq q P ≤ weightedDispersionCost q J := by
    unfold wassersteinDispersionSq
    exact sInf_le ⟨J, rfl⟩
  have hreal : (wassersteinDispersionSq q P).toReal ≤ (weightedDispersionCost q J).toReal :=
    ENNReal.toReal_mono (weightedDispersionCost_ne_top q J hint) hmem
  rw [jointPortfolioVariance_eq_secondMoments_sub_realizedDispersion q J hint,
    ← weightedDispersionCost_toReal_eq_realizedJointDispersion q J hint]
  linarith

/-- **Sharp information-certified coherent portfolio-variance bound.**

The observable aggregate carrier floor — inverse carrier distortion applied to multi-firm
characteristic dispersion, less one weighted root-mean-square transmission radius, truncated
at zero — deducts its square from coherent systematic portfolio variance.

This is the multi-marginal counterpart of `jointPortfolioVariance_le_informationCertifiedCap`.
It is sharper in two independent ways: the dispersion infimum dominates any weighted sum of
pairwise floors, and slack is subtracted once in aggregate rather than per pair.  It is
conditional on the same carrier, slack, joint-law, and integrability premises, and calibrates
none of them.
-/
theorem jointPortfolioVariance_le_aggregateCertifiedCap
    {K : ℝ≥0} (hK : K ≠ 0) {t : X → H} (ht : AntilipschitzWith K t)
    (hme : MeasurableEmbedding t) (q : ProbabilityWeight A)
    (C : A → WassersteinMeasure X) (T P : A → WassersteinMeasure H)
    (u : A → X → H) (htm : Measurable t) (hu : ∀ a, Measurable (u a))
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (hP : ∀ a, (P a).measure = (C a).measure.map (u a))
    (tau : A → ℝ≥0)
    (hrms : ∀ a, ∫⁻ x, (edist (u a x) (t x)) ^ 2 ∂(C a).measure ≤
      (tau a : ℝ≥0∞) ^ 2)
    (J : JointCoupling P)
    (hint : ∀ a b, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J.measure) :
    PricingPerspective.Transmission.portfolioVariance J.measure q ≤
      (∑ a, q a * secondMoment (P a)) - aggregateCarrierFloor K q C tau ^ 2 := by
  have hfloor : aggregateCarrierFloor K q C tau ≤ wassersteinDispersion q P :=
    aggregateCarrierFloor_le_of_rms_slack_unconditional
      hK ht hme q C T P u htm hu hT hP tau hrms
  have hnonneg : 0 ≤ aggregateCarrierFloor K q C tau := le_max_left _ _
  have hsq : aggregateCarrierFloor K q C tau ^ 2 ≤ (wassersteinDispersionSq q P).toReal := by
    rw [← wassersteinDispersion_sq_eq_toReal]
    nlinarith [hnonneg, hfloor, wassersteinDispersion_nonneg q P]
  have henvelope := jointPortfolioVariance_le_dispersionEnvelope q J hint
  linarith

end PricingPerspective.Transmission.Dispersion
