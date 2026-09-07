import PricingPerspective.Transmission.Slack

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory
open WassersteinGeometry

/-!
# Carrier and slack lower certificates

This module isolates the lower transport implication needed by the portfolio
risk certificate.  `CarrierLowerBound` names the measure-level consequence of
an injective, quantitatively co-Lipschitz carrier.  The reverse transport theorem
now constructs the required measurable inverse coupling, while the named
predicate remains useful for separating carrier geometry from slack algebra.

Bounded synchronous slack then weakens the transported lower bound by exactly
the two slack radii.  No change to the random-exposure model is made.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {X H : Type*}
  [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]
  [NormedAddCommGroup H] [MeasurableSpace H] [BorelSpace H]
  [SecondCountableTopology H] [CompleteSpace H] [Inhabited H]

/-- The measure-level lower-transport implication supplied by a carrier map.

For an `L`-Lipschitz inverse on the carrier image this is
`L⁻¹ W₂(Cᵢ,Cⱼ) ≤ W₂(t#Cᵢ,t#Cⱼ)`.  Making the implication a named hypothesis
separates the economic identification assumption from the slack algebra.
-/
def CarrierLowerBound (L : ℝ) (Ci Cj : WassersteinMeasure X)
    (TA TB : WassersteinMeasure H) : Prop :=
  L⁻¹ * WassersteinDistance Ci Cj ≤ WassersteinDistance TA TB

omit [CompleteSpace H] in
/-- A pointwise antilipschitz measurable carrier supplies the measure-level
lower-transport implication; it is no longer an independent premise.

The proof uses `wassersteinDistance_le_antilipschitz_map`, whose coupling is
pulled back through the measurable inverse on the carrier image.
-/
theorem carrierLowerBound_of_antilipschitz
    {L : ℝ≥0} {t : X → H}
    (ht : AntilipschitzWith L t) (hme : MeasurableEmbedding t)
    (Ci Cj : WassersteinMeasure X) (TA TB : WassersteinMeasure H)
    (hTA : TA.measure = Ci.measure.map t) (hTB : TB.measure = Cj.measure.map t) :
    CarrierLowerBound L Ci Cj TA TB := by
  have htransport : WassersteinDistance Ci Cj ≤
      (L : ℝ) * WassersteinDistance TA TB :=
    wassersteinDistance_le_antilipschitz_map ht hme Ci Cj TA TB hTA hTB
  unfold CarrierLowerBound
  rcases eq_or_ne L 0 with hL | hL
  · subst hL
    have hnonneg : 0 ≤ WassersteinDistance TA TB := by
      unfold WassersteinDistance
      positivity
    simpa using hnonneg
  · have hLr : (L : ℝ) ≠ 0 := by exact_mod_cast hL
    calc
      (L : ℝ)⁻¹ * WassersteinDistance Ci Cj ≤
          (L : ℝ)⁻¹ * ((L : ℝ) * WassersteinDistance TA TB) :=
        mul_le_mul_of_nonneg_left htransport (inv_nonneg.mpr L.coe_nonneg)
      _ = WassersteinDistance TA TB := by
        field_simp

/-- The nonnegative pairwise exposure-distance certificate. -/
noncomputable def pairwiseExposureFloor (L τi τj : ℝ)
    (Ci Cj : WassersteinMeasure X) : ℝ :=
  max 0 (L⁻¹ * WassersteinDistance Ci Cj - τi - τj)

omit [OpensMeasurableSpace X] [SecondCountableTopology X] in
/-- **Carrier plus synchronous slack yields a computable exposure floor.**

If the transmitted laws retain at least `L⁻¹` of characteristic distance and
each realized exposure law lies within its slack radius of its transmitted
law, then

`max 0 (L⁻¹ W₂(Cᵢ,Cⱼ) - τᵢ - τⱼ) ≤ W₂(Pᵢ,Pⱼ)`.
-/
theorem pairwiseExposureFloor_le
    {L : ℝ} {τi τj : ℝ≥0} {t ui uj : X → H}
    (ht : Measurable t) (hui : Measurable ui) (huj : Measurable uj)
    (hτi : ∀ x, edist (ui x) (t x) ≤ (τi : ℝ≥0∞))
    (hτj : ∀ x, edist (uj x) (t x) ≤ (τj : ℝ≥0∞))
    (Ci Cj : WassersteinMeasure X)
    (Pi Pj TA TB : WassersteinMeasure H)
    (hPi : Pi.measure = Ci.measure.map ui) (hPj : Pj.measure = Cj.measure.map uj)
    (hTA : TA.measure = Ci.measure.map t) (hTB : TB.measure = Cj.measure.map t)
    (hcarrier : CarrierLowerBound L Ci Cj TA TB) :
    pairwiseExposureFloor L τi τj Ci Cj ≤ WassersteinDistance Pi Pj := by
  have hi : WassersteinDistance Pi TA ≤ (τi : ℝ) :=
    wassersteinDistance_le_of_edist_le Ci hui ht hτi Pi TA hPi hTA
  have hj : WassersteinDistance TB Pj ≤ (τj : ℝ) := by
    rw [wasserstein_symm]
    exact wassersteinDistance_le_of_edist_le Cj huj ht hτj Pj TB hPj hTB
  have htri₁ : WassersteinDistance TA TB ≤
      WassersteinDistance TA Pi + WassersteinDistance Pi TB :=
    wasserstein_triangle TA TB Pi
  have htri₂ : WassersteinDistance Pi TB ≤
      WassersteinDistance Pi Pj + WassersteinDistance Pj TB :=
    wasserstein_triangle Pi TB Pj
  have hnonneg : 0 ≤ WassersteinDistance Pi Pj := wasserstein_nonneg Pi Pj
  unfold CarrierLowerBound at hcarrier
  unfold pairwiseExposureFloor
  rw [max_le_iff]
  constructor
  · exact hnonneg
  · rw [wasserstein_symm TA Pi] at htri₁
    rw [wasserstein_symm Pj TB] at htri₂
    linarith

/-- **Pointwise antilipschitz carrier plus synchronous slack yields the
computable exposure floor.**

This discharges `CarrierLowerBound` from the manuscript's primitive carrier
assumption and the measurable inverse on its image.
-/
theorem pairwiseExposureFloor_le_of_antilipschitz
    {L τi τj : ℝ≥0} {t ui uj : X → H}
    (hanti : AntilipschitzWith L t) (hme : MeasurableEmbedding t)
    (hui : Measurable ui) (huj : Measurable uj)
    (hτi : ∀ x, edist (ui x) (t x) ≤ (τi : ℝ≥0∞))
    (hτj : ∀ x, edist (uj x) (t x) ≤ (τj : ℝ≥0∞))
    (Ci Cj : WassersteinMeasure X)
    (Pi Pj TA TB : WassersteinMeasure H)
    (hPi : Pi.measure = Ci.measure.map ui) (hPj : Pj.measure = Cj.measure.map uj)
    (hTA : TA.measure = Ci.measure.map t) (hTB : TB.measure = Cj.measure.map t) :
    pairwiseExposureFloor L τi τj Ci Cj ≤ WassersteinDistance Pi Pj :=
  pairwiseExposureFloor_le hme.measurable hui huj hτi hτj Ci Cj Pi Pj TA TB
    hPi hPj hTA hTB
    (carrierLowerBound_of_antilipschitz hanti hme Ci Cj TA TB hTA hTB)

end PricingPerspective.Transmission.Dispersion
