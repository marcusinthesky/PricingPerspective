import WassersteinGeometry.Hilbert.Defs

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory

/-!
# Polarization and the covariance envelope on `P₂(H)`

The identity that makes quadratic transport the right geometry for systematic covariance.
For any coupling `π` of two loading laws in risk coordinates,

  `𝔼_π‖Z − Y‖² = v(P) + v(Q) − 2·𝔼_π⟪Z, Y⟫`,

so **minimizing transport cost and maximizing systematic covariance are the same problem**.
Taking the infimum over `Π(P, Q)` on the left gives `W₂²`, and the covariance ceiling

  `κ̄ = (v(P) + v(Q) − W₂²) / 2`

follows. Nothing here optimizes anything: it converts one optimization into the other, and
the sharpness that buys is precisely what an unoptimized distance such as energy cannot
deliver.

## Scope, honestly stated

* `transportCost_eq_polarization` holds for **every** coupling, not just optimal ones. It is
  an identity, not a bound.
* `systCov_le_envelope` bounds covariance under a *given* coupling by the ceiling. It does
  **not** say the ceiling is attained; attainment needs an optimal coupling, which the
  `OptimalCoupling` certificate in `Geodesics` supplies and which no theorem in this
  workspace proves to exist.
* Consequently, small `W₂` constrains covariance from below and large `W₂` merely relaxes
  the constraint. The relationship is one-sided; a regression of realized covariance on
  `W₂` tests a reduced form, not this theorem.

## References

* Panaretos & Zemel (2020), *An Invitation to Statistics in Wasserstein Space*, Ch. 2.
* Puccetti & Wang (2015), extremal dependence and quadratic transport.
-/

namespace WassersteinGeometry.Hilbert

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-! ### The polarization identity -/

-- `SecondCountableTopology` is unused in the *statement* but required by the proof, via
-- the `AEStronglyMeasurable` side goals of `secondMoment_{fst,snd}_of_mem_couplingSet`.
-- The linter only inspects statements, so its suggestion to `omit` here is wrong.
/-- **Polarization for couplings.**

    `𝔼_π‖Z − Y‖² = v(P) + v(Q) − 2·κ_π` for every coupling `π ∈ Π(P, Q)`.

    The second moments are marginal quantities, so the only coupling-dependent term is the
    covariance: transport cost and systematic covariance move in exact opposition. -/
theorem transportCost_eq_polarization
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure)
    (h₁ : Integrable (fun p : H × H => ‖p.1‖ ^ 2) π)
    (h₂ : Integrable (fun p : H × H => ‖p.2‖ ^ 2) π)
    (hc : Integrable (fun p : H × H => ⟪p.1, p.2⟫_ℝ) π) :
    transportCost π = secondMoment P + secondMoment Q - 2 * systCov π := by
  rw [secondMoment_fst_of_mem_couplingSet hπ, secondMoment_snd_of_mem_couplingSet hπ]
  unfold transportCost systCov
  have hrw : (fun p : H × H => ‖p.1 - p.2‖ ^ 2) =
      fun p : H × H => ‖p.1‖ ^ 2 - 2 * ⟪p.1, p.2⟫_ℝ + ‖p.2‖ ^ 2 := by
    funext p
    exact norm_sub_sq_real p.1 p.2
  -- Ascribe the integrability types so `integral_add`/`integral_sub` unify against the
  -- lambda the integrand actually is, rather than a pointwise sum of two functions.
  have hmul : Integrable (fun p : H × H => 2 * ⟪p.1, p.2⟫_ℝ) π := hc.const_mul 2
  have hsub : Integrable (fun p : H × H => ‖p.1‖ ^ 2 - 2 * ⟪p.1, p.2⟫_ℝ) π := h₁.sub hmul
  rw [hrw, integral_add hsub h₂, integral_sub h₁ hmul, integral_const_mul]
  ring

/-- Systematic covariance, solved out of the polarization identity. -/
theorem systCov_eq_of_polarization
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure)
    (h₁ : Integrable (fun p : H × H => ‖p.1‖ ^ 2) π)
    (h₂ : Integrable (fun p : H × H => ‖p.2‖ ^ 2) π)
    (hc : Integrable (fun p : H × H => ⟪p.1, p.2⟫_ℝ) π) :
    systCov π = (secondMoment P + secondMoment Q - transportCost π) / 2 := by
  have := transportCost_eq_polarization hπ h₁ h₂ hc
  linarith

/-! ### Bridging the `ℝ≥0∞` cost of `Geodesics` to the Bochner cost -/

omit [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The lower Lebesgue cost of `Geodesics` is the `ℝ≥0∞` image of the Bochner cost.

    `Geodesics` measures a coupling by `∫⁻ edist(x,y)² dπ` because that is always defined;
    polarization needs the signed Bochner integral. Under integrability the two agree, which
    is what connects the envelope below to the actual `W₂`. -/
theorem lintegral_edist_sq_eq_ofReal_transportCost
    {π : Measure (H × H)}
    (h : Integrable (fun p : H × H => ‖p.1 - p.2‖ ^ 2) π) :
    ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π = ENNReal.ofReal (transportCost π) := by
  have hpt : ∀ p : H × H, (edist p.1 p.2) ^ 2 = ENNReal.ofReal (‖p.1 - p.2‖ ^ 2) := by
    intro p
    rw [edist_dist, dist_eq_norm, ← ENNReal.ofReal_pow (norm_nonneg _)]
  calc ∫⁻ p, (edist p.1 p.2) ^ 2 ∂π
      = ∫⁻ p, ENNReal.ofReal (‖p.1 - p.2‖ ^ 2) ∂π := by simp_rw [hpt]
    _ = ENNReal.ofReal (transportCost π) :=
        (ofReal_integral_eq_lintegral_ofReal h (Filter.Eventually.of_forall fun p => by positivity)).symm

omit [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] in
/-- `W₂²` is a lower bound on the real transport cost of any admissible coupling. -/
theorem wassersteinDistanceSq_le_ofReal_transportCost
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure)
    (h : Integrable (fun p : H × H => ‖p.1 - p.2‖ ^ 2) π) :
    WassersteinDistanceSq P Q ≤ ENNReal.ofReal (transportCost π) := by
  rw [← lintegral_edist_sq_eq_ofReal_transportCost h]
  exact sInf_le ⟨π, hπ, rfl⟩

omit [InnerProductSpace ℝ H] [BorelSpace H] [SecondCountableTopology H] in
/-- The real form: `W₂²(P,Q) ≤ 𝔼_π‖Z − Y‖²` for every admissible coupling. -/
theorem wassersteinDistanceSq_toReal_le_transportCost
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure)
    (h : Integrable (fun p : H × H => ‖p.1 - p.2‖ ^ 2) π) :
    (WassersteinDistanceSq P Q).toReal ≤ transportCost π := by
  have hcost_nonneg : 0 ≤ transportCost π := by
    unfold transportCost
    exact integral_nonneg fun p => by positivity
  have hle := wassersteinDistanceSq_le_ofReal_transportCost hπ h
  have := ENNReal.toReal_mono ENNReal.ofReal_ne_top hle
  rwa [ENNReal.toReal_ofReal hcost_nonneg] at this

/-! ### The covariance envelope -/

/-- **Covariance envelope.**

    `κ_π ≤ (v(P) + v(Q) − W₂²(P,Q)) / 2` for every coupling `π ∈ Π(P, Q)`.

    Two marginal loading laws cap how much systematic covariance any joint law of the two
    assets can carry, and quadratic transport is exactly what sets the cap. Read in reverse:
    a *small* `W₂` forces the cap high, so it can exclude low covariance; a large `W₂`
    lowers the cap and excludes nothing. The bound is one-sided by construction.

    This does not identify realized covariance. Which coupling nature selects is an economic
    question that marginals cannot answer. -/
theorem systCov_le_envelope
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure)
    (h₁ : Integrable (fun p : H × H => ‖p.1‖ ^ 2) π)
    (h₂ : Integrable (fun p : H × H => ‖p.2‖ ^ 2) π)
    (hc : Integrable (fun p : H × H => ⟪p.1, p.2⟫_ℝ) π) :
    systCov π ≤
      (secondMoment P + secondMoment Q - (WassersteinDistanceSq P Q).toReal) / 2 := by
  have hcost := transportCost_integrable_of h₁ h₂ hc
  have hW := wassersteinDistanceSq_toReal_le_transportCost hπ hcost
  have hpol := systCov_eq_of_polarization hπ h₁ h₂ hc
  rw [hpol]
  linarith

end WassersteinGeometry.Hilbert
