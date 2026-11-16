import Architect
import PricingPerspective.Transmission.Frechet
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.Moments.Variance

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry WassersteinGeometry.Hilbert

/-!
# From the coupling's inner product to the covariance of returns

`Hilbert.systCov` is an integral of an inner product. Everything the envelope says is about
that number, so the envelope means nothing to a finance reader until `systCov` is shown to
*be* the systematic covariance of two assets' returns. This file is that step, at the
measure level; `Transmission.Factor` does the same for finite clouds.

## The model

An asset's systematic return against a factor draw `F` is

  `Sᵢ = ⟪Bᵢ, F⟫`,

with the loading pair `(Bᵢ, Bⱼ)` drawn from a coupling `π` and the factor drawn
*independently*, so the joint law is the product `π ⊗ μ_F`. Under mean-zero, isotropic `μ_F`,

  `Cov(Sᵢ, Sⱼ) = 𝔼_π⟪Zᵢ, Zⱼ⟫ = systCov π`   (`factorReturnCovariance_eq_systCov`).

## Why isotropy is the right normalization, not a restriction

`isotropic` says `∫⟪u, F⟫⟪v, F⟫ dμ_F = ⟪u, v⟫`: the factor covariance operator is the
identity. That is not an assumption about the world, it is the statement that loadings are
already expressed in **risk coordinates** `Z = Γ^{1/2}B`, which is exactly the convention of
`Transmission.RiskCoordinates` and `Transmission.Defs`. A general `Γ` is recovered by
applying these results to `riskCoords`-transformed laws; `inner_riskCoords` is the
translation.

## Independence is an assumption, and it is doing work

The factor is drawn from the *product* `π ⊗ μ_F`, so loadings and factor innovations are
independent by construction. That is the standard factor-model premise and it is what lets
the factor integrate out. If loadings covaried with the factor — leverage responding to
volatility, say — the identity below would fail, and no amount of transport geometry would
repair it. Stated here rather than buried, because it is a modelling choice.

## Integrability is a premise

Fubini needs it, and this file does not manufacture it: each theorem takes the integrability
of the specific product integrand it needs. A caller over a concrete space discharges it once.

## References

* `Transmission.Factor` — the finite counterpart, with `IsotropicFactor` as a weighted cloud.
-/

namespace PricingPerspective.Transmission

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-- A mean-zero, isotropic factor law in risk coordinates.

    `mean_zero` is stated against every loading direction rather than as a Bochner mean,
    which is the form the return identities actually use and avoids a vector-valued integral.
    `isotropic` fixes the factor covariance operator to the identity — see the module
    docstring on why that is a coordinate convention, not a restriction. -/
structure IsotropicFactorLaw (H : Type*) [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    [MeasurableSpace H] where
  /-- The law of the factor innovation. -/
  law : Measure H
  /-- It is a probability law. -/
  is_probability : IsProbabilityMeasure law
  /-- The factor is unpriced in every direction: `𝔼⟪b, F⟫ = 0`. -/
  mean_zero : ∀ b : H, ∫ f, ⟪b, f⟫_ℝ ∂law = 0
  /-- Risk coordinates: the factor covariance operator is the identity. -/
  isotropic : ∀ u v : H, ∫ f, ⟪u, f⟫_ℝ * ⟪v, f⟫_ℝ ∂law = ⟪u, v⟫_ℝ

attribute [instance] IsotropicFactorLaw.is_probability

namespace IsotropicFactorLaw

variable (F : IsotropicFactorLaw H)

/-- The realized systematic return of a loading against a factor draw. -/
def factorReturn (b f : H) : ℝ := ⟪b, f⟫_ℝ

omit [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
@[simp] lemma factorReturn_apply (b f : H) : factorReturn b f = ⟪b, f⟫_ℝ := rfl

end IsotropicFactorLaw

open IsotropicFactorLaw

/-! ### Means and cross moments of factor returns -/

/-- The mean systematic return of the first asset, under a coupling and an independent factor. -/
noncomputable def factorReturnMeanFst
    (π : Measure (H × H)) (F : IsotropicFactorLaw H) : ℝ :=
  ∫ z, factorReturn z.1.1 z.2 ∂(π.prod F.law)

/-- The mean systematic return of the second asset. -/
noncomputable def factorReturnMeanSnd
    (π : Measure (H × H)) (F : IsotropicFactorLaw H) : ℝ :=
  ∫ z, factorReturn z.1.2 z.2 ∂(π.prod F.law)

/-- The cross moment of the two assets' systematic returns. -/
noncomputable def factorReturnCrossMoment
    (π : Measure (H × H)) (F : IsotropicFactorLaw H) : ℝ :=
  ∫ z, factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2 ∂(π.prod F.law)

omit [BorelSpace H] [Inhabited H] in
/-- **Systematic returns have mean zero** — first asset. -/
theorem factorReturnMeanFst_eq_zero
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (hint : Integrable (fun z : (H × H) × H => factorReturn z.1.1 z.2) (π.prod F.law)) :
    factorReturnMeanFst π F = 0 := by
  unfold factorReturnMeanFst
  rw [integral_prod _ hint]
  simp [factorReturn, F.mean_zero]

omit [BorelSpace H] [Inhabited H] in
/-- **Systematic returns have mean zero** — second asset. -/
theorem factorReturnMeanSnd_eq_zero
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (hint : Integrable (fun z : (H × H) × H => factorReturn z.1.2 z.2) (π.prod F.law)) :
    factorReturnMeanSnd π F = 0 := by
  unfold factorReturnMeanSnd
  rw [integral_prod _ hint]
  simp [factorReturn, F.mean_zero]

omit [BorelSpace H] [Inhabited H] in
/-- **The cross moment of systematic returns is the coupling's `systCov`.**

    Integrating out the independent factor collapses `𝔼⟪Zᵢ, F⟫⟪Zⱼ, F⟫` to `𝔼_π⟪Zᵢ, Zⱼ⟫`.
    This is the identity that gives the whole envelope its financial meaning: the abstract
    inner product the transport geometry bounds *is* the co-movement of two assets' returns. -/
theorem factorReturnCrossMoment_eq_systCov
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (hint : Integrable
      (fun z : (H × H) × H => factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2)
      (π.prod F.law)) :
    factorReturnCrossMoment π F = systCov π := by
  unfold factorReturnCrossMoment systCov
  rw [integral_prod _ hint]
  exact integral_congr_ae (Filter.Eventually.of_forall fun p => F.isotropic p.1 p.2)

omit [BorelSpace H] [Inhabited H] in
/-- **Systematic covariance of returns equals `systCov`.**

    Covariance is the cross moment minus the product of means, and the means vanish, so the
    two coincide. Every statement in `Transmission.Frechet` about `systCov` is therefore a
    statement about the covariance of two assets' systematic returns. -/
theorem factorReturnCovariance_eq_systCov
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (hfst : Integrable (fun z : (H × H) × H => factorReturn z.1.1 z.2) (π.prod F.law))
    (hsnd : Integrable (fun z : (H × H) × H => factorReturn z.1.2 z.2) (π.prod F.law))
    (hcross : Integrable
      (fun z : (H × H) × H => factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2)
      (π.prod F.law)) :
    factorReturnCrossMoment π F - factorReturnMeanFst π F * factorReturnMeanSnd π F
      = systCov π := by
  rw [factorReturnMeanFst_eq_zero π F hfst, factorReturnMeanSnd_eq_zero π F hsnd,
    factorReturnCrossMoment_eq_systCov π F hcross]
  ring

/-! ### Returns with an idiosyncratic component -/

omit [BorelSpace H] [Inhabited H] in
/-- **Residuals drop out of systematic covariance under cross-orthogonality.**

    Writing `Rᵢ = ⟪Zᵢ, F⟫ + εᵢ`, the covariance of total returns splits into the systematic
    cross moment plus three residual terms. If each of those integrates to zero — residuals
    uncorrelated with each other and with the factor returns — total-return covariance is
    exactly `systCov π`.

    The orthogonality conditions are premises, not consequences. They are the standard factor
    model's identifying assumptions, and the envelope inherits whatever they are worth: a
    claim about *total* return covariance is only as good as the claim that residuals are
    orthogonal. -/
theorem totalReturnCrossMoment_eq_systCov
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    -- each of the four product terms is integrable
    (hff : Integrable
      (fun z : (H × H) × H => factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2) (π.prod F.law))
    (hfη : Integrable (fun z : (H × H) × H => factorReturn z.1.1 z.2 * η z) (π.prod F.law))
    (hεf : Integrable (fun z : (H × H) × H => ε z * factorReturn z.1.2 z.2) (π.prod F.law))
    (hεη : Integrable (fun z : (H × H) × H => ε z * η z) (π.prod F.law))
    -- and the three cross-orthogonality conditions
    (h₁ : ∫ z, factorReturn z.1.1 z.2 * η z ∂(π.prod F.law) = 0)
    (h₂ : ∫ z, ε z * factorReturn z.1.2 z.2 ∂(π.prod F.law) = 0)
    (h₃ : ∫ z, ε z * η z ∂(π.prod F.law) = 0) :
    ∫ z, (factorReturn z.1.1 z.2 + ε z) * (factorReturn z.1.2 z.2 + η z) ∂(π.prod F.law)
      = systCov π := by
  -- Expand the product, then split the integral term by term.
  have hexp : (fun z : (H × H) × H =>
      (factorReturn z.1.1 z.2 + ε z) * (factorReturn z.1.2 z.2 + η z)) =
      fun z : (H × H) × H =>
        factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2 + factorReturn z.1.1 z.2 * η z
          + ε z * factorReturn z.1.2 z.2 + ε z * η z := by
    funext z; ring
  have h12 : Integrable (fun z : (H × H) × H =>
      factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2 + factorReturn z.1.1 z.2 * η z)
      (π.prod F.law) := hff.add hfη
  have h123 : Integrable (fun z : (H × H) × H =>
      factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2 + factorReturn z.1.1 z.2 * η z
        + ε z * factorReturn z.1.2 z.2) (π.prod F.law) := h12.add hεf
  rw [hexp, integral_add h123 hεη, integral_add h12 hεf, integral_add hff hfη,
    h₁, h₂, h₃, add_zero, add_zero, add_zero]
  exact factorReturnCrossMoment_eq_systCov π F hff

/-! ### Residual-robust return envelope -/

/-- The aggregate departure of total-return cross moments from systematic covariance.

    The three terms are factor-to-residual covariance in each direction and residual-to-
    residual covariance. Keeping their sum explicit separates a maintained zero-orthogonality
    restriction from a quantitative sensitivity bound. -/
noncomputable def residualCrossMoment
    (π : Measure (H × H)) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) : ℝ :=
  (∫ z, factorReturn z.1.1 z.2 * η z ∂(π.prod F.law))
    + (∫ z, ε z * factorReturn z.1.2 z.2 ∂(π.prod F.law))
    + ∫ z, ε z * η z ∂(π.prod F.law)

omit [BorelSpace H] [Inhabited H] in
/-- **Exact residual decomposition.**

    Total-return cross moments equal systematic covariance plus the three residual cross
    terms. The earlier zero-orthogonality theorem is the zero-remainder specialization. -/
theorem totalReturnCrossMoment_eq_systCov_add_residualCrossMoment
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hff : Integrable
      (fun z : (H × H) × H => factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2)
      (π.prod F.law))
    (hfη : Integrable (fun z : (H × H) × H => factorReturn z.1.1 z.2 * η z)
      (π.prod F.law))
    (hεf : Integrable (fun z : (H × H) × H => ε z * factorReturn z.1.2 z.2)
      (π.prod F.law))
    (hεη : Integrable (fun z : (H × H) × H => ε z * η z) (π.prod F.law)) :
    ∫ z, (factorReturn z.1.1 z.2 + ε z) * (factorReturn z.1.2 z.2 + η z)
        ∂(π.prod F.law)
      = systCov π + residualCrossMoment π F ε η := by
  have hexp : (fun z : (H × H) × H =>
      (factorReturn z.1.1 z.2 + ε z) * (factorReturn z.1.2 z.2 + η z)) =
      fun z : (H × H) × H =>
        factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2
          + factorReturn z.1.1 z.2 * η z
          + ε z * factorReturn z.1.2 z.2
          + ε z * η z := by
    funext z
    ring
  have h12 : Integrable (fun z : (H × H) × H =>
      factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2
        + factorReturn z.1.1 z.2 * η z) (π.prod F.law) := hff.add hfη
  have h123 : Integrable (fun z : (H × H) × H =>
      factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2
        + factorReturn z.1.1 z.2 * η z
        + ε z * factorReturn z.1.2 z.2) (π.prod F.law) := h12.add hεf
  rw [hexp, integral_add h123 hεη, integral_add h12 hεf, integral_add hff hfη]
  change factorReturnCrossMoment π F + _ + _ + _ = _
  rw [factorReturnCrossMoment_eq_systCov π F hff]
  unfold residualCrossMoment
  ring

omit [BorelSpace H] [Inhabited H] in
/-- **Residual contamination widens any systematic-covariance interval additively.**

    If systematic covariance lies in `[lo, hi]` and the aggregate residual cross moment is
    bounded in absolute value by `δ`, then the total-return cross moment lies in
    `[lo - δ, hi + δ]`. -/
theorem totalReturnCrossMoment_mem_Icc_of_residual_bound
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) (lo hi δ : ℝ)
    (hff : Integrable
      (fun z : (H × H) × H => factorReturn z.1.1 z.2 * factorReturn z.1.2 z.2)
      (π.prod F.law))
    (hfη : Integrable (fun z : (H × H) × H => factorReturn z.1.1 z.2 * η z)
      (π.prod F.law))
    (hεf : Integrable (fun z : (H × H) × H => ε z * factorReturn z.1.2 z.2)
      (π.prod F.law))
    (hεη : Integrable (fun z : (H × H) × H => ε z * η z) (π.prod F.law))
    (hlo : lo ≤ systCov π) (hhi : systCov π ≤ hi)
    (hδ : |residualCrossMoment π F ε η| ≤ δ) :
    (∫ z, (factorReturn z.1.1 z.2 + ε z) * (factorReturn z.1.2 z.2 + η z)
        ∂(π.prod F.law)) ∈ Set.Icc (lo - δ) (hi + δ) := by
  rw [totalReturnCrossMoment_eq_systCov_add_residualCrossMoment
    π F ε η hff hfη hεf hεη]
  have hrem := abs_le.mp hδ
  constructor <;> linarith

/-! ### Squared return distance and residual variance -/

/-- Expected squared distance between the two systematic returns on the product law.

    The factor draw is independent of the coupled loading pair by construction. Under
    isotropy this observable return-space distance is exactly the coupling's quadratic
    transport cost; see `factorReturnSquaredDistance_eq_transportCost`. -/
noncomputable def factorReturnSquaredDistance
    (π : Measure (H × H)) (F : IsotropicFactorLaw H) : ℝ :=
  ∫ z, (factorReturn z.1.1 z.2 - factorReturn z.1.2 z.2) ^ 2 ∂(π.prod F.law)

/-- Expected squared distance between total returns after adding residuals `ε` and `η`. -/
noncomputable def totalReturnSquaredDistance
    (π : Measure (H × H)) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) : ℝ :=
  ∫ z, ((factorReturn z.1.1 z.2 + ε z) -
    (factorReturn z.1.2 z.2 + η z)) ^ 2 ∂(π.prod F.law)

/-- Sum of the two residual variances under the coupled loading-factor product law.

    This is the nonnegative correction omitted when total-return squared distance is used
    as though it were systematic squared distance. -/
noncomputable def residualVarianceSum
    (π : Measure (H × H)) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) : ℝ :=
  ProbabilityTheory.variance ε (π.prod F.law) +
    ProbabilityTheory.variance η (π.prod F.law)

/-- The empirical total-return diagnostic relative to a proposed optimal transport cost. -/
noncomputable def totalReturnDiagnostic
    (π : Measure (H × H)) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) (optimalCost : ℝ) : ℝ :=
  totalReturnSquaredDistance π F ε η - optimalCost

omit [BorelSpace H] [Inhabited H] in
/-- **Isotropy identifies systematic return distance with loading transport cost.**

    Integrating over the independent factor draw sends
    `(⟪Zᵢ,F⟫ - ⟪Zⱼ,F⟫)²` to `‖Zᵢ-Zⱼ‖²`; integrating the loading pair then gives
    `transportCost π`. The stated integrability premise is exactly what Fubini requires. -/
theorem factorReturnSquaredDistance_eq_transportCost
    (π : Measure (H × H)) [SFinite π] (F : IsotropicFactorLaw H)
    (hint : Integrable
      (fun z : (H × H) × H =>
        (factorReturn z.1.1 z.2 - factorReturn z.1.2 z.2) ^ 2)
      (π.prod F.law)) :
    factorReturnSquaredDistance π F = transportCost π := by
  unfold factorReturnSquaredDistance transportCost
  rw [integral_prod _ hint]
  apply integral_congr_ae
  filter_upwards with p
  simpa [factorReturn, inner_sub_left, real_inner_self_eq_norm_sq, pow_two] using
    F.isotropic (p.1 - p.2) (p.1 - p.2)

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The residual variance correction is nonnegative, without distributional assumptions. -/
theorem residualVarianceSum_nonneg
    (π : Measure (H × H)) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ) :
    0 ≤ residualVarianceSum π F ε η := by
  unfold residualVarianceSum
  exact add_nonneg (ProbabilityTheory.variance_nonneg _ _)
    (ProbabilityTheory.variance_nonneg _ _)

omit [BorelSpace H] [Inhabited H] in
/-- **Total squared return distance includes both residual variances.**

    Let `Sᵢ = ⟪Zᵢ,F⟫` and `Sⱼ = ⟪Zⱼ,F⟫`. For centered square-integrable residuals,
    expanding `((Sᵢ+ε)-(Sⱼ+η))²` gives systematic squared distance plus the two residual
    variances when all four factor-to-residual cross moments and the residual cross moment
    vanish. Isotropy then identifies the systematic term with `transportCost π`.

    The four factor-residual conditions are stated separately: own-return orthogonality
    alone is insufficient because squared *differences* also contain the two cross-asset
    factor-residual products. -/
theorem totalReturnSquaredDistance_eq_transportCost_add_residualVarianceSum
    (π : Measure (H × H)) [IsProbabilityMeasure π] (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hsi : MemLp (fun z : (H × H) × H => factorReturn z.1.1 z.2) 2 (π.prod F.law))
    (hsj : MemLp (fun z : (H × H) × H => factorReturn z.1.2 z.2) 2 (π.prod F.law))
    (hε : MemLp ε 2 (π.prod F.law)) (hη : MemLp η 2 (π.prod F.law))
    (hεmean : ∫ z, ε z ∂(π.prod F.law) = 0)
    (hηmean : ∫ z, η z ∂(π.prod F.law) = 0)
    (hsiε : ∫ z, factorReturn z.1.1 z.2 * ε z ∂(π.prod F.law) = 0)
    (hsiη : ∫ z, factorReturn z.1.1 z.2 * η z ∂(π.prod F.law) = 0)
    (hsjε : ∫ z, factorReturn z.1.2 z.2 * ε z ∂(π.prod F.law) = 0)
    (hsjη : ∫ z, factorReturn z.1.2 z.2 * η z ∂(π.prod F.law) = 0)
    (hεη : ∫ z, ε z * η z ∂(π.prod F.law) = 0) :
    totalReturnSquaredDistance π F ε η =
      transportCost π + residualVarianceSum π F ε η := by
  let si : (H × H) × H → ℝ := fun z => factorReturn z.1.1 z.2
  let sj : (H × H) × H → ℝ := fun z => factorReturn z.1.2 z.2
  let μ : Measure ((H × H) × H) := π.prod F.law
  have hsys : MemLp (fun z => si z - sj z) 2 μ := hsi.sub hsj
  have hsysε : Integrable (fun z => (si z - sj z) * ε z) μ :=
    hsys.integrable_mul hε
  have hsysη : Integrable (fun z => (si z - sj z) * η z) μ :=
    hsys.integrable_mul hη
  have hεηint : Integrable (fun z => ε z * η z) μ := hε.integrable_mul hη
  have hsysεzero : ∫ z, (si z - sj z) * ε z ∂μ = 0 := by
    change ∫ z, (factorReturn z.1.1 z.2 - factorReturn z.1.2 z.2) * ε z
      ∂(π.prod F.law) = 0
    calc
      _ = ∫ z, factorReturn z.1.1 z.2 * ε z -
          factorReturn z.1.2 z.2 * ε z ∂(π.prod F.law) := by
        congr with z
        ring
      _ = (∫ z, factorReturn z.1.1 z.2 * ε z ∂(π.prod F.law)) -
          (∫ z, factorReturn z.1.2 z.2 * ε z ∂(π.prod F.law)) :=
        integral_sub (hsi.integrable_mul hε) (hsj.integrable_mul hε)
      _ = 0 := by rw [hsiε, hsjε]; ring
  have hsysηzero : ∫ z, (si z - sj z) * η z ∂μ = 0 := by
    change ∫ z, (factorReturn z.1.1 z.2 - factorReturn z.1.2 z.2) * η z
      ∂(π.prod F.law) = 0
    calc
      _ = ∫ z, factorReturn z.1.1 z.2 * η z -
          factorReturn z.1.2 z.2 * η z ∂(π.prod F.law) := by
        congr with z
        ring
      _ = (∫ z, factorReturn z.1.1 z.2 * η z ∂(π.prod F.law)) -
          (∫ z, factorReturn z.1.2 z.2 * η z ∂(π.prod F.law)) :=
        integral_sub (hsi.integrable_mul hη) (hsj.integrable_mul hη)
      _ = 0 := by rw [hsiη, hsjη]; ring
  have hexp : (fun z => ((si z + ε z) - (sj z + η z)) ^ 2) =
      fun z => (si z - sj z) ^ 2 + ε z ^ 2 + η z ^ 2 +
        2 * ((si z - sj z) * ε z) - 2 * ((si z - sj z) * η z) -
        2 * (ε z * η z) := by
    funext z
    ring
  have hsys2 : Integrable (fun z => (si z - sj z) ^ 2) μ := hsys.integrable_sq
  have hε2 : Integrable (fun z => ε z ^ 2) μ := hε.integrable_sq
  have hη2 : Integrable (fun z => η z ^ 2) μ := hη.integrable_sq
  have hsyscost : ∫ z, (si z - sj z) ^ 2 ∂μ = transportCost π := by
    change factorReturnSquaredDistance π F = transportCost π
    exact factorReturnSquaredDistance_eq_transportCost π F hsys2
  have hεηzero : ∫ z, ε z * η z ∂μ = 0 := hεη
  have hsum2 : Integrable (fun z => (si z - sj z) ^ 2 + ε z ^ 2) μ := hsys2.add hε2
  have hsum3 : Integrable
      (fun z => (si z - sj z) ^ 2 + ε z ^ 2 + η z ^ 2) μ := hsum2.add hη2
  have hsysε2 : Integrable (fun z => 2 * ((si z - sj z) * ε z)) μ :=
    hsysε.const_mul 2
  have hsysη2 : Integrable (fun z => 2 * ((si z - sj z) * η z)) μ :=
    hsysη.const_mul 2
  have hεη2 : Integrable (fun z => 2 * (ε z * η z)) μ := hεηint.const_mul 2
  have hsum4 : Integrable
      (fun z => (si z - sj z) ^ 2 + ε z ^ 2 + η z ^ 2 +
        2 * ((si z - sj z) * ε z)) μ := hsum3.add hsysε2
  have hsum5 : Integrable
      (fun z => (si z - sj z) ^ 2 + ε z ^ 2 + η z ^ 2 +
        2 * ((si z - sj z) * ε z) - 2 * ((si z - sj z) * η z)) μ :=
    hsum4.sub hsysη2
  unfold totalReturnSquaredDistance
  change (∫ z, ((si z + ε z) - (sj z + η z)) ^ 2 ∂μ) = _
  rw [hexp]
  rw [integral_sub hsum5 hεη2, integral_sub hsum4 hsysη2,
    integral_add hsum3 hsysε2, integral_add hsum2 hη2, integral_add hsys2 hε2,
    integral_const_mul, integral_const_mul, integral_const_mul,
    hsyscost, hsysεzero, hsysηzero, hεηzero]
  unfold residualVarianceSum
  rw [ProbabilityTheory.variance_of_integral_eq_zero hε.aemeasurable hεmean,
    ProbabilityTheory.variance_of_integral_eq_zero hη.aemeasurable hηmean]
  ring

omit [BorelSpace H] [SecondCountableTopology H] in
/-- **The total-return diagnostic equals systematic excess plus residual variance.**

    Once the return-distance decomposition has been discharged from the explicit moment
    conditions above, subtracting the certified optimal cost gives
    `H = Δ + q`, where `Δ` is systematic excess transport cost and `q` is the residual
    variance sum. This is the precise correction to the unadjusted diagnostic. -/
theorem totalReturnDiagnostic_eq_excess_add_residualVarianceSum_of_decomposition
    {P Q : WassersteinMeasure H} (cert : OptimalTransportCost P Q)
    (π : IntegrableCoupling P Q) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hdecomp : totalReturnSquaredDistance π.toMeasure F ε η =
      transportCost π.toMeasure + residualVarianceSum π.toMeasure F ε η) :
    totalReturnDiagnostic π.toMeasure F ε η cert.value =
      excess cert π + residualVarianceSum π.toMeasure F ε η := by
  unfold totalReturnDiagnostic excess
  rw [hdecomp]
  ring

omit [BorelSpace H] in
/-- **Exact total-return diagnostic correction from the return model.**

    Under square integrability, centered residuals, orthogonality of each residual to both
    systematic returns, and zero residual cross moment, the observed total-return diagnostic
    is `H = Δ + q`: systematic excess transport cost plus the two residual variances.

    The coupling itself is a probability measure because its marginals are probability laws;
    no separate probability premise is imposed on the caller. -/
@[blueprint "eq:total-return-excess-correction"
  (statement := /--
    Under an independent centered isotropic factor law, square-integrable systematic returns and
    centered residuals, and zero systematic-residual and cross-residual moments, total-return
    squared distance minus certified optimal exposure cost equals systematic transport excess
    plus the two residual variances. Assumptions: an integrable coupling of exposure laws and a
    certified optimal transport plan; a centered isotropic factor law independent of the joint
    exposure draw; square-integrable systematic returns and centered square-integrable
    residuals; all four systematic-residual cross moments and the cross-residual moment are
    zero.
  -/)]
theorem totalReturnDiagnostic_eq_excess_add_residualVarianceSum
    {P Q : WassersteinMeasure H} (cert : OptimalTransportCost P Q)
    (π : IntegrableCoupling P Q) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hsi : MemLp (fun z : (H × H) × H => factorReturn z.1.1 z.2) 2
      (π.toMeasure.prod F.law))
    (hsj : MemLp (fun z : (H × H) × H => factorReturn z.1.2 z.2) 2
      (π.toMeasure.prod F.law))
    (hε : MemLp ε 2 (π.toMeasure.prod F.law))
    (hη : MemLp η 2 (π.toMeasure.prod F.law))
    (hεmean : ∫ z, ε z ∂(π.toMeasure.prod F.law) = 0)
    (hηmean : ∫ z, η z ∂(π.toMeasure.prod F.law) = 0)
    (hsiε : ∫ z, factorReturn z.1.1 z.2 * ε z ∂(π.toMeasure.prod F.law) = 0)
    (hsiη : ∫ z, factorReturn z.1.1 z.2 * η z ∂(π.toMeasure.prod F.law) = 0)
    (hsjε : ∫ z, factorReturn z.1.2 z.2 * ε z ∂(π.toMeasure.prod F.law) = 0)
    (hsjη : ∫ z, factorReturn z.1.2 z.2 * η z ∂(π.toMeasure.prod F.law) = 0)
    (hεη : ∫ z, ε z * η z ∂(π.toMeasure.prod F.law) = 0) :
    totalReturnDiagnostic π.toMeasure F ε η cert.value =
      excess cert π + residualVarianceSum π.toMeasure F ε η := by
  letI : IsProbabilityMeasure π.toMeasure :=
    coupling_isProbabilityMeasure P.is_probability π.mem
  apply totalReturnDiagnostic_eq_excess_add_residualVarianceSum_of_decomposition
  exact totalReturnSquaredDistance_eq_transportCost_add_residualVarianceSum
    π.toMeasure F ε η hsi hsj hε hη hεmean hηmean hsiε hsiη hsjε hsjη hεη

omit [BorelSpace H] [SecondCountableTopology H] in
/-- The unadjusted total-return diagnostic is an upper bound on systematic excess.

    The gap is exactly the nonnegative residual variance sum, so total-return distance can
    make the systematic excess look too large but cannot make it too small under the stated
    return-distance decomposition. -/
theorem excess_le_totalReturnDiagnostic
    {P Q : WassersteinMeasure H} (cert : OptimalTransportCost P Q)
    (π : IntegrableCoupling P Q) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hdecomp : totalReturnSquaredDistance π.toMeasure F ε η =
      transportCost π.toMeasure + residualVarianceSum π.toMeasure F ε η) :
    excess cert π ≤ totalReturnDiagnostic π.toMeasure F ε η cert.value := by
  rw [totalReturnDiagnostic_eq_excess_add_residualVarianceSum_of_decomposition
    cert π F ε η hdecomp]
  exact le_add_of_nonneg_right (residualVarianceSum_nonneg π.toMeasure F ε η)

omit [BorelSpace H] [SecondCountableTopology H] in
/-- A negative total-return diagnostic forces the systematic excess to be negative.

    Certified optimality also gives `0 ≤ excess cert π`. Hence a negative observed
    diagnostic contradicts at least one maintained bridge condition or the supplied
    optimal-cost calibration; residual variance cannot explain the negative sign. -/
theorem excess_neg_of_totalReturnDiagnostic_neg
    {P Q : WassersteinMeasure H} (cert : OptimalTransportCost P Q)
    (π : IntegrableCoupling P Q) (F : IsotropicFactorLaw H)
    (ε η : (H × H) × H → ℝ)
    (hdecomp : totalReturnSquaredDistance π.toMeasure F ε η =
      transportCost π.toMeasure + residualVarianceSum π.toMeasure F ε η)
    (hneg : totalReturnDiagnostic π.toMeasure F ε η cert.value < 0) :
    excess cert π < 0 :=
  lt_of_le_of_lt (excess_le_totalReturnDiagnostic cert π F ε η hdecomp) hneg


end PricingPerspective.Transmission
