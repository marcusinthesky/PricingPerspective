import PricingPerspective.Transmission.Frechet
import Mathlib.MeasureTheory.Integral.Prod

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

end PricingPerspective.Transmission
