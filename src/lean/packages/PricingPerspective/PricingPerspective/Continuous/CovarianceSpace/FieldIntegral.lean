import PricingPerspective.Continuous.CovarianceSpace.Defs

/-!
# Integration of admissible signed exposures

This module integrates a Hilbert- or Banach-valued factor field against the admissible signed
exposures from `Defs`.  The result is a linear map from exposures into the factor space.
-/

namespace PricingPerspective.ContinuousAPT.CovarianceSpace

open MeasureTheory

variable {Ω H : Type*} [MeasurableSpace Ω]
variable [NormedAddCommGroup H] [NormedSpace ℝ H]

noncomputable section

private abbrev signedMeasurePairing : H →L[ℝ] ℝ →L[ℝ] H :=
  (ContinuousLinearMap.lsmul ℝ ℝ (E := H)).flip

/--
Admissibility against variation implies integrability for mathlib's vector-measure integral
against the signed measure itself.
-/
theorem admissibleExposure_vectorMeasure_integrable {F : Ω → H}
    (β : admissibleExposureSubmodule F) :
    (β.1 : SignedMeasure Ω).Integrable F signedMeasurePairing := by
  apply (β.2.smul_measure_nnreal
    (c := ‖signedMeasurePairing (H := H)‖₊)).mono_measure
  exact VectorMeasure.variation_transpose_le β.1 signedMeasurePairing

private def fieldIntegralSubmodule [CompleteSpace H] (F : Ω → H) :
    admissibleExposureSubmodule F →ₗ[ℝ] H where
  toFun β := ∫ᵛ ω, F ω ∂<•(β.1 : SignedMeasure Ω)
  map_add' β γ := by
    change (∫ᵛ ω, F ω ∂<•((β.1 + γ.1) : SignedMeasure Ω)) =
      (∫ᵛ ω, F ω ∂<•(β.1 : SignedMeasure Ω)) +
        ∫ᵛ ω, F ω ∂<•(γ.1 : SignedMeasure Ω)
    exact VectorMeasure.integral_add_vectorMeasure
      (admissibleExposure_vectorMeasure_integrable β)
      (admissibleExposure_vectorMeasure_integrable γ)
  map_smul' c β := by
    change (∫ᵛ ω, F ω ∂<•((c • β.1) : SignedMeasure Ω)) =
      c • ∫ᵛ ω, F ω ∂<•(β.1 : SignedMeasure Ω)
    exact VectorMeasure.integral_smul_vectorMeasure F c

/--
The signed-measure Bochner integral of a factor field, as a linear map on coherently bundled
admissible exposures.

Completeness of `H` is the standard Bochner-integral target requirement.  Linearity in the
signed measure follows from vector-measure integral linearity together with admissibility.
-/
def fieldIntegral [CompleteSpace H] (F : Ω → H) : AdmissibleExposure F →ₗ[ℝ] H :=
  (fieldIntegralSubmodule F).comp (admissibleExposureLinearEquiv F).toLinearMap

/--
Evaluating `fieldIntegral F` is the vector-measure integral of `F` against the underlying
signed exposure.
-/
@[simp]
theorem fieldIntegral_apply [CompleteSpace H]
    (F : Ω → H) (β : AdmissibleExposure F) :
    fieldIntegral F β = ∫ᵛ ω, F ω ∂<•(β.val.1 : SignedMeasure Ω) :=
  rfl

/-- `fieldIntegral F` sends the zero exposure to zero. -/
@[simp]
theorem fieldIntegral_zero [CompleteSpace H] (F : Ω → H) : fieldIntegral F 0 = 0 :=
  map_zero (fieldIntegral F)

/-- `fieldIntegral F` sends sums of admissible exposures to sums. -/
theorem fieldIntegral_add [CompleteSpace H]
    (F : Ω → H) (β γ : AdmissibleExposure F) :
    fieldIntegral F (β + γ) = fieldIntegral F β + fieldIntegral F γ :=
  map_add (fieldIntegral F) β γ

/-- `fieldIntegral F` commutes with real scalar multiplication of exposures. -/
theorem fieldIntegral_smul [CompleteSpace H]
    (F : Ω → H) (c : ℝ) (β : AdmissibleExposure F) :
    fieldIntegral F (c • β) = c • fieldIntegral F β :=
  map_smul (fieldIntegral F) c β

end

end PricingPerspective.ContinuousAPT.CovarianceSpace
