import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.VectorMeasure.Decomposition.Jordan
import PricingPerspective.Continuous.CovarianceSpace.FieldIntegral

/-!
# Signed-measure inner-product interchange

This module proves the Paper 1 residual/field interchange for signed exposures.
The proof expands the signed measure into its Jordan positive and negative
parts, applies ordinary Bochner integral interchange to each part, and then
reassembles the vector-measure integral.
-/

namespace PricingPerspective.ContinuousAPT.CovarianceSpace

open MeasureTheory

variable {Ω H : Type*} [MeasurableSpace Ω]
variable [NormedAddCommGroup H] [InnerProductSpace ℝ H] [CompleteSpace H]

noncomputable section

/--
An inner-product functional commutes with integration against a signed measure.

Integrability is stated against the Jordan total variation, exactly matching
the manuscript's `∫ σ d|β| < ∞` admissibility condition.  The proof reduces to
`integral_inner` on the positive and negative Jordan parts.  This is the
measure-theoretic interchange needed in Paper 1, Corollary 2.
-/
theorem inner_signedMeasureIntegral
    (F : Ω → H) (β : SignedMeasure Ω)
    (hF : Integrable F β.totalVariation) (ε : H) :
    inner ℝ ε (∫ᵛ ω, F ω ∂<•β) =
      ∫ᵛ ω, inner ℝ ε (F ω) ∂<•β := by
  rcases subsingleton_or_nontrivial H with hH | hH
  · have hε : ε = 0 := Subsingleton.elim _ _
    simp [hε]
  letI := hH
  let j := β.toJordanDecomposition
  have hpos : Integrable F j.posPart := by
    apply hF.mono_measure
    simpa [j, SignedMeasure.totalVariation] using
      (Measure.le_add_right (le_refl j.posPart) : j.posPart ≤ j.posPart + j.negPart)
  have hneg : Integrable F j.negPart := by
    apply hF.mono_measure
    simpa [j, SignedMeasure.totalVariation] using
      (Measure.le_add_left (le_refl j.negPart) : j.negPart ≤ j.posPart + j.negPart)
  have hposv : j.posPart.toSignedMeasure.Integrable F
      ((ContinuousLinearMap.lsmul ℝ ℝ (E := H)).flip) := by
    unfold VectorMeasure.Integrable
    rw [VectorMeasure.variation_transpose_lsmul_flip,
      VectorMeasure.variation_toSignedMeasure]
    exact hpos
  have hnegv : j.negPart.toSignedMeasure.Integrable F
      ((ContinuousLinearMap.lsmul ℝ ℝ (E := H)).flip) := by
    unfold VectorMeasure.Integrable
    rw [VectorMeasure.variation_transpose_lsmul_flip,
      VectorMeasure.variation_toSignedMeasure]
    exact hneg
  have hposi : Integrable (fun ω => inner ℝ ε (F ω)) j.posPart :=
    hpos.const_inner ε
  have hnegi : Integrable (fun ω => inner ℝ ε (F ω)) j.negPart :=
    hneg.const_inner ε
  have hposiv : j.posPart.toSignedMeasure.Integrable
      (fun ω => inner ℝ ε (F ω))
      ((ContinuousLinearMap.lsmul ℝ ℝ (E := ℝ)).flip) := by
    unfold VectorMeasure.Integrable
    rw [VectorMeasure.variation_transpose_lsmul_flip,
      VectorMeasure.variation_toSignedMeasure]
    exact hposi
  have hnegiv : j.negPart.toSignedMeasure.Integrable
      (fun ω => inner ℝ ε (F ω))
      ((ContinuousLinearMap.lsmul ℝ ℝ (E := ℝ)).flip) := by
    unfold VectorMeasure.Integrable
    rw [VectorMeasure.variation_transpose_lsmul_flip,
      VectorMeasure.variation_toSignedMeasure]
    exact hnegi
  rw [← β.toSignedMeasure_toJordanDecomposition]
  change inner ℝ ε
      (∫ᵛ ω, F ω ∂<•(j.posPart.toSignedMeasure - j.negPart.toSignedMeasure)) =
    ∫ᵛ ω, inner ℝ ε (F ω)
      ∂<•(j.posPart.toSignedMeasure - j.negPart.toSignedMeasure)
  rw [VectorMeasure.integral_sub_vectorMeasure hposv hnegv,
    VectorMeasure.integral_sub_vectorMeasure hposiv hnegiv]
  simp only [VectorMeasure.integral_toSignedMeasure]
  rw [inner_sub_right, ← integral_inner hpos ε, ← integral_inner hneg ε]

/--
Pointwise residual/field orthogonality survives signed integration.

If `inner ℝ ε (F ω) = 0` for every characteristic `ω`, then the residual
`ε` is orthogonal to the signed field integral.  This is the formal analogue
of `E[ε_i ∫ f(ω) β(dω)] = ∫ E[ε_i f(ω)] β(dω) = 0` in Paper 1.
-/
theorem residual_orthogonal_to_signedFieldIntegral
    (F : Ω → H) (β : SignedMeasure Ω)
    (hF : Integrable F β.totalVariation) (ε : H)
    (horth : ∀ ω, inner ℝ ε (F ω) = 0) :
    inner ℝ ε (∫ᵛ ω, F ω ∂<•β) = 0 := by
  rw [inner_signedMeasureIntegral F β hF ε]
  simp [horth]

/--
The bundled field integral is orthogonal to a pointwise-orthogonal residual
whenever the underlying signed exposure satisfies Jordan-total-variation
integrability.
-/
theorem residual_orthogonal_to_fieldIntegral
    (F : Ω → H) (β : AdmissibleExposure F)
    (hF : Integrable F (β.val.1 : SignedMeasure Ω).totalVariation) (ε : H)
    (horth : ∀ ω, inner ℝ ε (F ω) = 0) :
    inner ℝ ε (fieldIntegral F β) = 0 := by
  rw [fieldIntegral_apply]
  exact residual_orthogonal_to_signedFieldIntegral F β.val.1 hF ε horth

end

end PricingPerspective.ContinuousAPT.CovarianceSpace
