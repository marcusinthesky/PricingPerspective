import Mathlib.Algebra.Module.TransferInstance
import Mathlib.MeasureTheory.VectorMeasure.Integral

/-!
# Admissible signed-measure exposures

This module defines the linear class of signed exposure measures whose total variation
integrates a Hilbert-valued factor field.  The condition is the coordinate-free form of
`∫ ‖F ω‖ ∂|β| < ∞`: mathlib's Bochner `Integrable` predicate packages both the required
almost-everywhere strong measurability and finiteness of the norm integral.

The characteristic-space first-moment condition used elsewhere in Paper 1 is deliberately
separate.  Here admissibility is relative to the factor field `F` and the variation of the
exposure measure.
-/

namespace PricingPerspective.ContinuousAPT.CovarianceSpace

open MeasureTheory

variable {Ω H : Type*} [MeasurableSpace Ω]
variable [NormedAddCommGroup H]

/--
The admissible signed exposures for a factor field `F : Ω → H`.

An exposure `β` is admitted exactly when `F` is Bochner integrable with respect to the
variation measure `β.variation`.  This is closed under addition and real scalar
multiplication by the variation inequalities for vector measures.
-/
noncomputable def admissibleExposureSubmodule (F : Ω → H) :
    Submodule ℝ (SignedMeasure Ω) where
  carrier := {β | Integrable F β.variation}
  zero_mem' := by simp
  add_mem' := by
    intro β γ hβ hγ
    apply (integrable_add_measure.2 ⟨hβ, hγ⟩).mono_measure
    exact VectorMeasure.variation_add_le
  smul_mem' := by
    intro c β hβ
    change Integrable F (c • β).variation
    rw [VectorMeasure.variation_smul]
    exact hβ.smul_measure_nnreal

/--
Membership in `admissibleExposureSubmodule F` is precisely Bochner integrability of `F`
against the exposure's variation measure.
-/
@[simp]
theorem mem_admissibleExposureSubmodule_iff {F : Ω → H} {β : SignedMeasure Ω} :
    β ∈ admissibleExposureSubmodule F ↔ Integrable F β.variation :=
  Iff.rfl

/--
An admissible exposure has finite integral of the norm of the factor field against its
variation measure.
-/
theorem admissibleExposure_hasFiniteIntegral {F : Ω → H}
    (β : admissibleExposureSubmodule F) : HasFiniteIntegral F β.1.variation :=
  β.2.hasFiniteIntegral

/--
The factor field of an admissible exposure is almost everywhere strongly measurable with
respect to the exposure's variation measure.
-/
theorem admissibleExposure_aestronglyMeasurable {F : Ω → H}
    (β : admissibleExposureSubmodule F) : AEStronglyMeasurable F β.1.variation :=
  β.2.aestronglyMeasurable

noncomputable section

/--
A coherently bundled admissible exposure.

This wrapper is equivalent to the subtype of `admissibleExposureSubmodule F`.  It supplies a
single coherent additive-group/module hierarchy for quotient constructions; its mathematical
elements and admissibility condition are unchanged.
-/
structure AdmissibleExposure (F : Ω → H) where
  val : admissibleExposureSubmodule F

/-- Forget the coherent wrapper and recover the admissible-submodule subtype. -/
def admissibleExposureEquiv (F : Ω → H) :
    AdmissibleExposure F ≃ admissibleExposureSubmodule F where
  toFun := AdmissibleExposure.val
  invFun β := ⟨β⟩
  left_inv β := by cases β; rfl
  right_inv _ := rfl

/-- Admissible exposures form an additive commutative group. -/
instance instAddCommGroupAdmissibleExposure (F : Ω → H) :
    AddCommGroup (AdmissibleExposure F) :=
  Equiv.addCommGroup (admissibleExposureEquiv F)

/-- Admissible exposures form a real module. -/
instance instModuleAdmissibleExposure (F : Ω → H) : Module ℝ (AdmissibleExposure F) :=
  Equiv.module ℝ (admissibleExposureEquiv F)

/-- The coherent admissible-exposure wrapper is linearly equivalent to the submodule subtype. -/
def admissibleExposureLinearEquiv (F : Ω → H) :
    AdmissibleExposure F ≃ₗ[ℝ] admissibleExposureSubmodule F :=
  Equiv.linearEquiv ℝ (admissibleExposureEquiv F)

/-- The underlying signed measure of a bundled admissible exposure is variation-integrable. -/
theorem AdmissibleExposure.integrable {F : Ω → H} (β : AdmissibleExposure F) :
    Integrable F β.val.1.variation :=
  β.val.2

end

end PricingPerspective.ContinuousAPT.CovarianceSpace
