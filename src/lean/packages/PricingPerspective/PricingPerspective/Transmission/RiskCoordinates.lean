import WassersteinGeometry.Contraction
import WassersteinGeometry.Hilbert.Polarization
import Mathlib.Analysis.Normed.Operator.NNNorm

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory WassersteinGeometry

/-!
# Risk coordinates and the factor-weighted transport distance

The loading law `Pᵢ` lives in a Hilbert space `H`, but systematic covariance is measured in
the factor-risk inner product, not the ambient one:

  `Cov(Sᵢ, Sⱼ) = 𝔼⟪Bᵢ, Γ Bⱼ⟫`.

Writing `A = Γ^{1/2}` and `Z = A B`, that becomes the *ambient* inner product `𝔼⟪Zᵢ, Zⱼ⟫`.
So risk coordinates are a pushforward, and the factor-weighted distance is

  `W_{2,Γ}(P, Q) = W₂(A_#P, A_#Q)`.

This file records that translation and its two consequences: the inner-product identity that
makes `WassersteinGeometry.Hilbert` applicable to a factor model at all, and the operator-norm
contraction `W_{2,Γ} ≤ ‖A‖ · W₂` that follows from `wassersteinDistance_map_le`.

## Self-adjointness as a hypothesis

`A` is taken self-adjoint through the explicit premise `∀ x y, ⟪A x, y⟫ = ⟪x, A y⟫` rather
than through `ContinuousLinearMap.adjoint`. The adjoint API needs `CompleteSpace`, which
nothing here otherwise requires, and the premise is the readable form of the assumption a
factor model actually makes about a covariance operator.

`A` being a genuine square root of a positive `Γ` is likewise never assumed: every result
below holds for an arbitrary bounded operator, with `Γ := A ∘ A` read off rather than
imposed. Positivity would be needed to interpret `Γ` as a covariance operator, not to prove
any statement here.

## References

* Panaretos & Zemel (2020), *An Invitation to Statistics in Wasserstein Space*, Ch. 2.
* `.context/chat/2026-08-16_random_functions/README.md`, "Core model".
-/

namespace PricingPerspective.Transmission

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-- Risk coordinates: the loading law pushed forward under `A = Γ^{1/2}`. -/
noncomputable def riskCoords (A : H →L[ℝ] H) (P : Measure H) : Measure H :=
  P.map A

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
@[simp] lemma riskCoords_apply (A : H →L[ℝ] H) (P : Measure H) :
    riskCoords A P = P.map A := rfl

/-! ### The inner product in risk coordinates is the Γ-weighted form -/

omit [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- **Risk coordinates realize the factor-weighted inner product.**

    For self-adjoint `A`, `⟪A x, A y⟫ = ⟪x, Γ y⟫` with `Γ = A ∘ A`. This is why the ambient
    Hilbert geometry of `WassersteinGeometry.Hilbert` is the right geometry for a factor
    model: once loadings are expressed in risk coordinates, no weighting remains to carry. -/
lemma inner_riskCoords (A : H →L[ℝ] H)
    (hA : ∀ x y : H, ⟪A x, y⟫_ℝ = ⟪x, A y⟫_ℝ) (x y : H) :
    ⟪A x, A y⟫_ℝ = ⟪x, (A.comp A) y⟫_ℝ :=
  hA x (A y)

omit [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The systematic second moment in risk coordinates is the `Γ`-weighted second moment. -/
lemma normSq_riskCoords (A : H →L[ℝ] H)
    (hA : ∀ x y : H, ⟪A x, y⟫_ℝ = ⟪x, A y⟫_ℝ) (x : H) :
    ‖A x‖ ^ 2 = ⟪x, (A.comp A) x⟫_ℝ := by
  rw [← real_inner_self_eq_norm_sq]
  exact inner_riskCoords A hA x x

/-! ### Factor-weighted transport contracts by the operator norm -/

/-- **`W_{2,Γ}(P, Q) ≤ ‖A‖ · W₂(P, Q)`.**

    Passing to risk coordinates cannot expand transport distance by more than the operator
    norm of `A = Γ^{1/2}`: a factor structure that loads weakly on some directions collapses
    loading differences along them. Directly `wassersteinDistance_map_le` for the Lipschitz
    constant `‖A‖₊` of a bounded operator.

    The risk-coordinate laws are supplied as `WassersteinMeasure`s with the pushforward
    identity as a hypothesis, so second-moment finiteness in risk coordinates is discharged
    once by the caller. -/
theorem wassersteinDistance_riskCoords_le
    (A : H →L[ℝ] H) (P Q : WassersteinMeasure H) (P' Q' : WassersteinMeasure H)
    (hP' : P'.measure = riskCoords A P.measure)
    (hQ' : Q'.measure = riskCoords A Q.measure) :
    WassersteinDistance P' Q' ≤ ‖A‖ * WassersteinDistance P Q := by
  have hmeas : Measurable (A : H → H) := A.continuous.measurable
  have hlip : LipschitzWith ‖A‖₊ (A : H → H) := A.lipschitz
  simpa using wassersteinDistance_map_le hlip hmeas P Q P' Q' hP' hQ'

/-- The squared form of `wassersteinDistance_riskCoords_le`. -/
theorem wassersteinDistanceSq_riskCoords_le
    (A : H →L[ℝ] H) (P Q : WassersteinMeasure H) (P' Q' : WassersteinMeasure H)
    (hP' : P'.measure = riskCoords A P.measure)
    (hQ' : Q'.measure = riskCoords A Q.measure) :
    WassersteinDistanceSq P' Q' ≤ (‖A‖₊ : ℝ≥0∞) ^ 2 * WassersteinDistanceSq P Q :=
  wassersteinDistanceSq_map_le A.lipschitz A.continuous.measurable P Q P' Q' hP' hQ'

end PricingPerspective.Transmission
