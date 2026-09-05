import PricingPerspective.Transmission.Dispersion.MinimumVariance

set_option linter.style.longLine false

open scoped ENNReal NNReal InnerProductSpace MeasureTheory
open MeasureTheory Set WassersteinGeometry WassersteinGeometry.Multimarginal

/-!
# Paper 3 information-certified portfolio variance

This module is the Paper 3 application layer.  It composes the pairwise carrier-and-slack
floor with the coherent joint-law portfolio-variance cap, then specializes the existing
compact-feasible-set optimizer to that observable floor.  The general barycenter and MMOT
results remain outside this module.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A X H : Type*} [Fintype A]
  [MeasurableSpace X] [PseudoMetricSpace X] [Inhabited X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]

/-- The Paper 3 pairwise certificate constructed from characteristic-law distance, a common
carrier constant, and asset-specific synchronous slack radii.

For assets `a` and `b`, the certificate is
`max 0 (L⁻¹ W₂(C a,C b) - τ a - τ b)`.  It is observable once `L` and the slack
radii are fixed; the carrier theorem below supplies its exposure-law interpretation.
-/
noncomputable def informationPairwiseFloor
    (L : ℝ≥0) (τ : A → ℝ≥0) (C : A → WassersteinMeasure X) (a b : A) : ℝ :=
  pairwiseExposureFloor (L : ℝ) (τ a : ℝ) (τ b : ℝ) (C a) (C b)

section JointLaw

variable [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H]
  [CompleteSpace H] [Inhabited H]

/-- **Paper 3 information-certified coherent portfolio-variance bound.**

For a finite asset universe, each characteristic law `C a` is pushed through a common
measurable `L`-antilipschitz carrier `t`, while the realized exposure map `u a` stays within
pointwise synchronous slack `τ a`.  If the resulting exposure marginals are coordinates of
one joint law `J` and all cross-inner-products are integrable, then systematic portfolio
variance is at most the weighted marginal second moments minus the observable pairwise
certificate.

The proof composes `pairwiseExposureFloor_le_of_antilipschitz` with
`jointPortfolioVariance_le_pairwiseFloorCap`.  It is the pairwise-computable Paper 3
application and assumes none of the barycenter or aggregate-dispersion structure used by
Paper 5.
-/
theorem jointPortfolioVariance_le_informationCertifiedCap
    {L : ℝ≥0} {t : X → H}
    (hanti : AntilipschitzWith L t) (hme : MeasurableEmbedding t)
    (C : A → WassersteinMeasure X) (u : A → X → H)
    (hu : ∀ a, Measurable (u a)) (τ : A → ℝ≥0)
    (hslack : ∀ a x, edist (u a x) (t x) ≤ (τ a : ℝ≥0∞))
    (P T : A → WassersteinMeasure H)
    (hP : ∀ a, (P a).measure = (C a).measure.map (u a))
    (hT : ∀ a, (T a).measure = (C a).measure.map t)
    (J : Measure (A → H))
    (hPjoint : ∀ a, (P a).measure = J.map fun x : A → H ↦ x a)
    (q : ProbabilityWeight A)
    (hint : ∀ a b : A, Integrable (fun x : A → H ↦ ⟪x a, x b⟫_ℝ) J) :
    PricingPerspective.Transmission.portfolioVariance J q ≤
      (∑ a, q a * WassersteinGeometry.Hilbert.secondMoment (P a)) -
        pairwiseRiskCertificate q (informationPairwiseFloor L τ C) := by
  apply jointPortfolioVariance_le_pairwiseFloorCap J P hPjoint q
      (informationPairwiseFloor L τ C)
  · intro a b
    unfold informationPairwiseFloor pairwiseExposureFloor
    exact le_max_left _ _
  · intro a b
    unfold informationPairwiseFloor
    exact pairwiseExposureFloor_le_of_antilipschitz
      hanti hme (hu a) (hu b) (hslack a) (hslack b)
      (C a) (C b) (P a) (P b) (T a) (T b)
      (hP a) (hP b) (hT a) (hT b)
  · exact hint

/-- **Paper 3 residual-budget wrapper.**

If systematic standardized variance is bounded by one minus the information certificate
and the aggregate residual contribution is at most the assumed nonnegative budget `δ`,
then total standardized variance is bounded by `1 - certificate + δ`.  The premise treats
`δ` as a sensitivity budget; the theorem neither estimates it nor derives residual
orthogonality.
-/
theorem totalVariance_le_informationCertificate_add_residualBudget
    (systematic residual total certificate δ : ℝ)
    (htotal : total = systematic + residual)
    (hsystematic : systematic ≤ 1 - certificate)
    (hresidual : residual ≤ δ) :
    total ≤ 1 - certificate + δ := by
  linarith

/-- Raw-scale corollary of the standardized residual-budget wrapper for a nonnegative
perfect-dependence volatility scale `A`. -/
theorem rawVariance_le_informationCertificate_add_residualBudget
    (standardized raw certificate δ A : ℝ)
    (hraw : raw = A ^ 2 * standardized)
    (hstandardized : standardized ≤ 1 - certificate + δ) :
    raw ≤ A ^ 2 * (1 - certificate + δ) := by
  rw [hraw]
  exact mul_le_mul_of_nonneg_left hstandardized (sq_nonneg A)

end JointLaw

omit [OpensMeasurableSpace X] [SecondCountableTopology X] in
/-- **Paper 3 optimizer application for the characteristic-induced certificate.**

Any nonempty compact feasible subset of the long-only simplex admits a minimizer of the
certified raw-variance objective formed with `informationPairwiseFloor`.  If actual variance
is pointwise bounded by that objective on the feasible set, the selected portfolio retains
the bound.  This is an existence and risk-guarantee result: it asserts neither uniqueness nor
unconditional convexity of the certificate.
-/
theorem exists_minimumInformationCertifiedVariance
    (L : ℝ≥0) (τ : A → ℝ≥0) (C : A → WassersteinMeasure X)
    (σ : A → ℝ) (S : Set (A → ℝ))
    (hcompact : IsCompact S) (hnonempty : S.Nonempty)
    (hlongOnly : S ⊆ stdSimplex ℝ A)
    (actualVariance : (A → ℝ) → ℝ)
    (hcertificate : ∀ x ∈ S,
      actualVariance x ≤
        certifiedVarianceObjective σ (informationPairwiseFloor L τ C) x) :
    ∃ xStar ∈ S,
      IsMinOn (certifiedVarianceObjective σ (informationPairwiseFloor L τ C)) S xStar ∧
        actualVariance xStar ≤
          certifiedVarianceObjective σ (informationPairwiseFloor L τ C) xStar := by
  exact exists_minimumCertifiedVariance
    σ (informationPairwiseFloor L τ C) S hcompact hnonempty hlongOnly
    actualVariance hcertificate

end PricingPerspective.Transmission.Dispersion
