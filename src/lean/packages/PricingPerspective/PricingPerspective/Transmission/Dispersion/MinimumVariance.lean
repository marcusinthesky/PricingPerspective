import PricingPerspective.Transmission.Dispersion.PortfolioVariance
import Mathlib.Analysis.Convex.StdSimplex

set_option linter.style.longLine false

open Finset Set

/-!
# Minimum certified portfolio variance

The optimization target is the computable raw-return certificate.  Compactness,
not an unproved matrix-positivity assertion, supplies existence.  The result is
therefore a minimum of the certificate with a bound on actual variance; it is
not labelled a true global-minimum-variance portfolio.
-/

namespace PricingPerspective.Transmission.Dispersion

variable {A : Type*} [Fintype A]

/-- Raw certified variance for positions `x`, marginal scales `σ`, and
pairwise exposure floors `d`. -/
noncomputable def certifiedVarianceObjective
    (σ : A → ℝ) (d : A → A → ℝ) (x : A → ℝ) : ℝ :=
  (∑ a, x a * σ a) ^ 2 -
    (1 / 2 : ℝ) * ∑ a, ∑ b, x a * x b * σ a * σ b * (d a b) ^ 2

/-- The raw certified-variance objective is continuous in portfolio weights. -/
theorem continuous_certifiedVarianceObjective (σ : A → ℝ) (d : A → A → ℝ) :
    Continuous (certifiedVarianceObjective σ d) := by
  unfold certifiedVarianceObjective
  fun_prop

/-- Conditional negative definiteness is the exact curvature condition used
for squared pairwise-distance certificates. -/
def IsConditionallyNegativeDefinite (d : A → A → ℝ) : Prop :=
  ∀ u : A → ℝ, (∑ a, u a) = 0 →
    ∑ a, ∑ b, u a * u b * (d a b) ^ 2 ≤ 0

/-- Under conditional negative definiteness, one minus the pairwise
certificate is convex along mixtures of probability weights.  No unconditional
convexity claim is made for an arbitrary distance matrix. -/
theorem standardizedCertifiedObjective_convex_of_cnd
    (d : A → A → ℝ) (hCND : IsConditionallyNegativeDefinite d)
    (q r : WassersteinGeometry.Multimarginal.ProbabilityWeight A)
    (θ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    1 - pairwiseRiskCertificate (q.mix r θ hθ0 hθ1) d ≤
      θ * (1 - pairwiseRiskCertificate q d) +
        (1 - θ) * (1 - pairwiseRiskCertificate r d) := by
  let Q : (A → ℝ) → ℝ := fun w ↦
    ∑ a, ∑ b, w a * w b * (d a b) ^ 2
  have hzero : (∑ a, (q a - r a)) = 0 := by
    rw [Finset.sum_sub_distrib, q.mass_one, r.mass_one]
    ring
  have hcnd := hCND (fun a ↦ q a - r a) hzero
  have hθ : 0 ≤ θ * (1 - θ) := mul_nonneg hθ0 (sub_nonneg.mpr hθ1)
  have hmix : θ * Q q + (1 - θ) * Q r ≤
      Q (fun a ↦ θ * q a + (1 - θ) * r a) := by
    have hkey : ∀ a b,
        (θ * q a + (1 - θ) * r a) *
            (θ * q b + (1 - θ) * r b) * (d a b) ^ 2 =
          θ * (q a * q b * (d a b) ^ 2) +
            (1 - θ) * (r a * r b * (d a b) ^ 2) -
              θ * (1 - θ) *
                ((q a - r a) * (q b - r b) * (d a b) ^ 2) := by
      intro a b
      ring
    unfold Q
    simp_rw [hkey, Finset.sum_sub_distrib, Finset.sum_add_distrib,
      ← Finset.mul_sum]
    nlinarith
  unfold Q at hmix
  unfold pairwiseRiskCertificate
  change 1 - (1 / 2 : ℝ) *
      (∑ a, ∑ b,
        (θ * q a + (1 - θ) * r a) *
          (θ * q b + (1 - θ) * r b) * (d a b) ^ 2) ≤ _
  nlinarith

/-- **Existence and risk guarantee for minimum certified variance.**

Any nonempty compact feasible subset of the long-only simplex has a minimizer
of the certified objective.  If actual variance is pointwise bounded by that
objective, it is bounded at the selected minimizer as well.
-/
theorem exists_minimumCertifiedVariance
    (σ : A → ℝ) (d : A → A → ℝ) (X : Set (A → ℝ))
    (hcompact : IsCompact X) (hnonempty : X.Nonempty)
    (_hlongOnly : X ⊆ stdSimplex ℝ A)
    (actualVariance : (A → ℝ) → ℝ)
    (hcertificate : ∀ x ∈ X, actualVariance x ≤ certifiedVarianceObjective σ d x) :
    ∃ xStar ∈ X,
      IsMinOn (certifiedVarianceObjective σ d) X xStar ∧
        actualVariance xStar ≤ certifiedVarianceObjective σ d xStar := by
  obtain ⟨xStar, hx, hmin⟩ := hcompact.exists_isMinOn hnonempty
    (continuous_certifiedVarianceObjective σ d).continuousOn
  exact ⟨xStar, hx, hmin, hcertificate xStar hx⟩

/-- The selected portfolio's actual variance is no larger than every feasible
portfolio's certified variance. -/
theorem minimumCertifiedVariance_guarantee
    (σ : A → ℝ) (d : A → A → ℝ) (X : Set (A → ℝ))
    (actualVariance : (A → ℝ) → ℝ) (xStar : A → ℝ)
    (hx : xStar ∈ X)
    (hmin : IsMinOn (certifiedVarianceObjective σ d) X xStar)
    (hcertificate : ∀ x ∈ X, actualVariance x ≤ certifiedVarianceObjective σ d x) :
    ∀ x ∈ X, actualVariance xStar ≤ certifiedVarianceObjective σ d x := by
  intro x hxX
  exact (hcertificate xStar hx).trans (hmin hxX)

end PricingPerspective.Transmission.Dispersion
