import WassersteinGeometry.Geodesics
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic

set_option linter.style.longLine false

open scoped ENNReal NNReal MeasureTheory InnerProductSpace
open MeasureTheory

/-!
# Quadratic functionals of measures on a Hilbert space

`Geodesics` develops `W₂` over a bare `PseudoMetricSpace`, where the cost is an
`ℝ≥0∞`-valued lower Lebesgue integral. Covariance is signed, so the covariance envelope
cannot be read off that development directly. This file supplies the real-valued Bochner
layer the envelope needs, for measures on a real Hilbert space `H`.

## The three functionals

For laws `P, Q` of random loadings **already in risk coordinates** (`Z = Γ^{1/2}B`, so the
Hilbert inner product *is* the factor-risk covariance form):

* `secondMoment P = 𝔼‖Z‖²` — systematic second moment, a property of the marginal alone;
* `systCov π = 𝔼_π⟪Z, Y⟫` — systematic covariance, which depends on the *coupling* `π` and
  is exactly the quantity marginals fail to identify;
* `transportCost π = 𝔼_π‖Z − Y‖²` — the real-valued transport cost whose infimum over
  couplings is `W₂²`.

## Integrability is a hypothesis, not a side condition

Every result below takes the integrability it needs as an explicit premise rather than
deriving it from `second_moment_finite`. That is deliberate and matches the certificate
discipline `OptimalCoupling` already follows: a caller working over a concrete space
discharges it once, and no theorem here silently assumes a Bochner integral converges.
`transportCost_integrable_of` records the one direction that is genuinely automatic — the
cost is integrable as soon as the three constituent pieces are.

## References

* Panaretos & Zemel (2020), *An Invitation to Statistics in Wasserstein Space*, Ch. 2.
* Gelbrich (1990), *On a formula for the L² Wasserstein metric*.
-/

namespace WassersteinGeometry.Hilbert

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [MeasurableSpace H] [BorelSpace H] [SecondCountableTopology H] [Inhabited H]

/-- Systematic second moment `v(P) = 𝔼‖Z‖²` of a loading law in risk coordinates.

    Identified by the marginal alone: no coupling enters. -/
noncomputable def secondMoment (P : WassersteinMeasure H) : ℝ :=
  ∫ x, ‖x‖ ^ 2 ∂P.measure

/-- Systematic covariance `κ_π = 𝔼_π⟪Z, Y⟫` induced by a coupling.

    In risk coordinates this *is* `Cov(Sᵢ, Sⱼ)`. Two marginals do not pin it down; the
    coupling does. That gap is the identification problem the envelope addresses. -/
noncomputable def systCov (π : Measure (H × H)) : ℝ :=
  ∫ p, ⟪p.1, p.2⟫_ℝ ∂π

/-- Real-valued quadratic transport cost `𝔼_π‖Z − Y‖²` of a coupling.

    Minimizing this over `Π(P, Q)` is the quadratic Wasserstein problem; polarization turns
    that minimization into a *maximization* of `systCov`. -/
noncomputable def transportCost (π : Measure (H × H)) : ℝ :=
  ∫ p, ‖p.1 - p.2‖ ^ 2 ∂π

/-! ### Measurability of the integrands -/

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] [Inhabited H] in
lemma aestronglyMeasurable_normSq (μ : Measure H) :
    AEStronglyMeasurable (fun x : H => ‖x‖ ^ 2) μ :=
  (continuous_norm.pow 2).aestronglyMeasurable

omit [InnerProductSpace ℝ H] [Inhabited H] in
lemma aestronglyMeasurable_fst_normSq (π : Measure (H × H)) :
    AEStronglyMeasurable (fun p : H × H => ‖p.1‖ ^ 2) π :=
  ((continuous_norm.pow 2).comp continuous_fst).aestronglyMeasurable

omit [InnerProductSpace ℝ H] [Inhabited H] in
lemma aestronglyMeasurable_snd_normSq (π : Measure (H × H)) :
    AEStronglyMeasurable (fun p : H × H => ‖p.2‖ ^ 2) π :=
  ((continuous_norm.pow 2).comp continuous_snd).aestronglyMeasurable

/-! ### Transferring a marginal integral to the coupling -/

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- The first-marginal second moment computed on the coupling.

    `∫ ‖x‖² dP = ∫ ‖p.1‖² dπ` whenever `π` has first marginal `P`. This is what lets
    polarization, which lives on `H × H`, produce a statement about the marginals. -/
lemma secondMoment_fst_of_mem_couplingSet
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure) :
    secondMoment P = ∫ p, ‖p.1‖ ^ 2 ∂π := by
  unfold secondMoment
  rw [← hπ.1, integral_map measurable_fst.aemeasurable]
  exact aestronglyMeasurable_normSq _

omit [InnerProductSpace ℝ H] [SecondCountableTopology H] in
/-- The second-marginal second moment computed on the coupling. -/
lemma secondMoment_snd_of_mem_couplingSet
    {P Q : WassersteinMeasure H} {π : Measure (H × H)}
    (hπ : π ∈ couplingSet P.measure Q.measure) :
    secondMoment Q = ∫ p, ‖p.2‖ ^ 2 ∂π := by
  unfold secondMoment
  rw [← hπ.2, integral_map measurable_snd.aemeasurable]
  exact aestronglyMeasurable_normSq _

/-! ### Integrability bookkeeping -/

omit [BorelSpace H] [SecondCountableTopology H] [Inhabited H] in
/-- The transport cost is integrable once its three polarization constituents are. -/
lemma transportCost_integrable_of
    {π : Measure (H × H)}
    (h₁ : Integrable (fun p : H × H => ‖p.1‖ ^ 2) π)
    (h₂ : Integrable (fun p : H × H => ‖p.2‖ ^ 2) π)
    (hc : Integrable (fun p : H × H => ⟪p.1, p.2⟫_ℝ) π) :
    Integrable (fun p : H × H => ‖p.1 - p.2‖ ^ 2) π := by
  have hrw : (fun p : H × H => ‖p.1 - p.2‖ ^ 2) =
      fun p : H × H => ‖p.1‖ ^ 2 - 2 * ⟪p.1, p.2⟫_ℝ + ‖p.2‖ ^ 2 := by
    funext p
    exact norm_sub_sq_real p.1 p.2
  rw [hrw]
  exact (h₁.sub (hc.const_mul 2)).add h₂

end WassersteinGeometry.Hilbert
