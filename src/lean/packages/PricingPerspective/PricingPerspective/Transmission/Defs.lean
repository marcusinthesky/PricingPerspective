import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

set_option linter.style.longLine false

open Finset
open scoped InnerProductSpace

/-!
# Random functional factor exposures — definitions

Formalizes the *empirical* layer of the random functional factor model

  `Bᵢ = T(ξᵢ) + δᵢ`,  `ξᵢ ∼ Cᵢ`,  `‖δᵢ‖ ≤ τᵢ`,

in which an asset's factor loading is a **random** element of a Hilbert space
rather than a fixed vector, and the observable primitive is the characteristic
distribution `Cᵢ` rather than the loading itself.

## Why finite index sets

Every object here is indexed by a `Fintype`. This is deliberate and is *not* an
approximation of a measure-theoretic development: it is the estimator. The
pipeline observes `m` article embeddings per firm and forms empirical measures
on those points, so a coupling is literally a nonnegative `m × m` weight matrix
with prescribed margins. Working at that level buys two things:

* the pushforward of a coupling under a map `T` acting on support points is the
  *same* weight matrix, so the transfer results in `Transfer.lean` need no
  measurable-inverse (Lusin–Souslin) argument; and
* every statement is directly checkable against the numbers the DVC graph emits.

The measure-theoretic `W₂` lives in `WassersteinGeometry`; nothing here depends
on it, so the finite-support theory remains an independently checkable layer.

## Main definitions

* `Coupling p q` — a joint pmf on `ι × ι'` with margins `p` and `q`.
* `systCov` — systematic covariance `𝔼_π ⟪Z, Y⟫` induced by a coupling.
* `transportCost` — expected squared distance `𝔼_π ‖Z − Y‖²` under a coupling.
* `secondMoment` — `v(P) = 𝔼 ‖Z‖²`, the systematic second moment.
* `barycentre` — the mean `μ_P = 𝔼 Z` of a weighted point cloud.

## Coordinates

Throughout, `Z` and `Y` are already in **risk coordinates**: `Z = Γ^{1/2} B`
for the factor covariance operator `Γ`. That pushforward is definitional, not an
assumption — it is forced by `Cov(Sᵢ, Sⱼ) = 𝔼⟪Bᵢ, Γ Bⱼ⟫` — so it is discharged
by the caller and does not appear as a hypothesis below.

## References

* Gelbrich (1990), *On a formula for the L² Wasserstein metric*.
* Panaretos & Zemel (2020), *An Invitation to Statistics in Wasserstein Space*, Ch. 2.
* Puccetti & Wang (2015), extremal dependence and quadratic transport.
-/

namespace PricingPerspective.RandomExposure

variable {ι ι' : Type*} [Fintype ι] [Fintype ι']
variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- A finite probability cloud: weighted support with unit total mass. -/
structure ProbabilityCloud (ι X : Type*) [Fintype ι] where
  /-- Probability weight at each support point. -/
  weight : ι → ℝ
  /-- Support point at each index. -/
  point : ι → X
  /-- Weights are nonnegative. -/
  nonneg : ∀ i, 0 ≤ weight i
  /-- The cloud has unit total mass. -/
  mass_one : ∑ i, weight i = 1

/-- An empirical random exposure cloud with observable characteristics. -/
structure RandomExposureCloud
    (ι Ω H : Type*) [Fintype ι] [NormedAddCommGroup H] where
  /-- Observable characteristic cloud. -/
  characteristic : ProbabilityCloud ι Ω
  /-- Support-indexed loading slack. -/
  slack : ι → H
  /-- Uniform slack radius. -/
  radius : ℝ
  /-- The radius is nonnegative. -/
  radius_nonneg : 0 ≤ radius
  /-- Every realized loading stays within the slack budget. -/
  slack_bound : ∀ i, ‖slack i‖ ≤ radius

namespace RandomExposureCloud

variable {ι Ω H : Type*} [Fintype ι] [NormedAddCommGroup H]

/-- Realized loading `Bᵢ = T(ξᵢ) + δᵢ` under the supplied common map. -/
def loading (T : Ω → H) (C : RandomExposureCloud ι Ω H) (i : ι) : H :=
  T (C.characteristic.point i) + C.slack i

/-- The latent loading cloud induced by a random exposure cloud and common map. -/
def exposure (T : Ω → H) (C : RandomExposureCloud ι Ω H) : ProbabilityCloud ι H :=
  { weight := C.characteristic.weight
    point := C.loading T
    nonneg := C.characteristic.nonneg
    mass_one := C.characteristic.mass_one }

@[simp] theorem exposure_weight (T : Ω → H) (C : RandomExposureCloud ι Ω H) :
    (C.exposure T).weight = C.characteristic.weight := rfl

@[simp] theorem exposure_point (T : Ω → H) (C : RandomExposureCloud ι Ω H) :
    (C.exposure T).point = C.loading T := rfl

end RandomExposureCloud

/-- A finite coupling (transference plan) of two weighted point clouds.

    `w a b` is the joint mass placed on the pair `(a, b)`. The margin conditions
    are the defining constraints of `Π(p, q)` in the optimal-transport
    literature; nonnegativity makes `w` a joint pmf whenever `p` and `q` are.

    No normalization `∑ p = 1` is imposed: the results below are homogeneous in
    the total mass, and leaving it free lets the same structure carry
    subprobability and unnormalized empirical clouds. -/
structure Coupling (p : ι → ℝ) (q : ι' → ℝ) where
  /-- Joint mass on each pair of support points. -/
  w : ι → ι' → ℝ
  /-- Masses are nonnegative. -/
  nonneg : ∀ a b, 0 ≤ w a b
  /-- First margin is `p`. -/
  marginal_fst : ∀ a, ∑ b, w a b = p a
  /-- Second margin is `q`. -/
  marginal_snd : ∀ b, ∑ a, w a b = q b

namespace Coupling

variable {p : ι → ℝ} {q : ι' → ℝ}

/-- Total mass is the same read along either margin. -/
lemma total_mass (π : Coupling p q) : ∑ a, p a = ∑ b, q b := by
  have h₁ : ∑ a, p a = ∑ a, ∑ b, π.w a b :=
    Finset.sum_congr rfl fun a _ => (π.marginal_fst a).symm
  have h₂ : ∑ b, q b = ∑ b, ∑ a, π.w a b :=
    Finset.sum_congr rfl fun b _ => (π.marginal_snd b).symm
  rw [h₁, h₂, Finset.sum_comm]

end Coupling

/-- A finite certificate for the minimum of an arbitrary coupling cost. -/
structure FiniteOptimalCostCertificate
    {p : ι → ℝ} {q : ι' → ℝ} (cost : Coupling p q → ℝ) where
  /-- A coupling attaining the certified value. -/
  plan : Coupling p q
  /-- Certified optimal value. -/
  value : ℝ
  /-- The supplied plan attains the value. -/
  plan_cost : cost plan = value
  /-- The value is no larger than every admissible cost. -/
  minimal : ∀ π : Coupling p q, value ≤ cost π


/-- Systematic covariance induced by a coupling, `κ_π = 𝔼_π ⟪Z, Y⟫`.

    Under `Assumption 2.1` of the manuscript (factor innovation `F` mean zero,
    independent of the loadings, with covariance operator `Γ`), this *is*
    `Cov(Sᵢ, Sⱼ)` once `Z`, `Y` are in risk coordinates. Marginals alone do not
    pin it down; the coupling does. That is the identification problem the
    envelope results address. -/
noncomputable def systCov {p : ι → ℝ} {q : ι' → ℝ}
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) : ℝ :=
  ∑ a, ∑ b, π.w a b * ⟪Z a, Y b⟫_ℝ

/-- Expected squared transport cost of a coupling, `𝔼_π ‖Z − Y‖²`.

    Minimizing this over `Π(p, q)` is the (squared) quadratic Wasserstein
    problem; `Envelope.lean` shows minimizing it is *the same problem* as
    maximizing `systCov`. -/
noncomputable def transportCost {p : ι → ℝ} {q : ι' → ℝ}
    (π : Coupling p q) (Z : ι → H) (Y : ι' → H) : ℝ :=
  ∑ a, ∑ b, π.w a b * ‖Z a - Y b‖ ^ 2

/-- Systematic second moment `v(P) = 𝔼 ‖Z‖²` of a weighted cloud.

    Under `Assumption 2.1` this is the systematic return variance of the asset;
    it is a property of the marginal law alone, hence identified without any
    coupling. -/
noncomputable def secondMoment (p : ι → ℝ) (Z : ι → H) : ℝ :=
  ∑ a, p a * ‖Z a‖ ^ 2

/-- Barycentre (mean exposure) `μ_P = 𝔼 Z` of a weighted cloud.

    For an empirical article cloud on the unit sphere this is the (unnormalized)
    centroid whose chord distance the pipeline already reports. -/
noncomputable def barycentre (p : ι → ℝ) (Z : ι → H) : H :=
  ∑ a, p a • Z a

end PricingPerspective.RandomExposure
