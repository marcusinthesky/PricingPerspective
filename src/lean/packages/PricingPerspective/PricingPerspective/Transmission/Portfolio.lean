import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.PosDef
import PricingPerspective.Transmission.Defs

set_option linter.style.longLine false

open Finset
open scoped InnerProductSpace

/-!
# Coherent multi-asset covariance from one joint exposure law

A single finite joint law produces a covariance matrix by averaging Gram
matrices. This is the finite coherence condition missing from independently
optimized pairwise couplings: positive semidefiniteness is structural rather
than repaired after the fact.
-/

namespace PricingPerspective.RandomExposure

variable {A ι H : Type*} [Fintype A] [Fintype ι]
variable [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- A finite joint law for all asset loadings. -/
structure JointExposureLaw (A ι H : Type*) [Fintype A] [Fintype ι]
    [NormedAddCommGroup H] [InnerProductSpace ℝ H] where
  /-- Joint probability weight on latent states. -/
  weight : ι → ℝ
  /-- Loading of every asset at every joint state. -/
  exposure : A → ι → H
  /-- Joint weights are nonnegative. -/
  nonneg : ∀ s, 0 ≤ weight s
  /-- The joint law has unit mass. -/
  mass_one : ∑ s, weight s = 1


/-- Covariance matrix induced by one globally coherent joint law. -/
def jointCovariance (J : JointExposureLaw A ι H) : Matrix A A ℝ :=
  fun i j => ∑ s, J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ
/-- The marginal exposure cloud for one asset under the shared joint law. -/
def JointExposureLaw.marginal
    (J : JointExposureLaw A ι H) (i : A) : ProbabilityCloud ι H :=
  { weight := J.weight
    point := J.exposure i
    nonneg := J.nonneg
    mass_one := J.mass_one }

/-- The shared-state diagonal coupling between two joint-law marginals. -/
noncomputable def JointExposureLaw.diagonalCoupling
    (J : JointExposureLaw A ι H) (_i _j : A) :
    Coupling J.weight J.weight := by
  letI : DecidableEq ι := Classical.decEq ι
  exact
    { w := fun s t => if s = t then J.weight s else 0
      nonneg := by
        intro s t
        split_ifs
        · exact J.nonneg s
        · exact le_refl 0
      marginal_fst := by
        intro s
        simp
      marginal_snd := by
        intro t
        simp [eq_comm] }

/-- The covariance induced by one joint law equals the covariance of its
    shared-state diagonal coupling. -/
theorem jointCovariance_eq_systCov
    (J : JointExposureLaw A ι H) (i j : A) :
    jointCovariance J i j =
      systCov (J.diagonalCoupling i j)
        (fun s => J.exposure i s)
        (fun t => J.exposure j t) := by
  classical
  simp [jointCovariance, systCov, JointExposureLaw.diagonalCoupling]

/-- Joint-law portfolio loading at one latent state. -/
def jointPortfolioLoading (J : JointExposureLaw A ι H) (x : A → ℝ) (s : ι) : H :=
  ∑ i, x i • J.exposure i s

/-- The covariance quadratic form of one shared joint exposure law equals the
    averaged squared portfolio norm. This is a coherence result for that joint
    law, not a claim that independently optimal pairwise couplings are jointly
    compatible. -/
theorem jointCovariance_quadratic_eq
    (J : JointExposureLaw A ι H) (x : A → ℝ) :
    x ⬝ᵥ ((jointCovariance J).mulVec x) =
      ∑ s, J.weight s * ‖jointPortfolioLoading J x s‖ ^ 2 := by
  unfold dotProduct Matrix.mulVec jointCovariance jointPortfolioLoading
  calc
    (∑ i, x i * ∑ j, (∑ s, J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ) * x j) =
        ∑ i, ∑ j, x i * x j *
          ∑ s, J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      ring
    _ = ∑ i, ∑ j, ∑ s,
          x i * x j * (J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [Finset.mul_sum]
    _ = ∑ i, ∑ s, ∑ j,
          x i * x j * (J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Finset.sum_comm]
    _ = ∑ s, ∑ i, ∑ j,
          x i * x j * (J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ) := by
      rw [Finset.sum_comm]
    _ = ∑ s, J.weight s *
          (∑ i, ∑ j, x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ) := by
      refine Finset.sum_congr rfl fun s _ => ?_
      calc
        (∑ i, ∑ j,
            x i * x j * (J.weight s * ⟪J.exposure i s, J.exposure j s⟫_ℝ)) =
            ∑ i, ∑ j,
              J.weight s * (x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ) := by
          apply Finset.sum_congr rfl
          intro i hi
          apply Finset.sum_congr rfl
          intro j hj
          ring
        _ = ∑ i, J.weight s *
              ∑ j, x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ := by
          refine Finset.sum_congr rfl fun i _ => ?_
          exact (Finset.mul_sum (Finset.univ : Finset A)
            (fun j => x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ)
            (J.weight s)).symm
        _ = J.weight s *
              (∑ i, ∑ j, x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ) :=
          (Finset.mul_sum (Finset.univ : Finset A)
            (fun i => ∑ j, x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ)
            (J.weight s)).symm
    _ = ∑ s, J.weight s * ‖∑ i, x i • J.exposure i s‖ ^ 2 := by
      refine Finset.sum_congr rfl fun s _ => ?_
      have hnorm :
          ‖∑ i, x i • J.exposure i s‖ ^ 2 =
            ∑ i, ∑ j, x i * x j * ⟪J.exposure i s, J.exposure j s⟫_ℝ := by
        rw [← real_inner_self_eq_norm_sq, inner_sum]
        simp_rw [sum_inner, real_inner_smul_left, real_inner_smul_right]
        rw [Finset.sum_comm]
        ring
      rw [hnorm]

/-- A covariance matrix induced by one shared joint exposure law is positive
    semidefinite. This proved implication does not assert compatibility of
    independently optimized pairwise couplings. -/
theorem jointCovariance_posSemidef (J : JointExposureLaw A ι H) :
    (jointCovariance J).PosSemidef := by
  apply Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
  · ext i j
    simp only [Matrix.conjTranspose_apply, jointCovariance, star_trivial]
    rw [Finset.sum_congr rfl fun s _ => by rw [real_inner_comm]]
  · intro x
    change 0 ≤ x ⬝ᵥ ((jointCovariance J).mulVec x)
    rw [jointCovariance_quadratic_eq]
    exact Finset.sum_nonneg fun s _ =>
      mul_nonneg (J.nonneg s) (sq_nonneg _)

end PricingPerspective.RandomExposure
