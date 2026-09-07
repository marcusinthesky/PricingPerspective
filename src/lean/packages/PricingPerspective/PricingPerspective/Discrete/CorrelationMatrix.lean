import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.Analysis.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.Order.Star.Real

/-!
# Discrete correlation-matrix geometry (Paper 3, C3/C4)

Two matrix-level results used by Paper 3's correlation/covariance-matrix
repair machinery:

* **C3 (chord metric).** For a real PSD unit-diagonal correlation matrix `R`,
  `D i j := √(2 * (1 - R i j))` is a genuine metric (zero-diagonal, symmetric,
  triangle inequality). We take the Gram-generating unit vectors as the
  primitive (`v : Fin n → E`, `‖v i‖ = 1`, `R i j := ⟪v i, v j⟫`) rather than
  factoring an abstract PSD matrix, since Mathlib's PSD-factorization API
  (`Matrix.posSemidef_iff_eq_sum_vecMulVec`) gives a sum-of-rank-one
  decomposition rather than a single Gram map — the vector-primitive form is
  the flagged acceptable equivalent from the plan.
* **C4 (eigenvalue-clip PSD repair).** For a real symmetric (Hermitian) matrix
  `A`, the spectral clip that recomposes `A` with eigenvalues replaced by
  `max λ 0` is symmetric PSD, and equals `A` when `A` is already PSD. Frobenius
  optimality and Higham's diagonal-preservation property stay cited only
  (Paper 3 `monte_carlo.typ:65-67`), not formalized here.
-/

namespace PricingPerspective.Discrete.CorrelationMatrix

section ChordMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

open scoped InnerProductSpace

/-- The chord distance between two unit vectors' Gram entry equals the norm of
    their difference: `√(2 * (1 - ⟪v i, v j⟫)) = ‖v i - v j‖`. This is the
    bridge lemma that reduces the chord-metric triangle inequality to the norm
    triangle inequality. -/
theorem chord_dist_eq_norm_sub {n : ℕ} (v : Fin n → E) (hv : ∀ i, ‖v i‖ = 1)
    (i j : Fin n) :
    Real.sqrt (2 * (1 - ⟪v i, v j⟫_ℝ)) = ‖v i - v j‖ := by
  have hvi2 : ‖v i‖ ^ 2 = 1 := by rw [hv i]; norm_num
  have hvj2 : ‖v j‖ ^ 2 = 1 := by rw [hv j]; norm_num
  have hsq : ‖v i - v j‖ ^ 2 = 2 * (1 - ⟪v i, v j⟫_ℝ) := by
    have h := norm_sub_sq_real (v i) (v j)
    rw [hvi2, hvj2] at h
    linarith
  rw [← hsq]
  exact Real.sqrt_sq (norm_nonneg _)

/-- **C3 (zero diagonal).** The chord distance of a unit vector to itself is
    zero. -/
theorem chord_dist_self_eq_zero {n : ℕ} (v : Fin n → E) (hv : ∀ i, ‖v i‖ = 1)
    (i : Fin n) :
    Real.sqrt (2 * (1 - ⟪v i, v i⟫_ℝ)) = 0 := by
  have h : ⟪v i, v i⟫_ℝ = 1 := by rw [real_inner_self_eq_norm_sq, hv i]; norm_num
  rw [h]
  simp

/-- **C3 (symmetry).** The chord distance is symmetric, from `real_inner_comm`. -/
theorem chord_dist_symm {n : ℕ} (v : Fin n → E) (i j : Fin n) :
    Real.sqrt (2 * (1 - ⟪v i, v j⟫_ℝ)) = Real.sqrt (2 * (1 - ⟪v j, v i⟫_ℝ)) := by
  rw [real_inner_comm]

/-- **C3 (triangle inequality).** The chord distance `D i j := √(2(1 - R i
    j))` of a real PSD unit-diagonal Gram matrix `R i j := ⟪v i, v j⟫`
    satisfies the triangle inequality, via `chord_dist_eq_norm_sub` and the
    norm triangle inequality `‖v i - v k‖ ≤ ‖v i - v j‖ + ‖v j - v k‖`. -/
theorem chord_dist_triangle {n : ℕ} (v : Fin n → E) (hv : ∀ i, ‖v i‖ = 1)
    (i j k : Fin n) :
    Real.sqrt (2 * (1 - ⟪v i, v k⟫_ℝ)) ≤
      Real.sqrt (2 * (1 - ⟪v i, v j⟫_ℝ)) + Real.sqrt (2 * (1 - ⟪v j, v k⟫_ℝ)) := by
  rw [chord_dist_eq_norm_sub v hv i k, chord_dist_eq_norm_sub v hv i j,
    chord_dist_eq_norm_sub v hv j k]
  have heq : v i - v k = (v i - v j) + (v j - v k) := by rw [sub_add_sub_cancel]
  rw [heq]
  exact norm_add_le _ _

end ChordMetric

section EigenvalueClip

open scoped Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- **C4.** The spectral clip of a real symmetric (Hermitian) matrix `A`:
    recompose `A` in its own eigenbasis with eigenvalues replaced by
    `max λ 0`. -/
noncomputable def spectralClip {A : Matrix n n ℝ} (hA : A.IsHermitian) :
    Matrix n n ℝ :=
  (hA.eigenvectorUnitary : Matrix n n ℝ) *
    Matrix.diagonal (fun i => max (hA.eigenvalues i) 0) *
    (hA.eigenvectorUnitary : Matrix n n ℝ)ᴴ

/-- **C4 (symmetric PSD).** The spectral clip of any real symmetric matrix is
    symmetric positive semidefinite: the clipped diagonal has nonnegative
    entries (hence is PSD), and conjugation by the eigenvector unitary
    preserves PSD. -/
theorem spectralClip_posSemidef {A : Matrix n n ℝ} (hA : A.IsHermitian) :
    (spectralClip hA).PosSemidef := by
  have hd : (0 : n → ℝ) ≤ fun i => max (hA.eigenvalues i) 0 := fun i => le_max_right _ 0
  exact (Matrix.PosSemidef.diagonal hd).mul_mul_conjTranspose_same _

/-- **C4 (fixes PSD inputs).** If `A` is already positive semidefinite, its
    spectral clip equals `A`: the eigenvalues are already nonnegative
    (`IsHermitian.posSemidef_iff_eigenvalues_nonneg`), so clipping is the
    identity on the eigenvalue vector, and the spectral theorem recomposes
    `A` exactly. -/
theorem spectralClip_eq_self_of_posSemidef {A : Matrix n n ℝ} (hA : A.IsHermitian)
    (hpsd : A.PosSemidef) :
    spectralClip hA = A := by
  have heig : (0 : n → ℝ) ≤ hA.eigenvalues :=
    (Matrix.IsHermitian.posSemidef_iff_eigenvalues_nonneg hA).mp hpsd
  have hclip : (fun i => max (hA.eigenvalues i) 0) = hA.eigenvalues :=
    funext fun i => max_eq_left (heig i)
  show (hA.eigenvectorUnitary : Matrix n n ℝ) *
      Matrix.diagonal (fun i => max (hA.eigenvalues i) 0) *
      (hA.eigenvectorUnitary : Matrix n n ℝ)ᴴ = A
  rw [hclip]
  conv_rhs => rw [hA.spectral_theorem]
  rfl

end EigenvalueClip

end PricingPerspective.Discrete.CorrelationMatrix
