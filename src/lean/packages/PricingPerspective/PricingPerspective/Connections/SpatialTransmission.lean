import Mathlib.LinearAlgebra.Matrix.Module
import Mathlib.LinearAlgebra.Matrix.Stochastic
import PricingPerspective.Discrete.Spatial

/-!
# Paper 1 transmission as the zero-feedback case of a Hilbert-valued SAR

Paper 1 supplies each firm with a primitive exposure `xi i = T (X i) (U i)`.  This module
adds the Paper 5 spatial closure: firms pay a literal quadratic adjustment cost relative to
`xi` and to the exposures of their economic neighbours.  A completing-square identity proves
that coordinatewise minimization is equivalent to the exact stationarity equation, avoiding
an otherwise irrelevant Fréchet-derivative formalization.

For a row-stochastic interaction matrix, stationarity at adjustment intensity `lambda >= 0`
gives the Hilbert-valued SAR

`b = rho W b + (1 - rho) xi`, where `rho = lambda / (1 + lambda)`.

The existing matrix Leontief inverse then gives

`b = (1 - rho) (I - rho W)⁻¹ xi`.

Every scalar inner-product projection satisfies the corresponding ordinary scalar SAR, and
`rho = 0` recovers the primitive Paper 1 exposure exactly.
-/

namespace PricingPerspective.Connections.SpatialTransmission

open Matrix
open PricingPerspective.Discrete.Spatial
open scoped InnerProductSpace Matrix.Module Matrix.Norms.Operator

variable {ι H : Type*} [Fintype ι] [DecidableEq ι]
variable [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- Matrix action on a finite family of Hilbert-valued exposures:
`(W • b) i = ∑ j, W i j • b j`. -/
def spatialMix (W : Matrix ι ι ℝ) (b : ι → H) : ι → H :=
  W • b

/-- Coordinate formula for the Hilbert-valued spatial action. -/
@[simp]
theorem spatialMix_apply (W : Matrix ι ι ℝ) (b : ι → H) (i : ι) :
    spatialMix W b i = ∑ j, W i j • b j :=
  rfl

/-- The weighted displacement of firm `i` from its economic neighbours. -/
def weightedPeerAdjustment (W : Matrix ι ι ℝ) (b : ι → H) (i : ι) : H :=
  ∑ j, W i j • (b i - b j)

/-- With a unit row sum, the weighted displacement of any candidate `a` from the peer profile
is `a - W • peers`. -/
theorem weightedPeerDeviation_eq_sub_spatialMix
    {W : Matrix ι ι ℝ} {peers : ι → H} {i : ι} {a : H}
    (hrow : ∑ j, W i j = 1) :
    (∑ j, W i j • (a - peers j)) = a - spatialMix W peers i := by
  simp only [smul_sub, Finset.sum_sub_distrib]
  rw [← Finset.sum_smul, hrow, one_smul]
  rfl

/-- The literal one-firm quadratic adjustment objective, evaluated at candidate exposure `a`
while the neighbouring exposures `peers` are held fixed. -/
noncomputable def quadraticAdjustmentObjective
    (lambda : ℝ) (W : Matrix ι ι ℝ) (xi peers : ι → H) (i : ι) (a : H) : ℝ :=
  (1 / 2) * ‖a - xi i‖ ^ 2 +
    (lambda / 2) * ∑ j, W i j * ‖a - peers j‖ ^ 2

/-- Closed-form best response to a fixed peer profile for the quadratic adjustment objective.
Its substantive uses assume `lambda ≥ 0`, so `1 + lambda` is invertible. -/
noncomputable def quadraticAdjustmentBestResponse
    (lambda : ℝ) (W : Matrix ι ι ℝ) (xi peers : ι → H) (i : ι) : H :=
  (1 + lambda)⁻¹ • (xi i + lambda • spatialMix W peers i)

/-- A profile is a coordinatewise equilibrium of the quadratic adjustment problem when every
firm's exposure minimizes its objective with the neighbouring profile held fixed. -/
noncomputable def IsQuadraticAdjustmentEquilibrium
    (lambda : ℝ) (W : Matrix ι ι ℝ) (xi b : ι → H) : Prop :=
  ∀ i a, quadraticAdjustmentObjective lambda W xi b i (b i) ≤
    quadraticAdjustmentObjective lambda W xi b i a

omit [InnerProductSpace ℝ H] in
/-- With nonnegative intensity and a row-stochastic interaction matrix, every coordinatewise
quadratic adjustment objective is nonnegative. -/
theorem quadraticAdjustmentObjective_nonneg
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi peers : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι) (i : ι) (a : H) :
    0 ≤ quadraticAdjustmentObjective lambda W xi peers i a := by
  unfold quadraticAdjustmentObjective
  apply add_nonneg
  · positivity
  · apply mul_nonneg (div_nonneg hlambda (by norm_num))
    exact Finset.sum_nonneg fun j _ ↦
      mul_nonneg (Matrix.nonneg_of_mem_rowStochastic hW) (sq_nonneg _)

/-- Three-point expansion used to complete the quadratic adjustment square. -/
theorem norm_sub_sq_eq_norm_sub_sq_add_cross
    (a center c : H) :
    ‖a - c‖ ^ 2 = ‖center - c‖ ^ 2 +
      2 * ⟪center - c, a - center⟫_ℝ + ‖a - center‖ ^ 2 := by
  have hsplit : a - c = (center - c) + (a - center) := by abel
  rw [hsplit, norm_add_sq_real]

omit [DecidableEq ι] in
/-- Completing-square identity for one coordinate. If `center` has zero adjustment gradient
and the relevant row sums to one, moving to `a` raises the objective by exactly
`((1 + lambda) / 2) ‖a - center‖²`. -/
theorem quadraticAdjustmentObjective_eq_of_gradient_eq_zero
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi peers : ι → H} {i : ι} {center a : H}
    (hrow : ∑ j, W i j = 1)
    (hgradient : center - xi i +
      lambda • (∑ j, W i j • (center - peers j)) = 0) :
    quadraticAdjustmentObjective lambda W xi peers i a =
      quadraticAdjustmentObjective lambda W xi peers i center +
        ((1 + lambda) / 2) * ‖a - center‖ ^ 2 := by
  have hcross :
      ⟪center - xi i, a - center⟫_ℝ +
        lambda * ∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ = 0 := by
    have hinner := congrArg (fun x : H ↦ ⟪x, a - center⟫_ℝ) hgradient
    simpa [inner_add_left, real_inner_smul_left, sum_inner, Finset.mul_sum] using hinner
  have hpeer :
      ∑ j, W i j * ‖a - peers j‖ ^ 2 =
        (∑ j, W i j * ‖center - peers j‖ ^ 2) +
          2 * (∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ) +
            ‖a - center‖ ^ 2 := by
    calc
      ∑ j, W i j * ‖a - peers j‖ ^ 2 =
          ∑ j, (W i j * ‖center - peers j‖ ^ 2 +
            W i j * (2 * ⟪center - peers j, a - center⟫_ℝ) +
              W i j * ‖a - center‖ ^ 2) := by
        apply Finset.sum_congr rfl
        intro j _
        rw [norm_sub_sq_eq_norm_sub_sq_add_cross]
        ring
      _ = (∑ j, W i j * ‖center - peers j‖ ^ 2) +
          (∑ j, W i j * (2 * ⟪center - peers j, a - center⟫_ℝ)) +
            ∑ j, W i j * ‖a - center‖ ^ 2 := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = (∑ j, W i j * ‖center - peers j‖ ^ 2) +
          2 * (∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ) +
            (∑ j, W i j) * ‖a - center‖ ^ 2 := by
        have hmiddle :
            (∑ j, W i j * (2 * ⟪center - peers j, a - center⟫_ℝ)) =
              2 * ∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro j _
          ring
        have hlast :
            (∑ j, W i j * ‖a - center‖ ^ 2) =
              (∑ j, W i j) * ‖a - center‖ ^ 2 := by
          rw [Finset.sum_mul]
        rw [hmiddle, hlast]
      _ = (∑ j, W i j * ‖center - peers j‖ ^ 2) +
          2 * (∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ) +
            ‖a - center‖ ^ 2 := by rw [hrow, one_mul]
  unfold quadraticAdjustmentObjective
  rw [norm_sub_sq_eq_norm_sub_sq_add_cross, hpeer]
  calc
    (1 / 2) *
          (‖center - xi i‖ ^ 2 + 2 * ⟪center - xi i, a - center⟫_ℝ +
            ‖a - center‖ ^ 2) +
        lambda / 2 *
          ((∑ j, W i j * ‖center - peers j‖ ^ 2) +
            2 * (∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ) +
              ‖a - center‖ ^ 2) =
        ((1 / 2) * ‖center - xi i‖ ^ 2 +
          lambda / 2 * ∑ j, W i j * ‖center - peers j‖ ^ 2) +
          (⟪center - xi i, a - center⟫_ℝ +
            lambda * ∑ j, W i j * ⟪center - peers j, a - center⟫_ℝ) +
            (1 + lambda) / 2 * ‖a - center‖ ^ 2 := by ring
    _ = (1 / 2) * ‖center - xi i‖ ^ 2 +
          lambda / 2 * ∑ j, W i j * ‖center - peers j‖ ^ 2 +
            (1 + lambda) / 2 * ‖a - center‖ ^ 2 := by rw [hcross, add_zero]

/-- With a unit row sum, weighted peer adjustment is own exposure minus spatially mixed
exposure. No symmetry or column-stochasticity is required. -/
theorem weightedPeerAdjustment_eq_sub_spatialMix
    {W : Matrix ι ι ℝ} {b : ι → H} {i : ι} (hrow : ∑ j, W i j = 1) :
    weightedPeerAdjustment W b i = b i - spatialMix W b i := by
  exact weightedPeerDeviation_eq_sub_spatialMix hrow

/-- Exact first-order stationarity condition for the quadratic adjustment problem

`(1 / 2) ‖a - xi i‖² + (lambda / 2) ∑ j, W i j ‖a - b j‖²`

at `a = b i`, treating neighbours' choices as fixed.  The derivative itself is not bundled:
this predicate records its exact vanishing-gradient equation. -/
def IsQuadraticAdjustmentStationary
    (lambda : ℝ) (W : Matrix ι ι ℝ) (xi b : ι → H) : Prop :=
  ∀ i, b i - xi i + lambda • weightedPeerAdjustment W b i = 0

/-- The closed-form best response has zero quadratic-adjustment gradient when the relevant
interaction row sums to one and `lambda ≥ 0`. -/
theorem quadraticAdjustmentBestResponse_gradient_eq_zero
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi peers : ι → H}
    (hlambda : 0 ≤ lambda) {i : ι} (hrow : ∑ j, W i j = 1) :
    quadraticAdjustmentBestResponse lambda W xi peers i - xi i +
      lambda • (∑ j, W i j •
        (quadraticAdjustmentBestResponse lambda W xi peers i - peers j)) = 0 := by
  have hc : 1 + lambda ≠ 0 := ne_of_gt (by linarith)
  let center := quadraticAdjustmentBestResponse lambda W xi peers i
  have hscale : (1 + lambda) • center =
      xi i + lambda • spatialMix W peers i := by
    dsimp [center, quadraticAdjustmentBestResponse]
    rw [smul_smul, mul_inv_cancel₀ hc, one_smul]
  rw [add_smul, one_smul] at hscale
  rw [weightedPeerDeviation_eq_sub_spatialMix hrow]
  calc
    center - xi i + lambda • (center - spatialMix W peers i) =
        (center + lambda • center) -
          (xi i + lambda • spatialMix W peers i) := by
      rw [smul_sub]
      abel
    _ = 0 := sub_eq_zero.mpr hscale

/-- Every stationary profile is a coordinatewise minimizer of the literal quadratic objective.
The proof is the completing-square identity, not a differentiability argument. -/
theorem quadraticAdjustmentEquilibrium_of_stationary
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι)
    (hstat : IsQuadraticAdjustmentStationary lambda W xi b) :
    IsQuadraticAdjustmentEquilibrium lambda W xi b := by
  intro i a
  have hgradient : b i - xi i +
      lambda • (∑ j, W i j • (b i - b j)) = 0 := by
    simpa [weightedPeerAdjustment] using hstat i
  have hgap := quadraticAdjustmentObjective_eq_of_gradient_eq_zero
    (a := a) (Matrix.sum_row_of_mem_rowStochastic hW i) hgradient
  have hcoef : 0 ≤ (1 + lambda) / 2 := div_nonneg (by linarith) (by norm_num)
  calc
    quadraticAdjustmentObjective lambda W xi b i (b i) ≤
        quadraticAdjustmentObjective lambda W xi b i (b i) +
          ((1 + lambda) / 2) * ‖a - b i‖ ^ 2 :=
      le_add_of_nonneg_right (mul_nonneg hcoef (sq_nonneg _))
    _ = quadraticAdjustmentObjective lambda W xi b i a := hgap.symm

/-- Conversely, every coordinatewise minimizer is stationary.  Comparing the chosen exposure
to the explicit best response in the completing-square identity forces their distance to zero. -/
theorem quadraticAdjustmentStationary_of_equilibrium
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι)
    (heq : IsQuadraticAdjustmentEquilibrium lambda W xi b) :
    IsQuadraticAdjustmentStationary lambda W xi b := by
  intro i
  let center := quadraticAdjustmentBestResponse lambda W xi b i
  have hrow : ∑ j, W i j = 1 := Matrix.sum_row_of_mem_rowStochastic hW i
  have hgradient : center - xi i +
      lambda • (∑ j, W i j • (center - b j)) = 0 := by
    exact quadraticAdjustmentBestResponse_gradient_eq_zero hlambda hrow
  have hgap := quadraticAdjustmentObjective_eq_of_gradient_eq_zero
    (a := b i) hrow hgradient
  have hle := heq i center
  have hcoef : 0 < (1 + lambda) / 2 := div_pos (by linarith) (by norm_num)
  have hprod : ((1 + lambda) / 2) * ‖b i - center‖ ^ 2 ≤ 0 := by
    linarith
  have hsquare_le : ‖b i - center‖ ^ 2 ≤ 0 := by
    exact (mul_le_mul_iff_of_pos_left hcoef).mp (by simpa using hprod)
  have hsquare : ‖b i - center‖ ^ 2 = 0 :=
    le_antisymm hsquare_le (sq_nonneg _)
  have hnorm : ‖b i - center‖ = 0 := by
    nlinarith [norm_nonneg (b i - center)]
  have hbi : b i = center := sub_eq_zero.mp (norm_eq_zero.mp hnorm)
  rw [weightedPeerAdjustment, hbi]
  exact hgradient

/-- Under nonnegative adjustment intensity and row-stochastic interactions, coordinatewise
optimality for the quadratic problem is equivalent to the exact stationarity equation. -/
theorem quadraticAdjustmentEquilibrium_iff_stationary
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι) :
    IsQuadraticAdjustmentEquilibrium lambda W xi b ↔
      IsQuadraticAdjustmentStationary lambda W xi b :=
  ⟨quadraticAdjustmentStationary_of_equilibrium hlambda hW,
    quadraticAdjustmentEquilibrium_of_stationary hlambda hW⟩

/-- Spatial autoregression parameter generated by quadratic adjustment intensity `lambda`. -/
noncomputable def adjustmentRho (lambda : ℝ) : ℝ :=
  lambda / (1 + lambda)

/-- A Hilbert-valued SAR with primitive Paper 1 exposure `xi` and spatial feedback `rho`.
The factor `(1 - rho)` is the normalization generated by quadratic adjustment. -/
def IsHilbertSAR (rho : ℝ) (W : Matrix ι ι ℝ) (xi b : ι → H) : Prop :=
  b = rho • spatialMix W b + (1 - rho) • xi

/-- The quadratic stationarity equation implies the unnormalized best-response system
`(1 + lambda) b = xi + lambda W b` when `W` has unit row sums. -/
theorem unnormalized_sar_of_quadraticAdjustmentStationary
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hrow : ∀ i, ∑ j, W i j = 1)
    (hstat : IsQuadraticAdjustmentStationary lambda W xi b) :
    (1 + lambda) • b = xi + lambda • spatialMix W b := by
  ext i
  have hi := hstat i
  rw [weightedPeerAdjustment_eq_sub_spatialMix (hrow i)] at hi
  rw [Pi.smul_apply, Pi.add_apply, Pi.smul_apply, add_smul, one_smul]
  apply eq_of_sub_eq_zero
  calc
    (b i + lambda • b i) - (xi i + lambda • spatialMix W b i) =
        b i - xi i + (lambda • b i - lambda • spatialMix W b i) := by
      abel
    _ = b i - xi i + lambda • (b i - spatialMix W b i) := by rw [smul_sub]
    _ = 0 := hi

/-- For nonnegative adjustment intensity, the quadratic first-order condition is exactly the
normalized Hilbert-valued SAR with `rho = lambda / (1 + lambda)`. -/
theorem sar_of_quadraticAdjustmentStationary
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hrow : ∀ i, ∑ j, W i j = 1)
    (hstat : IsQuadraticAdjustmentStationary lambda W xi b) :
    IsHilbertSAR (adjustmentRho lambda) W xi b := by
  have hc : 1 + lambda ≠ 0 := ne_of_gt (by linarith)
  have hbest := unnormalized_sar_of_quadraticAdjustmentStationary hrow hstat
  have hrho : (1 + lambda)⁻¹ * lambda = adjustmentRho lambda := by
    simp only [adjustmentRho, div_eq_mul_inv]
    ring
  have hone : (1 + lambda)⁻¹ = 1 - adjustmentRho lambda := by
    rw [adjustmentRho]
    field_simp
    ring
  unfold IsHilbertSAR
  ext i
  have hi := congrArg (fun x : H ↦ (1 + lambda)⁻¹ • x) (congrFun hbest i)
  simp only [Pi.smul_apply, Pi.add_apply, smul_add, smul_smul] at hi
  rw [inv_mul_cancel₀ hc, one_smul, hrho, hone] at hi
  simpa [add_comm] using hi

/-- The literal coordinatewise quadratic adjustment problem yields the normalized
Hilbert-valued SAR. Row-stochasticity supplies both nonnegative weights and unit row sums. -/
theorem sar_of_quadraticAdjustmentEquilibrium
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι)
    (heq : IsQuadraticAdjustmentEquilibrium lambda W xi b) :
    IsHilbertSAR (adjustmentRho lambda) W xi b :=
  sar_of_quadraticAdjustmentStationary hlambda
    (Matrix.sum_row_of_mem_rowStochastic hW)
    (quadraticAdjustmentStationary_of_equilibrium hlambda hW heq)

/-- Hilbert-valued reduced form for a generic spatial-lag equation.  Under the same norm gate
as `Discrete.Spatial.reduced_form_of_spatial_lag`, the unique solution is obtained by applying
the existing Leontief matrix to the Hilbert-valued innovation. -/
theorem hilbert_reduced_form_of_spatial_lag
    {rho : ℝ} {W : Matrix ι ι ℝ} {b innovation : ι → H}
    (hnorm : ‖rho • W‖ < 1)
    (hsar : b = rho • spatialMix W b + innovation) :
    b = spatialMix (leontief rho W) innovation := by
  have hmul : leontief rho W * (1 - rho • W) = 1 :=
    leontief_mul_one_sub_smul hnorm
  have hkey : spatialMix (1 - rho • W) b = innovation := by
    calc
      spatialMix (1 - rho • W) b = b - rho • spatialMix W b := by
        change (1 - rho • W) • b = b - rho • W • b
        simp only [sub_smul, one_smul]
        rw [smul_assoc]
      _ = innovation := sub_eq_of_eq_add (hsar.trans (add_comm _ _))
  calc
    b = spatialMix (1 : Matrix ι ι ℝ) b := by simp [spatialMix]
    _ = spatialMix (leontief rho W * (1 - rho • W)) b := by rw [hmul]
    _ = spatialMix (leontief rho W) (spatialMix (1 - rho • W) b) := by
      simp [spatialMix, mul_smul]
    _ = spatialMix (leontief rho W) innovation := by rw [hkey]

/-- Leontief reduced form of the normalized Hilbert-valued SAR:
`b = (1 - rho) • ((I - rho W)⁻¹ • xi)`. -/
theorem hilbert_sar_reduced_form
    {rho : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hnorm : ‖rho • W‖ < 1) (hsar : IsHilbertSAR rho W xi b) :
    b = (1 - rho) • spatialMix (leontief rho W) xi := by
  have hred := hilbert_reduced_form_of_spatial_lag hnorm hsar
  calc
    b = spatialMix (leontief rho W) ((1 - rho) • xi) := hred
    _ = (1 - rho) • spatialMix (leontief rho W) xi := by
      change leontief rho W • ((1 - rho) • xi) =
        (1 - rho) • leontief rho W • xi
      rw [smul_comm]

/-- End-to-end Paper 1 to Paper 5 bridge: quadratic adjustment stationarity, unit row sums,
and the existing Leontief norm gate imply the normalized Hilbert-valued reduced form. -/
theorem quadraticAdjustment_leontief_reduced_form
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hrow : ∀ i, ∑ j, W i j = 1)
    (hstat : IsQuadraticAdjustmentStationary lambda W xi b)
    (hnorm : ‖adjustmentRho lambda • W‖ < 1) :
    b = (1 - adjustmentRho lambda) •
      spatialMix (leontief (adjustmentRho lambda) W) xi :=
  hilbert_sar_reduced_form hnorm
    (sar_of_quadraticAdjustmentStationary hlambda hrow hstat)

/-- End-to-end optimization bridge: a coordinatewise equilibrium of the literal quadratic
adjustment objectives has the scaled Leontief reduced form used by Paper 5. -/
theorem quadraticAdjustmentEquilibrium_leontief_reduced_form
    {lambda : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (hlambda : 0 ≤ lambda) (hW : W ∈ Matrix.rowStochastic ℝ ι)
    (heq : IsQuadraticAdjustmentEquilibrium lambda W xi b)
    (hnorm : ‖adjustmentRho lambda • W‖ < 1) :
    b = (1 - adjustmentRho lambda) •
      spatialMix (leontief (adjustmentRho lambda) W) xi :=
  hilbert_sar_reduced_form hnorm
    (sar_of_quadraticAdjustmentEquilibrium hlambda hW heq)

/-- Scalar inner-product projection of a finite family of Hilbert-valued exposures. -/
def scalarProjection (h : H) (b : ι → H) : ι → ℝ :=
  fun i ↦ ⟪b i, h⟫_ℝ

/-- Scalar projection commutes with spatial mixing. -/
theorem scalarProjection_spatialMix
    (h : H) (W : Matrix ι ι ℝ) (b : ι → H) :
    scalarProjection h (spatialMix W b) = W *ᵥ scalarProjection h b := by
  ext i
  simp [scalarProjection, spatialMix_apply, Matrix.mulVec, dotProduct, sum_inner,
    real_inner_smul_left]

/-- Every scalar projection of a Hilbert-valued SAR satisfies the ordinary scalar SAR with
the same interaction matrix and feedback parameter. -/
theorem scalarProjection_sar
    {rho : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (h : H) (hsar : IsHilbertSAR rho W xi b) :
    scalarProjection h b =
      rho • (W *ᵥ scalarProjection h b) + (1 - rho) • scalarProjection h xi := by
  ext i
  have hi := congrArg (fun x : H ↦ ⟪x, h⟫_ℝ) (congrFun hsar i)
  simpa [IsHilbertSAR, scalarProjection, spatialMix_apply, inner_add_left, sum_inner,
    real_inner_smul_left, Matrix.mulVec, dotProduct, Finset.mul_sum] using hi

/-- The existing scalar Leontief theorem applies to every projection of the Hilbert-valued
SAR, giving the projected reduced form without a separate inverse construction. -/
theorem scalarProjection_reduced_form
    {rho : ℝ} {W : Matrix ι ι ℝ} {xi b : ι → H}
    (h : H) (hnorm : ‖rho • W‖ < 1) (hsar : IsHilbertSAR rho W xi b) :
    scalarProjection h b =
      (1 - rho) • (leontief rho W *ᵥ scalarProjection h xi) := by
  have hscalar := scalarProjection_sar h hsar
  have hred := reduced_form_of_spatial_lag hnorm hscalar
  simpa only [Matrix.mulVec_smul] using hred

/-- Zero quadratic adjustment intensity generates zero SAR feedback. -/
@[simp]
theorem adjustmentRho_zero : adjustmentRho 0 = 0 := by
  simp [adjustmentRho]

/-- At zero spatial feedback, the Hilbert-valued SAR is exactly the primitive Paper 1
transmission exposure `xi = T(X, U)`. -/
theorem hilbert_sar_zero_eq_primitive
    {W : Matrix ι ι ℝ} {xi b : ι → H} (hsar : IsHilbertSAR 0 W xi b) :
    b = xi := by
  simpa [IsHilbertSAR] using hsar

/-!
## Two priced interaction fields

The application prices peer alignment against two admissible interaction fields rather than
one.  Because both fields are row-stochastic, their convex mixture is again admissible and
the two-field adjustment objective is *identical* to the one-field objective evaluated at
that mixture.  Every result above therefore transfers without a second completing-square
argument, and the one-field model is recovered exactly at the endpoints of the mixture
weight.
-/

/-- Convex mixture of two interaction fields at weight `theta`. -/
def mixField (theta : ℝ) (WB WN : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  theta • WB + (1 - theta) • WN

omit [Fintype ι] [DecidableEq ι] in
@[simp]
theorem mixField_one (WB WN : Matrix ι ι ℝ) : mixField 1 WB WN = WB := by
  simp [mixField]

omit [Fintype ι] [DecidableEq ι] in
@[simp]
theorem mixField_zero (WB WN : Matrix ι ι ℝ) : mixField 0 WB WN = WN := by
  simp [mixField]

omit [Fintype ι] [DecidableEq ι] in
/-- Scaling a mixture is the same as scaling each field by its own share. -/
theorem smul_mixField (rho theta : ℝ) (WB WN : Matrix ι ι ℝ) :
    rho • mixField theta WB WN = (rho * theta) • WB + (rho * (1 - theta)) • WN := by
  simp [mixField, smul_add, smul_smul]

/-- A convex mixture of two admissible interaction fields is admissible: the peer-average
interpretation survives combining the fields. -/
theorem mixField_mem_rowStochastic {theta : ℝ} {WB WN : Matrix ι ι ℝ}
    (hWB : WB ∈ Matrix.rowStochastic ℝ ι) (hWN : WN ∈ Matrix.rowStochastic ℝ ι)
    (h0 : 0 ≤ theta) (h1 : theta ≤ 1) :
    mixField theta WB WN ∈ Matrix.rowStochastic ℝ ι := by
  rw [Matrix.mem_rowStochastic_iff_sum] at hWB hWN ⊢
  obtain ⟨hBnonneg, hBsum⟩ := hWB
  obtain ⟨hNnonneg, hNsum⟩ := hWN
  have hco : (0 : ℝ) ≤ 1 - theta := by linarith
  refine ⟨fun i j ↦ ?_, fun i ↦ ?_⟩
  · simpa [mixField] using
      add_nonneg (mul_nonneg h0 (hBnonneg i j)) (mul_nonneg hco (hNnonneg i j))
  · simp only [mixField, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hBsum i, hNsum i]
    ring

/-- The Hilbert-valued action of a mixture splits into the two field actions. -/
theorem spatialMix_mixField (theta : ℝ) (WB WN : Matrix ι ι ℝ) (b : ι → H) :
    spatialMix (mixField theta WB WN) b =
      theta • spatialMix WB b + (1 - theta) • spatialMix WN b := by
  ext i
  simp only [spatialMix_apply, mixField, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
    Pi.add_apply, Pi.smul_apply]
  rw [Finset.smul_sum, Finset.smul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ ↦ by rw [add_smul, smul_smul, smul_smul]

/-- The two-field quadratic adjustment objective: one stand-alone cost and two separately
priced peer-alignment costs, one for each admissible interaction field. -/
noncomputable def twoFieldAdjustmentObjective
    (lambdaB lambdaN : ℝ) (WB WN : Matrix ι ι ℝ) (xi peers : ι → H) (i : ι) (a : H) : ℝ :=
  (1 / 2) * ‖a - xi i‖ ^ 2 +
    (lambdaB / 2) * ∑ j, WB i j * ‖a - peers j‖ ^ 2 +
      (lambdaN / 2) * ∑ j, WN i j * ‖a - peers j‖ ^ 2

omit [DecidableEq ι] [InnerProductSpace ℝ H] in
/-- Pricing two fields at intensities `theta * lambda` and `(1 - theta) * lambda` is the
same objective as pricing their mixture at total intensity `lambda`.  Nothing is assumed of
`theta`, `lambda`, or the matrices: this is distributivity of the peer penalty over the
mixture. -/
theorem twoFieldAdjustmentObjective_eq
    (theta lambda : ℝ) (WB WN : Matrix ι ι ℝ) (xi peers : ι → H) (i : ι) (a : H) :
    twoFieldAdjustmentObjective (theta * lambda) ((1 - theta) * lambda) WB WN xi peers i a =
      quadraticAdjustmentObjective lambda (mixField theta WB WN) xi peers i a := by
  have hsplit : ∑ j, (theta * WB i j + (1 - theta) * WN i j) * ‖a - peers j‖ ^ 2 =
      theta * (∑ j, WB i j * ‖a - peers j‖ ^ 2) +
        (1 - theta) * (∑ j, WN i j * ‖a - peers j‖ ^ 2) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ ↦ by ring
  unfold twoFieldAdjustmentObjective quadraticAdjustmentObjective mixField
  simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  rw [hsplit]
  ring

/-- A profile is a two-field equilibrium when every firm's exposure minimizes its two-field
objective with the neighbouring profile held fixed. -/
noncomputable def IsTwoFieldAdjustmentEquilibrium
    (lambdaB lambdaN : ℝ) (WB WN : Matrix ι ι ℝ) (xi b : ι → H) : Prop :=
  ∀ i a, twoFieldAdjustmentObjective lambdaB lambdaN WB WN xi b i (b i) ≤
    twoFieldAdjustmentObjective lambdaB lambdaN WB WN xi b i a

omit [DecidableEq ι] [InnerProductSpace ℝ H] in
/-- Two-field equilibrium is one-field equilibrium at the mixture. -/
theorem isTwoFieldAdjustmentEquilibrium_iff
    (theta lambda : ℝ) (WB WN : Matrix ι ι ℝ) (xi b : ι → H) :
    IsTwoFieldAdjustmentEquilibrium (theta * lambda) ((1 - theta) * lambda) WB WN xi b ↔
      IsQuadraticAdjustmentEquilibrium lambda (mixField theta WB WN) xi b := by
  unfold IsTwoFieldAdjustmentEquilibrium IsQuadraticAdjustmentEquilibrium
  simp only [twoFieldAdjustmentObjective_eq]

/-- A Hilbert-valued spatial autoregression in two interaction fields.  The stand-alone
exposure keeps the residual weight `1 - rhoB - rhoN`. -/
def IsTwoFieldHilbertSAR
    (rhoB rhoN : ℝ) (WB WN : Matrix ι ι ℝ) (xi b : ι → H) : Prop :=
  b = rhoB • spatialMix WB b + rhoN • spatialMix WN b + (1 - rhoB - rhoN) • xi

/-- A one-field SAR at the mixture is a two-field SAR whose channel coefficients are the
mixture shares of the total feedback. -/
theorem isTwoFieldHilbertSAR_of_mixField
    {theta rho : ℝ} {WB WN : Matrix ι ι ℝ} {xi b : ι → H}
    (hsar : IsHilbertSAR rho (mixField theta WB WN) xi b) :
    IsTwoFieldHilbertSAR (rho * theta) (rho * (1 - theta)) WB WN xi b := by
  have hcoef : (1 : ℝ) - rho * theta - rho * (1 - theta) = 1 - rho := by ring
  unfold IsTwoFieldHilbertSAR
  rw [hcoef]
  calc
    b = rho • spatialMix (mixField theta WB WN) b + (1 - rho) • xi := hsar
    _ = (rho * theta) • spatialMix WB b + (rho * (1 - theta)) • spatialMix WN b +
          (1 - rho) • xi := by
      rw [spatialMix_mixField, smul_add, smul_smul, smul_smul]

/-- Mixture weight generated by two nonnegative adjustment intensities. -/
noncomputable def mixWeight (lambdaB lambdaN : ℝ) : ℝ :=
  lambdaB / (lambdaB + lambdaN)

/-- Spatial feedback contributed by one channel priced at `lambdaOwn`, given the total
adjustment intensity `lambdaTotal` charged across all channels. -/
noncomputable def channelRho (lambdaOwn lambdaTotal : ℝ) : ℝ :=
  lambdaOwn / (1 + lambdaTotal)

variable {lambdaB lambdaN : ℝ}

theorem mixWeight_nonneg (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    0 ≤ mixWeight lambdaB lambdaN :=
  div_nonneg hB (by linarith)

theorem mixWeight_le_one (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    mixWeight lambdaB lambdaN ≤ 1 := by
  rcases eq_or_lt_of_le (by linarith : (0 : ℝ) ≤ lambdaB + lambdaN) with h | h
  · simp [mixWeight, ← h]
  · rw [mixWeight, div_le_one h]
    linarith

/-- The mixture weight recovers the first channel's intensity from the total. -/
theorem mixWeight_mul_total (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    mixWeight lambdaB lambdaN * (lambdaB + lambdaN) = lambdaB := by
  rcases eq_or_lt_of_le (by linarith : (0 : ℝ) ≤ lambdaB + lambdaN) with h | h
  · have hB0 : lambdaB = 0 := by linarith
    simp [mixWeight, hB0]
  · have hne : lambdaB + lambdaN ≠ 0 := ne_of_gt h
    unfold mixWeight
    field_simp

/-- The complementary weight recovers the second channel's intensity from the total. -/
theorem one_sub_mixWeight_mul_total (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    (1 - mixWeight lambdaB lambdaN) * (lambdaB + lambdaN) = lambdaN := by
  have hexpand : (1 - mixWeight lambdaB lambdaN) * (lambdaB + lambdaN) =
      (lambdaB + lambdaN) - mixWeight lambdaB lambdaN * (lambdaB + lambdaN) := by ring
  rw [hexpand, mixWeight_mul_total hB hN]
  ring

/-- **Two-field spatial closure.**  When a firm prices misalignment against two admissible
interaction fields at nonnegative intensities, every coordinatewise equilibrium of the
literal quadratic objective satisfies a two-field Hilbert-valued spatial autoregression
whose channel coefficients are `lambda_k / (1 + lambda_B + lambda_N)`. -/
theorem twoField_sar_of_equilibrium
    {WB WN : Matrix ι ι ℝ} {xi b : ι → H}
    (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN)
    (hWB : WB ∈ Matrix.rowStochastic ℝ ι) (hWN : WN ∈ Matrix.rowStochastic ℝ ι)
    (heq : IsTwoFieldAdjustmentEquilibrium lambdaB lambdaN WB WN xi b) :
    IsTwoFieldHilbertSAR (channelRho lambdaB (lambdaB + lambdaN))
      (channelRho lambdaN (lambdaB + lambdaN)) WB WN xi b := by
  have hBeq : mixWeight lambdaB lambdaN * (lambdaB + lambdaN) = lambdaB :=
    mixWeight_mul_total hB hN
  have hNeq : (1 - mixWeight lambdaB lambdaN) * (lambdaB + lambdaN) = lambdaN :=
    one_sub_mixWeight_mul_total hB hN
  have hmix : mixField (mixWeight lambdaB lambdaN) WB WN ∈ Matrix.rowStochastic ℝ ι :=
    mixField_mem_rowStochastic hWB hWN (mixWeight_nonneg hB hN) (mixWeight_le_one hB hN)
  have heqmix : IsQuadraticAdjustmentEquilibrium (lambdaB + lambdaN)
      (mixField (mixWeight lambdaB lambdaN) WB WN) xi b := by
    rw [← isTwoFieldAdjustmentEquilibrium_iff, hBeq, hNeq]
    exact heq
  have hsar := isTwoFieldHilbertSAR_of_mixField
    (sar_of_quadraticAdjustmentEquilibrium (by linarith) hmix heqmix)
  have hrB : adjustmentRho (lambdaB + lambdaN) * mixWeight lambdaB lambdaN =
      channelRho lambdaB (lambdaB + lambdaN) := by
    unfold adjustmentRho channelRho
    rw [div_mul_eq_mul_div, mul_comm (lambdaB + lambdaN) (mixWeight lambdaB lambdaN), hBeq]
  have hrN : adjustmentRho (lambdaB + lambdaN) * (1 - mixWeight lambdaB lambdaN) =
      channelRho lambdaN (lambdaB + lambdaN) := by
    unfold adjustmentRho channelRho
    rw [div_mul_eq_mul_div,
      mul_comm (lambdaB + lambdaN) (1 - mixWeight lambdaB lambdaN), hNeq]
  rwa [hrB, hrN] at hsar

/-- The two channel coefficients sum to the one-field feedback generated by the total
adjustment intensity, so the two-field stability region is `rhoB + rhoN < 1`. -/
theorem channelRho_add (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    channelRho lambdaB (lambdaB + lambdaN) + channelRho lambdaN (lambdaB + lambdaN) =
      adjustmentRho (lambdaB + lambdaN) := by
  have hne : (1 : ℝ) + (lambdaB + lambdaN) ≠ 0 := ne_of_gt (by linarith)
  unfold channelRho adjustmentRho
  field_simp

/-- The stand-alone weight left over by the two channels. -/
theorem one_sub_channelRho (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    1 - channelRho lambdaB (lambdaB + lambdaN) - channelRho lambdaN (lambdaB + lambdaN) =
      (1 + (lambdaB + lambdaN))⁻¹ := by
  have hne : (1 : ℝ) + (lambdaB + lambdaN) ≠ 0 := ne_of_gt (by linarith)
  unfold channelRho
  field_simp
  ring

/-- Inverting the closure: each channel's adjustment intensity is its own spatial
coefficient divided by the stand-alone residual weight `1 - rhoB - rhoN`. -/
theorem channelRho_inversion (hB : 0 ≤ lambdaB) (hN : 0 ≤ lambdaN) :
    channelRho lambdaB (lambdaB + lambdaN) /
      (1 - channelRho lambdaB (lambdaB + lambdaN) -
        channelRho lambdaN (lambdaB + lambdaN)) = lambdaB := by
  have hne : (1 : ℝ) + (lambdaB + lambdaN) ≠ 0 := ne_of_gt (by linarith)
  rw [one_sub_channelRho hB hN]
  unfold channelRho
  field_simp

/-- Zero feedback in both channels leaves the firm at its stand-alone exposure. -/
theorem twoField_zero_eq_primitive
    {WB WN : Matrix ι ι ℝ} {xi b : ι → H}
    (hsar : IsTwoFieldHilbertSAR 0 0 WB WN xi b) : b = xi := by
  simpa [IsTwoFieldHilbertSAR] using hsar

end PricingPerspective.Connections.SpatialTransmission
