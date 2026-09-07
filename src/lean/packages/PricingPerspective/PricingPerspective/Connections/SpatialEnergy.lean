import PricingPerspective.Discrete.Spatial
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow

/-!
# Energy-distance-implied spatial weights: certified radius propagation

Paper 5 rungs S4b/S5a/S5b (plan `2026-07-12_paper5-spatial-energy-weights`): the bridge
from per-entry energy-distance radii to certified perturbation bounds on the spatial
covariance of `Discrete/Spatial.lean`. One-writer-per-file: this module CONSUMES
`Discrete/Spatial.lean` and never edits it.

Contents:
* **S4b (joint perturbation).** `norm_smul_sub_smul_le` splits `‖ρ'•W' − ρ•W‖`;
  `leontief_sub_norm_le_joint` bounds the Leontief perturbation under simultaneous
  (ρ, W) ambiguity; `leontief_transpose` transports everything to the transpose side
  (needed because the L∞ operator norm is not transpose-invariant);
  `spatialCov_sub_norm_le_joint` / `spatialCov_entry_bracket_joint` give the certified
  entrywise covariance brackets — the energy-robustness box ∩ PSD interface under W-ambiguity.
* **S5a (kernel Lipschitz).** `mul_exp_neg_sq_le` (elementary AM-GM + `1 + x ≤ eˣ` core)
  and `gaussKernel_sub_abs_le`: the Gaussian kernel `t ↦ exp(−t²/h)` is Lipschitz with
  constant `√(2/h)·e^(−1/2)`, mapping energy-distance radii to kernel-entry radii.
* **S5b (row normalization).** `rowNormalize` (the pre-registered W construction),
  `norm_le_of_abs_row_sum_le` (row-sum criterion for the L∞ operator norm),
  `rowNormalize_sub_norm_le` (quotient perturbation bound — row normalization is NOT
  globally Lipschitz; the row-sum floor `smin` is an explicit hypothesis and an
  empirical gate), and `rowNormalize_norm_le_one` (the `‖W‖ ≤ 1` gate fact that turns
  the machine-checked hypothesis `‖ρ•W‖ < 1` into `|ρ| < 1`).
* **S5a′ (barycentre-QP Lipschitz).** `IsStronglyConvexQPSolution` discloses — as a named
  hypothesis in the style of `Connections/SpatialAsymptotics.lean`'s `IsPerronLimit`, never an
  `axiom` — that the energy-barycentre QP solution map `W♭` is globally Lipschitz in the
  energy-distance data. `barycentre_weight_sub_norm_le` / `barycentre_leontief_sub_norm_le`
  propagate a data radius through `W♭` to the Leontief / `Σ_SAR` brackets, certifying the
  microfounded construction and not only the kernel `W^h`.

S5c (full composition into the T3/T8 regret chain of `Continuous/RobustEnergy.lean`)
is deliberately deferred until the empirical pipeline fixes its exact hypothesis shape;
with the rungs here it is pure assembly.
-/

namespace PricingPerspective.Connections.SpatialEnergy

open Matrix PricingPerspective.Discrete.Spatial
open scoped Matrix.Norms.Operator

/-! ## S4b: joint (ρ, W) perturbation -/

section JointPerturbation

/-- Splitting a joint scalar–matrix perturbation:
`‖a•x − b•y‖ ≤ |a|‖x − y‖ + |a − b|‖y‖` (any real normed space). -/
theorem norm_smul_sub_smul_le {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (a b : ℝ) (x y : E) : ‖a • x - b • y‖ ≤ |a| * ‖x - y‖ + |a - b| * ‖y‖ := by
  have hsplit : a • x - b • y = a • (x - y) + (a - b) • y := by
    rw [smul_sub, sub_smul]; abel
  calc ‖a • x - b • y‖ = ‖a • (x - y) + (a - b) • y‖ := by rw [hsplit]
    _ ≤ ‖a • (x - y)‖ + ‖(a - b) • y‖ := norm_add_le _ _
    _ = |a| * ‖x - y‖ + |a - b| * ‖y‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]

variable {ι : Type*} [Fintype ι] [DecidableEq ι]
variable {ρ ρ' : ℝ} {W W' V : Matrix ι ι ℝ}

/-- Joint (ρ, W) perturbation of the Leontief inverse: under the spectral gates for both
parameter pairs,
`‖(1−ρ'W')⁻¹ − (1−ρW)⁻¹‖ ≤ (|ρ'|‖W'−W‖ + |ρ'−ρ|‖W‖) / ((1−‖ρ'•W'‖)(1−‖ρ•W‖))`.
Instance of the generic resolvent rung plus `norm_smul_sub_smul_le`. -/
theorem leontief_sub_norm_le_joint (h : ‖ρ • W‖ < 1) (h' : ‖ρ' • W'‖ < 1) :
    ‖leontief ρ' W' - leontief ρ W‖ ≤
      (|ρ'| * ‖W' - W‖ + |ρ' - ρ| * ‖W‖) / ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := by
  rcases isEmpty_or_nonempty ι with hι | hι
  · haveI := hι
    have hle : ‖leontief ρ' W' - leontief ρ W‖ = 0 := by rw [linfty_opNorm_def]; simp
    rw [hle]
    apply div_nonneg
    · have h1 : 0 ≤ |ρ'| * ‖W' - W‖ := mul_nonneg (abs_nonneg _) (norm_nonneg _)
      have h2 : 0 ≤ |ρ' - ρ| * ‖W‖ := mul_nonneg (abs_nonneg _) (norm_nonneg _)
      linarith
    · exact mul_nonneg (by linarith) (by linarith)
  · haveI := hι
    have key := norm_inverse_one_sub_sub_le h' h
    have hnum : ‖ρ' • W' - ρ • W‖ ≤ |ρ'| * ‖W' - W‖ + |ρ' - ρ| * ‖W‖ :=
      norm_smul_sub_smul_le ρ' ρ W' W
    have hd : (0 : ℝ) < (1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖) := mul_pos (by linarith) (by linarith)
    unfold leontief
    simp only [nonsing_inv_eq_ringInverse]
    calc ‖Ring.inverse (1 - ρ' • W') - Ring.inverse (1 - ρ • W)‖
        ≤ ‖ρ' • W' - ρ • W‖ / ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := key
      _ ≤ (|ρ'| * ‖W' - W‖ + |ρ' - ρ| * ‖W‖) / ((1 - ‖ρ' • W'‖) * (1 - ‖ρ • W‖)) := by
          rw [div_eq_mul_inv, div_eq_mul_inv]
          exact mul_le_mul_of_nonneg_right hnum (inv_nonneg.mpr hd.le)

/-- The Leontief inverse commutes with transposition: `((1−ρW)⁻¹)ᵀ = (1−ρWᵀ)⁻¹`.
Transports all perturbation bounds to the transpose side, whose L∞ operator norm is a
different quantity (max column sum) with its own spectral gate `‖ρ•Wᵀ‖ < 1`. -/
theorem leontief_transpose (ρ : ℝ) (W : Matrix ι ι ℝ) :
    (leontief ρ W)ᵀ = leontief ρ Wᵀ := by
  unfold leontief
  rw [Matrix.transpose_nonsing_inv, Matrix.transpose_sub, Matrix.transpose_one,
    Matrix.transpose_smul]

/-- Norm bound on the spatial-covariance perturbation under joint (ρ, W) ambiguity
(shared innovation covariance `V`); purely algebraic, no invertibility hypotheses. -/
theorem spatialCov_sub_norm_le_joint :
    ‖spatialCov ρ' W' V - spatialCov ρ W V‖ ≤
      ‖leontief ρ' W' - leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W')ᵀ‖ +
        ‖leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W')ᵀ - (leontief ρ W)ᵀ‖ := by
  unfold spatialCov
  set A' := leontief ρ' W'
  set A := leontief ρ W
  have hid : A' * V * A'ᵀ - A * V * Aᵀ = (A' - A) * V * A'ᵀ + A * V * (A'ᵀ - Aᵀ) := by
    simp only [sub_mul, mul_sub]; abel
  rw [hid]
  refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
  · refine (linfty_opNorm_mul _ _).trans ?_
    gcongr
    exact linfty_opNorm_mul _ _
  · refine (linfty_opNorm_mul _ _).trans ?_
    gcongr
    exact linfty_opNorm_mul _ _

/-- Certified entrywise covariance bracket under joint (ρ, W) ambiguity — the Paper 5 energy-robustness
box ∩ PSD interface with W-uncertainty included. -/
theorem spatialCov_entry_bracket_joint (i j : ι) :
    |spatialCov ρ' W' V i j - spatialCov ρ W V i j| ≤
      ‖leontief ρ' W' - leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W')ᵀ‖ +
        ‖leontief ρ W‖ * ‖V‖ * ‖(leontief ρ' W')ᵀ - (leontief ρ W)ᵀ‖ := by
  calc |spatialCov ρ' W' V i j - spatialCov ρ W V i j|
      = |(spatialCov ρ' W' V - spatialCov ρ W V) i j| := by rw [Matrix.sub_apply]
    _ ≤ ‖spatialCov ρ' W' V - spatialCov ρ W V‖ := abs_entry_le_norm _ i j
    _ ≤ _ := spatialCov_sub_norm_le_joint

end JointPerturbation

/-! ## S5a: Gaussian-kernel Lipschitz bound -/

section KernelLipschitz

/-- Elementary core (no calculus): `u·e^(−u²) ≤ e^(−1/2)/√2` for `u ≥ 0`.
Route: AM-GM `√2·u ≤ u² + 1/2`, then `1 + x ≤ eˣ` at `x = u² − 1/2`. -/
theorem mul_exp_neg_sq_le (u : ℝ) (hu : 0 ≤ u) :
    u * Real.exp (-(u ^ 2)) ≤ Real.exp (-(1 / 2)) / Real.sqrt 2 := by
  have hs2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have hs_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hexp_pos : 0 < Real.exp (-(u ^ 2)) := Real.exp_pos _
  have hamgm : Real.sqrt 2 * u ≤ u ^ 2 + 1 / 2 := by
    nlinarith [sq_nonneg (Real.sqrt 2 * u - 1), hs2]
  have hexp : u ^ 2 + 1 / 2 ≤ Real.exp (u ^ 2 - 1 / 2) := by
    have := Real.add_one_le_exp (u ^ 2 - 1 / 2); linarith
  have hprod : Real.sqrt 2 * u * Real.exp (-(u ^ 2))
      ≤ Real.exp (u ^ 2 - 1 / 2) * Real.exp (-(u ^ 2)) :=
    mul_le_mul_of_nonneg_right (le_trans hamgm hexp) hexp_pos.le
  have hRHS : Real.exp (u ^ 2 - 1 / 2) * Real.exp (-(u ^ 2)) = Real.exp (-(1 / 2)) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [hRHS] at hprod
  rw [le_div_iff₀ hs_pos]
  calc u * Real.exp (-(u ^ 2)) * Real.sqrt 2
      = Real.sqrt 2 * u * Real.exp (-(u ^ 2)) := by ring
    _ ≤ Real.exp (-(1 / 2)) := hprod

/-- The Gaussian kernel map `t ↦ exp(−t²/h)` used in the pre-registered W construction
is Lipschitz with constant `√(2/h)·e^(−1/2)`: per-entry energy-distance radii map to
kernel-entry radii with this factor (Paper 5 rung S5a). -/
theorem gaussKernel_sub_abs_le {h : ℝ} (hh : 0 < h) (a b : ℝ) :
    |Real.exp (-(a ^ 2) / h) - Real.exp (-(b ^ 2) / h)| ≤
      Real.sqrt (2 / h) * Real.exp (-(1 / 2)) * |a - b| := by
  have hderiv : ∀ x : ℝ,
      HasDerivAt (fun t : ℝ => Real.exp (-(t ^ 2) / h))
        (Real.exp (-(x ^ 2) / h) * (-(2 * x) / h)) x := by
    intro x
    have hp : HasDerivAt (fun t : ℝ => t ^ 2) (2 * x) x := by simpa using hasDerivAt_pow 2 x
    exact (hp.neg.div_const h).exp
  have hdiff : ∀ x ∈ (Set.univ : Set ℝ),
      DifferentiableAt ℝ (fun t : ℝ => Real.exp (-(t ^ 2) / h)) x :=
    fun x _ => (hderiv x).differentiableAt
  have hbound : ∀ x ∈ (Set.univ : Set ℝ),
      ‖deriv (fun t : ℝ => Real.exp (-(t ^ 2) / h)) x‖ ≤
        Real.sqrt (2 / h) * Real.exp (-(1 / 2)) := by
    intro x _
    rw [(hderiv x).deriv, Real.norm_eq_abs]
    have hsh : 0 < Real.sqrt h := Real.sqrt_pos.mpr hh
    set u : ℝ := |x| / Real.sqrt h with hu
    have hu0 : 0 ≤ u := div_nonneg (abs_nonneg _) hsh.le
    have husq : u ^ 2 = x ^ 2 / h := by
      rw [hu, div_pow, sq_abs, Real.sq_sqrt hh.le]
    have hexp_eq : Real.exp (-(x ^ 2) / h) = Real.exp (-(u ^ 2)) := by
      rw [husq]; congr 1; ring
    have habs : |Real.exp (-(x ^ 2) / h) * (-(2 * x) / h)|
        = Real.exp (-(u ^ 2)) * (2 * |x| / h) := by
      rw [abs_mul, abs_of_pos (Real.exp_pos _), hexp_eq]
      congr 1
      rw [abs_div, abs_of_pos hh, abs_neg, abs_mul]
      norm_num
    have hcoef : (2 / Real.sqrt h) * u = 2 * |x| / h := by
      rw [hu, div_mul_div_comm, Real.mul_self_sqrt hh.le]
    have hid : 2 / Real.sqrt h / Real.sqrt 2 = Real.sqrt (2 / h) := by
      rw [Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 2), div_div,
        mul_comm (Real.sqrt h) (Real.sqrt 2), ← div_div, Real.div_sqrt]
    rw [habs]
    calc Real.exp (-(u ^ 2)) * (2 * |x| / h)
        = (2 / Real.sqrt h) * (u * Real.exp (-(u ^ 2))) := by rw [← hcoef]; ring
      _ ≤ (2 / Real.sqrt h) * (Real.exp (-(1 / 2)) / Real.sqrt 2) := by
          apply mul_le_mul_of_nonneg_left (mul_exp_neg_sq_le u hu0)
          positivity
      _ = Real.sqrt (2 / h) * Real.exp (-(1 / 2)) := by rw [← hid]; ring
  have hmvt := Convex.norm_image_sub_le_of_norm_deriv_le hdiff hbound convex_univ
    (Set.mem_univ b) (Set.mem_univ a)
  simpa [Real.norm_eq_abs] using hmvt

end KernelLipschitz

/-! ## S5b: row normalization -/

section RowNormalize

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Row-sum criterion for the L∞ operator norm: if every absolute row sum is at most
`c` (with `c ≥ 0` covering the empty index type), then `‖M‖ ≤ c`. -/
theorem norm_le_of_abs_row_sum_le {M : Matrix ι ι ℝ} {c : ℝ}
    (hc0 : 0 ≤ c) (hc : ∀ i, ∑ j, |M i j| ≤ c) : ‖M‖ ≤ c := by
  rw [linfty_opNorm_def]
  have key : (Finset.univ.sup fun i : ι => ∑ j, ‖M i j‖₊) ≤ c.toNNReal := by
    apply Finset.sup_le
    intro i _
    rw [← NNReal.coe_le_coe, Real.coe_toNNReal c hc0]
    simp only [NNReal.coe_sum, coe_nnnorm, Real.norm_eq_abs]
    exact hc i
  calc ((Finset.univ.sup fun i : ι => ∑ j, ‖M i j‖₊ : NNReal) : ℝ)
      ≤ (c.toNNReal : ℝ) := NNReal.coe_le_coe.mpr key
    _ = c := Real.coe_toNNReal c hc0

/-- Row normalization of a kernel matrix: `(rowNormalize K) i j = K i j / ∑ₖ K i k`.
The pre-registered Paper 5 W construction applies this to `K i j = exp(−𝓔²ᵢⱼ/h)`. -/
noncomputable def rowNormalize (K : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of fun i j ↦ K i j / ∑ k, K i k

/-- Gate fact: a row-normalized nonnegative kernel has `‖W‖ ≤ 1` in the L∞ operator
norm, so the machine-checked spectral hypothesis `‖ρ • W‖ < 1` reduces to `|ρ| < 1`. -/
theorem rowNormalize_norm_le_one {K : Matrix ι ι ℝ} (hK : ∀ i j, 0 ≤ K i j)
    (hs : ∀ i, 0 < ∑ k, K i k) : ‖rowNormalize K‖ ≤ 1 := by
  apply norm_le_of_abs_row_sum_le (by norm_num)
  intro i
  have hcong : ∀ j ∈ Finset.univ, |rowNormalize K i j| = K i j / (∑ k, K i k) := by
    intro j _
    rw [rowNormalize, Matrix.of_apply, abs_of_nonneg (div_nonneg (hK i j) (hs i).le)]
  rw [Finset.sum_congr rfl hcong, ← Finset.sum_div, div_self (hs i).ne']

/-- Quotient perturbation bound for row normalization (Paper 5 rung S5b): if `K` is
entrywise nonnegative, both row-sum vectors are floored at `smin > 0`, and the kernels
differ entrywise by at most `δ`, then
`‖rowNormalize K' − rowNormalize K‖ ≤ 2·n·δ/smin`.
Row normalization is a quotient map, NOT globally Lipschitz: the floor `smin` is an
essential hypothesis and doubles as an empirical gate (reported per fold). -/
theorem rowNormalize_sub_norm_le {K K' : Matrix ι ι ℝ} {smin δ : ℝ}
    (hK : ∀ i j, 0 ≤ K i j) (hs : ∀ i, smin ≤ ∑ k, K i k) (hs' : ∀ i, smin ≤ ∑ k, K' i k)
    (hsmin : 0 < smin) (hδ : ∀ i j, |K' i j - K i j| ≤ δ) :
    ‖rowNormalize K' - rowNormalize K‖ ≤ 2 * (Fintype.card ι : ℝ) * δ / smin := by
  apply norm_le_of_abs_row_sum_le
  · rcases isEmpty_or_nonempty ι with he | hne
    · haveI := he; simp [Fintype.card_eq_zero]
    · obtain ⟨i⟩ := hne
      have hδ0 : 0 ≤ δ := le_trans (abs_nonneg _) (hδ i i)
      have hcard : (0 : ℝ) ≤ 2 * (Fintype.card ι : ℝ) * δ := by positivity
      exact div_nonneg hcard hsmin.le
  · intro i
    set S := ∑ k, K i k with hSdef
    set S' := ∑ k, K' i k with hS'def
    have hSpos : 0 < S := lt_of_lt_of_le hsmin (hs i)
    have hS'pos : 0 < S' := lt_of_lt_of_le hsmin (hs' i)
    have hSne : S ≠ 0 := hSpos.ne'
    have hS'ne : S' ≠ 0 := hS'pos.ne'
    have hδ0 : 0 ≤ δ := le_trans (abs_nonneg _) (hδ i i)
    have hbound_j : ∀ j, |(rowNormalize K' - rowNormalize K) i j|
        ≤ |K' i j - K i j| / S' + K i j * |S - S'| / (S' * S) := by
      intro j
      have he : (rowNormalize K' - rowNormalize K) i j = K' i j / S' - K i j / S := by
        simp only [Matrix.sub_apply, rowNormalize, Matrix.of_apply, ← hSdef, ← hS'def]
      rw [he]
      have hdecomp : K' i j / S' - K i j / S
          = (K' i j - K i j) / S' + K i j * (S - S') / (S' * S) := by
        field_simp; ring
      rw [hdecomp]
      refine (abs_add_le _ _).trans (le_of_eq ?_)
      rw [abs_div, abs_of_pos hS'pos, abs_div, abs_of_pos (mul_pos hS'pos hSpos), abs_mul,
        abs_of_nonneg (hK i j)]
    have hsum1 : (∑ j, |K' i j - K i j| / S') ≤ (Fintype.card ι : ℝ) * δ / smin := by
      calc ∑ j, |K' i j - K i j| / S'
          ≤ ∑ _j : ι, δ / smin := by
            apply Finset.sum_le_sum
            intro j _
            have h1 : |K' i j - K i j| ≤ δ := hδ i j
            have h2 : smin ≤ S' := hs' i
            gcongr
        _ = (Fintype.card ι : ℝ) * (δ / smin) := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        _ = (Fintype.card ι : ℝ) * δ / smin := by ring
    have hSS' : |S - S'| ≤ (Fintype.card ι : ℝ) * δ := by
      have heq : S - S' = ∑ k, (K i k - K' i k) := by
        rw [hSdef, hS'def, ← Finset.sum_sub_distrib]
      rw [heq]
      calc |∑ k, (K i k - K' i k)| ≤ ∑ k, |K i k - K' i k| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _k : ι, δ := by
            apply Finset.sum_le_sum; intro k _; rw [abs_sub_comm]; exact hδ i k
        _ = (Fintype.card ι : ℝ) * δ := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    have hsum2 : (∑ j, K i j * |S - S'| / (S' * S)) ≤ (Fintype.card ι : ℝ) * δ / smin := by
      have hsum2eq : (∑ j, K i j * |S - S'| / (S' * S)) = |S - S'| / S' := by
        rw [← Finset.sum_div, ← Finset.sum_mul, ← hSdef]
        field_simp
      rw [hsum2eq]
      have hden : smin ≤ S' := hs' i
      gcongr
    calc ∑ j, |(rowNormalize K' - rowNormalize K) i j|
        ≤ ∑ j, (|K' i j - K i j| / S' + K i j * |S - S'| / (S' * S)) :=
          Finset.sum_le_sum (fun j _ => hbound_j j)
      _ = (∑ j, |K' i j - K i j| / S') + ∑ j, K i j * |S - S'| / (S' * S) :=
          Finset.sum_add_distrib
      _ ≤ (Fintype.card ι : ℝ) * δ / smin + (Fintype.card ι : ℝ) * δ / smin :=
          add_le_add hsum1 hsum2
      _ = 2 * (Fintype.card ι : ℝ) * δ / smin := by ring

end RowNormalize

/-! ## S5a′: barycentre-QP solution-map Lipschitz (disclosed) -/

section BarycentreQP

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Barycentre-QP solution-map Lipschitz hypothesis** (assumed; NOT proved in-repo).

The program meant here is the **finite-mixture energy barycentre** solved by
`simulation.optimize.energy_barycentre_weights_qp` (the barycentric interaction operator `W♭`):
row `i` minimizes the squared energy distance `𝓔²(pᵢ, ∑_{j ≠ i} wⱼ pⱼ)` over the probability
simplex, equivalently the **quadratic** objective `2 cᵀw − wᵀQw` with `cⱼ = 𝔼‖Xᵢ − Xⱼ‖` and
`Q_{jk} = 𝔼‖Xⱼ − X_k‖` (the constant `b = 𝔼‖Xᵢ − Xᵢ′‖` is independent of `w` and drops out of the
argmin). It is NOT the linear/LP objective: an LP over the simplex is minimized at a vertex and is
not strongly convex. Strong convexity holds on the simplex's affine hull `{𝟙ᵀw = 1}`, where `Q` is
conditionally negative definite (a precondition the solver validates on the tangent space), so the
modulus is `2·λ_min(−Q|_tangent) > 0`; the solution map of such a program over a fixed polytope is
globally Lipschitz in the problem data, with a constant `Lw` bundling the reciprocal
strong-convexity modulus and the simplex diameter. This predicate packages exactly that Lipschitz
bound for the data-to-weights map `wflat = W♭`. A full mathlib proof of QP-solution-map
Lipschitz continuity is out of scope; rather than an `axiom` declaration (which the repository's
placeholder-inventory gate forbids), it is disclosed as a named hypothesis threaded into the
theorems below — exactly as `IsPerronLimit` in `Connections/SpatialAsymptotics.lean`. -/
def IsStronglyConvexQPSolution (wflat : Matrix ι ι ℝ → Matrix ι ι ℝ) (Lw : ℝ) : Prop :=
  ∀ E E' : Matrix ι ι ℝ, ‖wflat E' - wflat E‖ ≤ Lw * ‖E' - E‖

/-- **S5a′ (weight radius).** Conditional on the disclosed strong-convex-QP Lipschitz property, the
barycentric interaction operator `W♭ = wflat` maps an energy-distance data radius `‖𝓔' − 𝓔‖ ≤ r`
to a weight radius `Lw · r` — the same `𝓔 → W` propagation `gaussKernel_sub_abs_le` (S5a) gives
the kernel `W^h`, now for the microfounded construction `W♭`. -/
theorem barycentre_weight_sub_norm_le
    {wflat : Matrix ι ι ℝ → Matrix ι ι ℝ} {Lw r : ℝ} {E E' : Matrix ι ι ℝ}
    (hQP : IsStronglyConvexQPSolution wflat Lw) (hLw : 0 ≤ Lw) (hr : ‖E' - E‖ ≤ r) :
    ‖wflat E' - wflat E‖ ≤ Lw * r :=
  le_trans (hQP E E') (mul_le_mul_of_nonneg_left hr hLw)

/-- **S5a′ (Leontief bracket).** The weight radius of `barycentre_weight_sub_norm_le` propagates
through the joint S4b resolvent bound `leontief_sub_norm_le_joint` (at a common spatial parameter
`ρ` under the spectral gates `‖ρ • W♭‖ < 1`) to a certified perturbation bound on the Leontief
inverse of the barycentre weights: `W♭` — not only the kernel `W^h` of S5a — feeds the `Σ_SAR`
brackets, so `C1`'s microfounded construction is certifiable. -/
theorem barycentre_leontief_sub_norm_le
    {wflat : Matrix ι ι ℝ → Matrix ι ι ℝ} {Lw r ρ : ℝ} {E E' : Matrix ι ι ℝ}
    (hQP : IsStronglyConvexQPSolution wflat Lw) (hLw : 0 ≤ Lw) (hr : ‖E' - E‖ ≤ r)
    (h : ‖ρ • wflat E‖ < 1) (h' : ‖ρ • wflat E'‖ < 1) :
    ‖leontief ρ (wflat E') - leontief ρ (wflat E)‖ ≤
      |ρ| * (Lw * r) / ((1 - ‖ρ • wflat E'‖) * (1 - ‖ρ • wflat E‖)) := by
  have hW : ‖wflat E' - wflat E‖ ≤ Lw * r := barycentre_weight_sub_norm_le hQP hLw hr
  have hd : (0 : ℝ) < (1 - ‖ρ • wflat E'‖) * (1 - ‖ρ • wflat E‖) :=
    mul_pos (by linarith) (by linarith)
  have hbase := leontief_sub_norm_le_joint (ρ := ρ) (ρ' := ρ) (W := wflat E) (W' := wflat E') h h'
  refine le_trans hbase ?_
  rw [sub_self, abs_zero, zero_mul, add_zero, div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hW (abs_nonneg ρ))
    (inv_nonneg.mpr hd.le)

end BarycentreQP

end PricingPerspective.Connections.SpatialEnergy
