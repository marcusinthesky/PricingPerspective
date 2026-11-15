import PricingPerspective.Spatial.Model

/-!
# Stability of spatial operators

Metric-neutral perturbation bounds for the Leontief inverse and covariance,
with row normalization under an explicit positive row-sum floor.
-/

namespace PricingPerspective.Spatial.Stability

open Matrix PricingPerspective.Spatial
open scoped Matrix.Norms.Operator

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

end PricingPerspective.Spatial.Stability
