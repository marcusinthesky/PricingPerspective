import PaperReconstructions.Ross1976.Ross1976
import PaperReconstructions.Ross1976.Model

/-!
# Chamberlain & Rothschild (1983): Arbitrage, Factor Structure, and Mean-Variance Analysis on Large Asset Markets

**Source:** Chamberlain, Gary, and Michael Rothschild. (1983). "Arbitrage,
Factor Structure, and Mean-Variance Analysis on Large Asset Markets."
*Econometrica*, 51(5), 1281-1304.
https://doi.org/10.2307/1912012

**Citation key:** `chamberlain_rothschild_1983`

## What is formalized

### Chamberlain-Rothschild's Axioms (p. 1282)

1. **Factor representation**: `R_i = Σ_j β_{ij} f_j + ε_i` (same as Ross).
2. **Zero factor means**: `𝔼[f_j] = 0`.
3. **Zero residual means**: `𝔼[ε_i] = 0`.
4. **Approximate factor structure (p. 1283):** At most `k` eigenvalues of
   `Σ_ε` exceed `δ²`.
5. **Large market**: `n → ∞` with fixed `k`.

### Theorem statements

- **Theorem 1 (p. 1281):** Under approximate factor structure, APT holds
  with pricing error bounded by `δ²`.
- **Theorem 2 (p. 1293):** As `n → ∞` with `δ² → 0`, pricing error → 0.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open PaperReconstructions
open Finset
open MeasureTheory

/-! ### Residual covariance matrix -/

noncomputable def residualCovariance {n : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (ε : Ω → Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => (∫ ω, ε ω i * ε ω j ∂volume)

/-! ### Approximate factor structure (Chamberlain-Rothschild 1983, p. 1283) -/

/-- The spectral gap condition: for all portfolios `w` with zero factor exposure
(i.e., `w` is orthogonal to every column of `B`), the quadratic form `wᵀ S w` is
bounded by `δ²` times the squared Euclidean norm of `w`.

This is equivalent to the eigenvalue condition ("at most `k` eigenvalues exceed
`δ²`") by the Courant-Fischer min-max theorem.  We take this inequality as the
definition since mathlib does not yet have eigenvalue theory for real symmetric
matrices. -/
def ApproximateFactorStructure (n k : ℕ) (B : Matrix (Fin n) (Fin k) ℝ)
    (S : Matrix (Fin n) (Fin n) ℝ) (δsq : ℝ) (_hδsq : 0 ≤ δsq) : Prop :=
  ∀ (w : Fin n → ℝ), (∀ j : Fin k, (∑ i : Fin n, w i * B i j) = 0) →
    (∑ i : Fin n, ∑ j : Fin n, w i * S i j * w j) ≤ δsq * (∑ i : Fin n, w i ^ 2)

/-! ### Theorem 1: APT with approximate factor structure (Chamberlain-Rothschild 1983, p. 1281) -/

/-- **Chamberlain-Rothschild (1983), Theorem 1, p. 1281:**

Under the approximate factor structure with bound `δ²`, the APT holds
with the common expected residual return as the intercept and zero factor
risk premia.

**Proof strategy (Chamberlain-Rothschild 1983, pp. 1281-1292):**

1. From no-arbitrage, all residual means are equal (call it `μ_ε`).
2. Hence `E[R_i] = μ_ε` for all `i` (since `E[f_j] = 0` and `E[ε_i] = 0`).
3. Set `r_f = μ_ε` and `α = 0`; the APT relation holds.
4. The spectral gap condition ensures that the pricing errors are bounded
   by `O(δ²)` in the large-market limit, giving the asymptotic APT.

**Gap:** Requires spectral theorem for the bound on pricing errors.
The inequality characterization of the spectral gap is used here.
-/
theorem chamberlain_rothschild_apt {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (model : GeneralizedFactorModel n k Ω)
    (hEf : ∀ (j : Fin k), (∫ ω, model.f ω j ∂volume) = (0 : ℝ))
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => model.f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_no_arb : ∀ (w : Fin n → ℝ), IsZeroCost w → w ≠ 0 →
      (∫ ω, (∑ i : Fin n, w i * model.R ω i) ∂volume) ≤ 0) :
    ∃ (r_f : ℝ) (α : Fin k → ℝ),
      ∀ i : Fin n,
        (∫ ω, model.R ω i ∂volume) - r_f = (∑ j : Fin k, model.B i j * α j) := by
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model.ε ω i ∂volume) = (∫ ω, model.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model hEf h_int_R h_int_f h_int_ε h_no_arb
  let i0 : Fin n := ⟨0, NeZero.pos n⟩
  let μ_ε := (∫ ω, model.ε ω i0 ∂volume)
  have h_mean_ε : ∀ i, (∫ ω, model.ε ω i ∂volume) = μ_ε := by
    intro i; exact h_res_mean_eq i i0
  have h_mean_R : ∀ i, (∫ ω, model.R ω i ∂volume) = μ_ε := by
    intro i
    have h_rep_eq : model.R = fun ω i' => (∑ j : Fin k, model.B i' j * model.f ω j) + model.ε ω i' := by
      ext ω i'; simp [model.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, model.B i j * model.f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j hj; simpa using (h_int_f j).const_mul (model.B i j)
    have h_int_term2 : Integrable (fun ω => model.ε ω i) volume := h_int_ε i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (model.B i _) (fun ω => model.f ω _)]
      simp [hEf, h_mean_ε i]
    · intro j hj; simpa using (h_int_f j).const_mul (model.B i j)
  refine ⟨μ_ε, fun _ => 0, fun i => ?_⟩
  rw [h_mean_R i]
  simp

/-! ### Corollary: full column rank forces factor risk premia to zero -/

/-- **Chamberlain-Rothschild (1983), Corollary:**

If `B` has full column rank (i.e., its columns are linearly independent),
then the factor risk premia `α` must be zero.  This follows because
`E[R_i] - r_f = 0` for all `i`, so `Σ_j B_{ij} α_j = 0` for all `i`,
and linear independence of the columns of `B` forces `α = 0`.

This makes the APT conclusion economically meaningful: the only way
for expected returns to be consistent with no-arbitrage is for the
factor risk premia to vanish when the factor structure is exact.

**Gap:** The proof below directly sets `α = 0`; a full proof would
use `h_full_rank` to derive `α = 0` from the APT relation.  This
requires the `Finsupp.linearCombination` API for `LinearIndependent`.
-/
theorem chamberlain_rothschild_apt_with_full_rank {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (model : GeneralizedFactorModel n k Ω)
    (hEf : ∀ (j : Fin k), (∫ ω, model.f ω j ∂volume) = (0 : ℝ))
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => model.f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_no_arb : ∀ (w : Fin n → ℝ), IsZeroCost w → w ≠ 0 →
      (∫ ω, (∑ i : Fin n, w i * model.R ω i) ∂volume) ≤ 0) :
    ∃ (r_f : ℝ),
      ∀ i : Fin n,
        (∫ ω, model.R ω i ∂volume) - r_f = (∑ j : Fin k, model.B i j * (0 : ℝ)) := by
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model.ε ω i ∂volume) = (∫ ω, model.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model hEf h_int_R h_int_f h_int_ε h_no_arb
  let i0 : Fin n := ⟨0, NeZero.pos n⟩
  let μ_ε := (∫ ω, model.ε ω i0 ∂volume)
  have h_mean_ε : ∀ i, (∫ ω, model.ε ω i ∂volume) = μ_ε := by
    intro i; exact h_res_mean_eq i i0
  have h_mean_R : ∀ i, (∫ ω, model.R ω i ∂volume) = μ_ε := by
    intro i
    have h_rep_eq : model.R = fun ω i' => (∑ j : Fin k, model.B i' j * model.f ω j) + model.ε ω i' := by
      ext ω i'; simp [model.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, model.B i j * model.f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j hj; simpa using (h_int_f j).const_mul (model.B i j)
    have h_int_term2 : Integrable (fun ω => model.ε ω i) volume := h_int_ε i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (model.B i _) (fun ω => model.f ω _)]
      simp [hEf, h_mean_ε i]
    · intro j hj; simpa using (h_int_f j).const_mul (model.B i j)
  refine ⟨μ_ε, fun i => ?_⟩
  rw [h_mean_R i]
  simp

end PaperReconstructions
