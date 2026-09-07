import PaperReconstructions.Ross1976.Model

/-!
# Middleton & Satchell (2000): Deriving the APT when the Number of Factors is Unknown

**Source:** Middleton, L., and S. Satchell. (2000). "Deriving the APT when
the Number of Factors is Unknown."  (Working paper, University of
Cambridge, 1999).

**Citation key:** `middleton2000deriving`

## The source paper's setup and results

Sections up to and including "Proposition 3" below describe **Middleton & Satchell's**
results. What this file actually proves is a strict subset; see "What is actually
formalized" at the end of this header before citing anything here.

### Middleton-Satchell's setup (p. 92-93)

Middleton and Satchell consider two factor structures:

**True factor structure:**
`R = E + B_T T + e_T` where:
- `R` is the `n`-vector of asset returns
- `E` is the `n`-vector of expected returns
- `B_T` is `n × m` sensitivity matrix to `m` true factors
- `T` is the `m`-vector of true factors with `E[T] = 0`, `Cov(T) = I_m`
- `e_T` is the `n`-vector of idiosyncratic risk with `E[e_T] = 0`,
  `Cov(e_T) = D` (diagonal matrix)

**Proxy factor structure:**
`R = E + B_P P + e_P` where:
- `B_P` is `n × k` sensitivity matrix to `k` proxy factors
- `P` is the `k`-vector of proxies with `E[P] = 0`, `Cov(P) = I_k`
- `e_P` is the `n`-vector of mean-zero idiosyncratic risk

### Best Linear Projection (BLP) of proxies on true factors (p. 94)

The BLP of `P` given `T` is:
`P = D T + x` where `D` is the `k × m` regression matrix and `x` is
a `k`-vector of residuals with mean zero.

### Key condition: rank(D) = m

If `rank(D) = m` (i.e., `k ≥ m` and the proxies span the true factor space),
then `D` has a left inverse `N` such that `N D = I_m`.  This gives:
`T = D^{-1} P - N x` (Eq. 6)

Substituting into the true model yields the proxy representation with
correlated errors (Eq. 7).

### No-arbitrage condition (p. 102)

A sequence of portfolios `w_n` generates an asymptotic arbitrage opportunity
if along some subsequence:

| Equation | Condition |
| -------- | ---------- |
| `lim_{n→∞} Var(w_n' R_n) = 0` | (12a) Variance vanishes |
| `w_n' E[R_n] > 0` for all `n` | (12b) Positive expected return |

### Proposition 1 (p. 102): k = m case

When `k = m` and `rank(D) = m`, the APT holds with the proxy factors
as reference variables.  The pricing relation is:
`E[l] = λ_0 i + B_λ λ` where `λ` is a `k`-vector of risk premia.

### Proposition 2 (p. 103): k > m case

When `k > m` and `rank(D) = m`, the APT holds with `B* = B_T D^{-1}`.
The pricing errors are of order `O(1/k)` in a sense made precise.

### Proposition 3 (p. 104): k < m case

When `k < m`, the APT does NOT hold in the same sense.  The pricing
errors are of higher order `O(1/m)` vs `O(1/k)`.

## What is actually formalized (and what is not)

The three theorems below (`middleton_satchell_equal_factors`,
`middleton_satchell_more_proxies`, `middleton_satchell_fewer_proxies`) share one
conclusion — existence of `r_f` and `α` with `E[R_i] - r_f = ∑_j B_ij α_j` for every
asset `i`. None formalizes the risk-premium structure, the pricing-error bounds, or the
`O(1/k)` / `O(1/m)` rates that distinguish Propositions 1-3 in the source.

The cause is the strength of the `NoArbitrage` hypothesis (`Ross1976/Model.lean:75-78`):
it asserts `E[w'R] ≤ 0` for *every* nonzero zero-cost `w`, and `-w` is again nonzero and
zero-cost, so `E[w'R] = 0` for all such `w`; taking `w = e_i - e_j` forces every asset to
have the same expected return. The shared conclusion is then discharged with `α ≡ 0` and
`r_f` the common mean, for any `m`, `k`, and any factor structure. The `k = m` / `k > m` /
`k < m` case split, the rank condition `h_rank_D`, the diagonality condition `h_diag_D`,
and `NoAsymptoticArbitrage` (itself `True`) do no work in these proofs.

These theorems are therefore **not** evidence for Middleton-Satchell's Propositions 1-3.
Realizing them needs (a) a payoff-level `NoArbitrage` rather than the expectation-level
condition above, and (b) a genuine `NoAsymptoticArbitrage`. Both are tracked as item D9 in
`src/lean/TODO.md`.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open PaperReconstructions
open Finset
open MeasureTheory

/-! ### Best Linear Projection (Middleton-Satchell Eq. 5) -/

/-- The Best Linear Projection (BLP) of `P` on `T` is `P = D T + x`
where `D` is the `k × m` regression matrix and `x` is the projection
residual (mean zero). -/
def BestLinearProjection {m k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (P : Ω → Fin k → ℝ) (T : Ω → Fin m → ℝ) (D : Matrix (Fin k) (Fin m) ℝ)
    (x : Ω → Fin k → ℝ) : Prop :=
  ∀ ω, P ω = fun i => (∑ j : Fin m, D i j * T ω j) + x ω i

/-! ### Zero means for factors -/

/-- True factors have zero mean. -/
def trueFactorsZeroMean {m : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (T : Ω → Fin m → ℝ) : Prop :=
  ∀ j, (∫ ω, T ω j ∂volume) = (0 : ℝ)

/-- Proxy factors have zero mean. -/
def proxyFactorsZeroMean {k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (P : Ω → Fin k → ℝ) : Prop :=
  ∀ j, (∫ ω, P ω j ∂volume) = (0 : ℝ)

/-! ### Diagonal residual covariance -/

/-- The residual covariance matrix `D` is diagonal:
`D i j = 0` whenever `i ≠ j`. -/
def residualCovarianceDiagonal {n : ℕ} (D : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j : Fin n, i ≠ j → D i j = 0

/-! ### Rank condition (Middleton-Satchell Lemma 1, p. 94) -/

/-- The regression matrix `D` has full row rank: `rank(D) = m`.
The `m` columns (each a vector in `ℝ^k`) are linearly independent. -/
def regressionMatrixFullRowRank {m k : ℕ} (D : Matrix (Fin k) (Fin m) ℝ) (_hmk : m ≤ k) : Prop :=
  LinearIndependent ℝ (fun (j : Fin m) => fun (i : Fin k) => D i j)

/-! ### No-arbitrage condition (Middleton-Satchell Eq. 12) -/

/-- Asymptotic arbitrage: a sequence of portfolios `w_n` has variance
vanishing along a subsequence but positive expected return.

**Vacuous placeholder** — this is literally `True`, so every `h_no_arb_asymp`
hypothesis below is free and constrains nothing. Realizing it is item D9 in
`src/lean/TODO.md`. -/
def NoAsymptoticArbitrage {n : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (R : Ω → Fin n → ℝ) : Prop :=
  True

/-! ### Proposition 1: k = m case (Middleton-Satchell p. 102) -/

/-- **Middleton & Satchell (2000), Proposition 1, p. 102:**

Assume the true factor structure `R = B_T T + e_T` with `E[T] = 0`,
`Cov(T) = I_m`, `E[e_T] = 0`, `Cov(e_T) = D` (diagonal), and the proxy
factor structure `R = B_P P + e_P` with `E[P] = 0`, `Cov(P) = I_k`.
Assume the BLP `P = D_reg T + x` with `rank(D_reg) = m` (so `k ≥ m`).
If no asymptotic arbitrage exists, then the APT holds with the proxy
factors as reference variables.

**Formalized here:** only the existence of `r_f` and `α` with
`E[R_i] - r_f = ∑_j B_ij α_j`. The proof uses neither `h_rank_D`, `h_diag_D`,
`h_no_arb_asymp`, nor the true-factor model: `h_no_arb` alone already forces all
expected returns to coincide (see the module header), so the conclusion is
discharged with `α ≡ 0` and `r_f` the common mean. The risk premia `λ` and the
left inverse `N` with `N D = I_m` are **not** formalized, and neither is any
pricing-error bound. See `src/lean/TODO.md` D9.
-/
theorem middleton_satchell_equal_factors {n m k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (hmk : m ≤ k)
    (model_T : GeneralizedFactorModel n m Ω) (model_P : GeneralizedFactorModel n k Ω)
    (hE_T : trueFactorsZeroMean model_T.f)
    (hE_P : proxyFactorsZeroMean model_P.f)
    (h_int_R : ∀ i, Integrable (fun ω => model_P.R ω i) volume)
    (h_int_T : ∀ j, Integrable (fun ω => model_T.f ω j) volume)
    (h_int_P : ∀ j, Integrable (fun ω => model_P.f ω j) volume)
    (h_int_eT : ∀ i, Integrable (fun ω => model_T.ε ω i) volume)
    (h_int_eP : ∀ i, Integrable (fun ω => model_P.ε ω i) volume)
    (h_no_arb : NoArbitrage model_P.R)
    (h_diag_D : residualCovarianceDiagonal
      (fun i j => (∫ ω, model_T.ε ω i * model_T.ε ω j ∂volume)))
    (h_rank_D : regressionMatrixFullRowRank
      (fun i j => (∫ ω, model_P.f ω i * model_T.f ω j ∂volume)) hmk)
    (h_no_arb_asymp : NoAsymptoticArbitrage model_P.R) :
    ∃ (r_f : ℝ) (α : Fin k → ℝ),
      ∀ i : Fin n,
        (∫ ω, model_P.R ω i ∂volume) - r_f = (∑ j : Fin k, model_P.B i j * α j) := by
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model_P.ε ω i ∂volume) = (∫ ω, model_P.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model_P hE_P h_int_R h_int_P h_int_eP
      (by
        intro w hw_cost hw_ne_zero
        exact h_no_arb w hw_cost hw_ne_zero)
  let i0 : Fin n := ⟨0, NeZero.pos n⟩
  let μ_ε := (∫ ω, model_P.ε ω i0 ∂volume)
  have h_mean_ε : ∀ i, (∫ ω, model_P.ε ω i ∂volume) = μ_ε := by
    intro i; exact h_res_mean_eq i i0
  have h_mean_R : ∀ i, (∫ ω, model_P.R ω i ∂volume) = μ_ε := by
    intro i
    have h_rep_eq : model_P.R = fun ω i' => (∑ j : Fin k, model_P.B i' j * model_P.f ω j) + model_P.ε ω i' := by
      ext ω i'; simp [model_P.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, model_P.B i j * model_P.f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j hj; simpa using (h_int_P j).const_mul (model_P.B i j)
    have h_int_term2 : Integrable (fun ω => model_P.ε ω i) volume := h_int_eP i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (model_P.B i _) (fun ω => model_P.f ω _)]
      simp [show ∀ j, (∫ ω, model_P.f ω j ∂volume) = 0 from hE_P, h_mean_ε i]
    · intro j hj; simpa using (h_int_P j).const_mul (model_P.B i j)
  refine ⟨μ_ε, fun _ => 0, fun i => ?_⟩
  rw [h_mean_R i]
  simp

/-! ### Proposition 2: k > m case (Middleton-Satchell p. 103) -/

/-- **Middleton & Satchell (2000), Proposition 2, p. 103:**

In the source, when there are more proxy factors than true factors (`k > m`) and
`rank(D_reg) = m`, the APT holds with `B* = B_T D^{-1}`, and the pricing errors
converge to zero as `n → ∞` at rate `O(1/k)`.

**Formalized here: nothing yet.** The statement below carries no `B*`, no rate and
no `n → ∞` — it is the same degenerate conclusion as Proposition 1 — and it is
`sorry`-ed rather than proven. It *reduces* to `middleton_satchell_equal_factors`
after a case split on `n = 0`, but closing it that way would certify the degenerate
conclusion, not Proposition 2. Left open deliberately; see `src/lean/TODO.md` D9.
-/
theorem middleton_satchell_more_proxies {n m k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (hmk : m < k)
    (model_T : GeneralizedFactorModel n m Ω) (model_P : GeneralizedFactorModel n k Ω)
    (hE_T : trueFactorsZeroMean model_T.f)
    (hE_P : proxyFactorsZeroMean model_P.f)
    (h_int_R : ∀ i, Integrable (fun ω => model_P.R ω i) volume)
    (h_int_T : ∀ j, Integrable (fun ω => model_T.f ω j) volume)
    (h_int_P : ∀ j, Integrable (fun ω => model_P.f ω j) volume)
    (h_int_eT : ∀ i, Integrable (fun ω => model_T.ε ω i) volume)
    (h_int_eP : ∀ i, Integrable (fun ω => model_P.ε ω i) volume)
    (h_no_arb : NoArbitrage model_P.R)
    (h_diag_D : residualCovarianceDiagonal
      (fun i j => (∫ ω, model_T.ε ω i * model_T.ε ω j ∂volume)))
    (h_rank_D : regressionMatrixFullRowRank
      (fun i j => (∫ ω, model_P.f ω i * model_T.f ω j ∂volume)) (by omega))
    (h_no_arb_asymp : NoAsymptoticArbitrage model_P.R)
    (h_bound_e : True) :  -- E[e_T^2] ≤ σ² < ∞
    ∃ (r_f : ℝ) (α : Fin k → ℝ),
      ∀ i : Fin n,
        (∫ ω, model_P.R ω i ∂volume) - r_f = (∑ j : Fin k, model_P.B i j * α j) := by
  -- Same proof pattern as Proposition 1
  sorry

/-! ### Proposition 3: k < m case (Middleton-Satchell p. 104) -/

/-- **Middleton & Satchell (2000), Proposition 3, p. 104:**

In the source, when there are fewer proxy factors than true factors (`k < m`), the
APT does **not** hold in the same sense: pricing errors are of higher order `O(1/m)`
compared to the `O(1/k)` rate in the `k ≥ m` case, and the sum of squared pricing
errors diverges as `n → ∞`.

**Formalized here: an opposite-looking statement, and deliberately so.** The
conclusion below is identical to Proposition 1's and *is* provable, because
`h_no_arb` alone forces all expected returns to coincide (see the module header); it
is discharged with `α ≡ 0`. Read this as a statement about the weakness of the
formalized `NoArbitrage`, **not** as evidence that the APT holds when `k < m`. The
divergence claim needs the matrix algebra and the genuine asymptotic-arbitrage
condition tracked in `src/lean/TODO.md` D9.
-/
theorem middleton_satchell_fewer_proxies {n m k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (hkm : k < m)
    (model_T : GeneralizedFactorModel n m Ω) (model_P : GeneralizedFactorModel n k Ω)
    (hE_T : trueFactorsZeroMean model_T.f)
    (hE_P : proxyFactorsZeroMean model_P.f)
    (h_int_R : ∀ i, Integrable (fun ω => model_P.R ω i) volume)
    (h_int_T : ∀ j, Integrable (fun ω => model_T.f ω j) volume)
    (h_int_P : ∀ j, Integrable (fun ω => model_P.f ω j) volume)
    (h_int_eT : ∀ i, Integrable (fun ω => model_T.ε ω i) volume)
    (h_int_eP : ∀ i, Integrable (fun ω => model_P.ε ω i) volume)
    (h_no_arb : NoArbitrage model_P.R)
    (h_diag_D : residualCovarianceDiagonal
      (fun i j => (∫ ω, model_T.ε ω i * model_T.ε ω j ∂volume)))
    (h_no_arb_asymp : NoAsymptoticArbitrage model_P.R) :
    ∃ (r_f : ℝ) (α : Fin k → ℝ),
      ∀ i : Fin n,
        (∫ ω, model_P.R ω i ∂volume) - r_f = (∑ j : Fin k, model_P.B i j * α j) := by
  -- With zero means, the APT holds trivially; the non-convergence
  -- is about the rate of the bound, which requires matrix algebra.
  -- Here we use the same proof as Proposition 1.
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model_P.ε ω i ∂volume) = (∫ ω, model_P.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model_P hE_P h_int_R h_int_P h_int_eP
      (by
        intro w hw_cost hw_ne_zero
        exact h_no_arb w hw_cost hw_ne_zero)
  let i0 : Fin n := ⟨0, NeZero.pos n⟩
  let μ_ε := (∫ ω, model_P.ε ω i0 ∂volume)
  have h_mean_ε : ∀ i, (∫ ω, model_P.ε ω i ∂volume) = μ_ε := by
    intro i; exact h_res_mean_eq i i0
  have h_mean_R : ∀ i, (∫ ω, model_P.R ω i ∂volume) = μ_ε := by
    intro i
    have h_rep_eq : model_P.R = fun ω i' => (∑ j : Fin k, model_P.B i' j * model_P.f ω j) + model_P.ε ω i' := by
      ext ω i'; simp [model_P.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, model_P.B i j * model_P.f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j hj; simpa using (h_int_P j).const_mul (model_P.B i j)
    have h_int_term2 : Integrable (fun ω => model_P.ε ω i) volume := h_int_eP i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (model_P.B i _) (fun ω => model_P.f ω _)]
      simp [show ∀ j, (∫ ω, model_P.f ω j ∂volume) = 0 from hE_P, h_mean_ε i]
    · intro j hj; simpa using (h_int_P j).const_mul (model_P.B i j)
  refine ⟨μ_ε, fun _ => 0, fun i => ?_⟩
  rw [h_mean_R i]
  simp

end PaperReconstructions
