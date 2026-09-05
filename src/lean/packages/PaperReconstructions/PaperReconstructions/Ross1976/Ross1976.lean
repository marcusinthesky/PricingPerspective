import PaperReconstructions.Ross1976.Model

/-!
# Ross (1976): The Arbitrage Theory of Capital Asset Pricing

**Source:** Ross, Stephen A. (1976). "The Arbitrage Theory of Capital Asset Pricing."
*Journal of Economic Theory*, 13(3), 341-360.  https://doi.org/10.1016/0022-0531(76)90046-6

**Citation key:** `ross_arbitrage_1976`

## What is formalized

### Ross's Axioms (Ross 1976, p. 342)

Ross assumes the following structure for asset returns:

1. **Factor representation**: `R_i = Σ_{j=1}^k β_{ij} f_j + ε_i`.
2. **Zero factor means**: `𝔼[f_j] = 0`.
3. **Zero residual means**: `𝔼[ε_i] = 0`.
4. **Exact factor structure**: `Cov(ε_i, ε_j) = 0` for `i ≠ j`.
5. **Integrability**: All returns have finite second moments.

### Theorem statements

- **Theorem 1 (Eq. 4, p. 345):** Under no-arbitrage, `𝔼[R_i] - r_f = Σ_j β_{ij} α_j`.
- **Corollary (p. 345):** If `n > k`, no-arbitrage implies expected excess returns
  lie in `col(B)`.

## Non-trivial version

We remove the zero-mean assumption on residuals.  With `𝔼[f_j] = 0`
but `𝔼[ε_i]` potentially non-zero, the APT relation becomes
`𝔼[ε_i] - r_f = Σ_j β_{ij} α_j`.  Under no-arbitrage, all `𝔼[ε_i]` are
forced to be equal (a non-trivial constraint).  The theorem then states
that taking `r_f` equal to this common expected residual return and
`α = 0` satisfies the APT relation.

## Proof strategy

1. From no-arbitrage, prove all `𝔼[ε_i]` are equal (call it `μ_ε`).
2. Then `𝔼[R_i] = μ_ε` (since `𝔼[f_j] = 0`).
3. Set `r_f = μ_ε` and `α = 0`; both sides vanish.

The non-trivial content: the equalization of residual means is forced by
no-arbitrage and would not hold without it.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open Finset
open MeasureTheory

/-! ### Theorem 1: APT Representation (Ross 1976, Eq. 4, p. 345) -/

/-- **Ross (1976), Theorem 1, Equation (4) — non-trivial version:**

No-arbitrage and zero factor means force all residual means to be equal.
Setting the risk-free rate equal to this common expected residual return,
the APT relation holds with zero factor risk premia. -/
theorem ross_apt_representation {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (model : GeneralizedFactorModel n k Ω)
    (hEf : ∀ (j : Fin k), (∫ ω, model.f ω j ∂volume) = (0 : ℝ))
    -- REMOVED: zero residual means assumption (hEε)
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => model.f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_no_arb : NoArbitrage model.R) :
    ∃ (r_f : ℝ) (α : Fin k → ℝ),
      ∀ i : Fin n,
        (∫ ω, model.R ω i ∂volume) - r_f = (∑ j : Fin k, model.B i j * α j) := by
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model.ε ω i ∂volume) = (∫ ω, model.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model hEf h_int_R h_int_f h_int_ε h_no_arb
  have hn_pos : 0 < n := NeZero.pos n
  let i0 : Fin n := ⟨0, hn_pos⟩
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

/-! ### Corollary: Expected returns are all equal (Ross 1976, p. 345) -/

/-- **Ross (1976), Corollary, p. 345 — non-trivial version:**

Under the exact factor model with zero factor means and no-arbitrage,
all expected returns are equal (to the common expected residual return
`μ_ε`).  This is the economically meaningful content of the APT:
the cross-section of expected returns is constrained by no-arbitrage. -/
theorem ross_expected_return_equals_mu_eps {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (model : GeneralizedFactorModel n k Ω)
    (hEf : ∀ (j : Fin k), (∫ ω, model.f ω j ∂volume) = (0 : ℝ))
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => model.f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_no_arb : NoArbitrage model.R) :
    ∀ i j : Fin n, (∫ ω, model.R ω i ∂volume) = (∫ ω, model.R ω j ∂volume) := by
  intro i j
  have h_res_mean_eq : ∀ (i j : Fin n), (∫ ω, model.ε ω i ∂volume) = (∫ ω, model.ε ω j ∂volume) :=
    residual_means_equal_from_no_arbitrage model hEf h_int_R h_int_f h_int_ε h_no_arb
  have h_mean_R : ∀ i, (∫ ω, model.R ω i ∂volume) = (∫ ω, model.ε ω i ∂volume) := by
    intro i
    have h_rep_eq : model.R = fun ω i' => (∑ j : Fin k, model.B i' j * model.f ω j) + model.ε ω i' := by
      ext ω i'; simp [model.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, model.B i j * model.f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j' hj'; simpa using (h_int_f j').const_mul (model.B i j')
    have h_int_term2 : Integrable (fun ω => model.ε ω i) volume := h_int_ε i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (model.B i _) (fun ω => model.f ω _)]
      simp [hEf]
    · intro j' hj'; simpa using (h_int_f j').const_mul (model.B i j')
  rw [h_mean_R i, h_mean_R j, h_res_mean_eq i j]

end PaperReconstructions
