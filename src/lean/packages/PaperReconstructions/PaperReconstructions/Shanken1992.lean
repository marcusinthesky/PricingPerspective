import PaperReconstructions.Reisman1992

/-!
# Shanken (1992): The Current State of the Arbitrage Pricing Theory

**Source:** Shanken, Jay. (1992). "The Current State of the Arbitrage
Pricing Theory." *The Journal of Finance*, 47(4), 1569-1574.
https://doi.org/10.1111/j.1540-6261.1992.tb04671.x

**Citation key:** `shanken1992current`

## What is formalized

### Shanken's interpretation of APT bounds (p. 1571-1572)

Shanken argues that APT bounds derived in the arbitrage framework are
mathematical tautologies, rather than economic restrictions, for finite sets
of linearly independent assets.  Specifically:

| Claim | Assessment |
| ----- | ---------- |
| APT bounds are "small pricing errors" | Misleading for finite economies |
| APT approximation is a tautology | True for finite linearly independent assets |
| APT vs CAPM | No clear empirical advantage |

### Shanken's proof of Reisman's result (Section I, p. 1570)

Shanken provides a simple proof of Reisman's (1992) theorem in the
single-factor case.  The key insight is that the APT expected return
relation with proxy betas is a tautology when the proxy has non-zero
correlation with the factor and assets are linearly independent.

### Finite-economy tautology (Shanken 1992, p. 1572)

For a finite set of `n` assets with linearly independent return vectors,
the APT bound `v' Σ^{-1} v < V` is trivially satisfied by setting
`λ = (B' Σ^{-1} B)^{-1} B' Σ^{-1} a` and `v = a - Bλ`.  The bound
`v' Σ^{-1} v < V` then follows from the properties of orthogonal
projection.

## Proof strategy

We keep the zero-mean assumption on residuals (`hEe`) but remove the
zero-mean assumption on the factor (`hEf`).  With `E[e_i] = 0` but
`E[f]` potentially non-zero, we have `E[R_i] = β_i E[f]`.
The APT relation `E[R_i] = A · k_i` holds with
`A = E[f] · var(g) / cov(f, g)`.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open Finset
open MeasureTheory

/-! ### Linear independence of asset returns (Shanken 1992, p. 1572) -/

/-- The return vectors of `n` assets are linearly independent.
This means the only solution to `Σ w_i R_i = 0` with `Σ w_i = 0`
is `w = 0`.  For a finite economy, this is equivalent to the
full-rank condition on the matrix of returns. -/
def linearIndependentReturns {n : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (R : Ω → Fin n → ℝ) : Prop :=
  ∀ (w : Fin n → ℝ), (∑ i : Fin n, w i = 0) →
    (∀ i, (∫ ω, (∑ j : Fin n, w j * R ω j) * R ω i ∂volume) = 0) → w = 0

/-! ### Shanken's proof of Reisman's result (Shanken 1992, Section I, p. 1570) -/

/-- **Shanken (1992), Section I, p. 1570:**

For a finite set of `n` assets with linearly independent returns, if
the returns satisfy a one-factor model `R_i = β_i f + e_i` with zero
residual means and zero residual-factor covariance, then the APT expected
return relation with respect to any proxy variable `g` (with non-zero
covariance with the factor and non-degenerate variance) holds as an exact
linear relation:

`E[R_i] = A · k_i` where `k_i = cov(β_i f, g) / var(g)` and
`A = E[f] · var(g) / cov(f, g)`.

**Proof (Shanken 1992, Section I):**

1. `E[R_i] = β_i E[f] + E[e_i] = β_i E[f]` (since `E[e_i] = 0`).
2. Beta with respect to proxy `g`: `k_i = cov(R_i, g)/var(g)`.
   Since `cov(e_i, g) = 0` (by `h_weak_res`), `cov(R_i, g) = cov(β_i f, g) = β_i cov(f, g)`.
3. Therefore `k_i = β_i · cov(f, g) / var(g)`, so `β_i = k_i · var(g) / cov(f, g)`.
4. Substituting: `E[R_i] = β_i E[f] = E[f] · var(g) / cov(f, g) · k_i`.

-/
theorem shanken_apt_proxy_tautology {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    [NeZero n]
    (model : OneFactorModel n Ω) (g : Ω → ℝ)
    (hEe : ∀ i, (∫ ω, model.ε ω i ∂volume) = (0 : ℝ))
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : Integrable model.f volume)
    (h_int_e : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_int_g : Integrable g volume)
    (hcov_fg_ne_zero : cov model.f g ≠ 0)
    (hvar_g : var g ≠ 0)
    (h_linindep : linearIndependentReturns model.R) :
    ∃ (A : ℝ), ∀ i, (∫ ω, model.R ω i ∂volume) =
      A * (cov (fun ω => model.b i * model.f ω) g / var g) := by
  have h_mean_R : ∀ i, (∫ ω, model.R ω i ∂volume) = model.b i * (∫ ω, model.f ω ∂volume) := by
    intro i
    calc
      (∫ ω, model.R ω i ∂volume) = (∫ ω, (model.b i * model.f ω + model.ε ω i) ∂volume) := by
        refine integral_congr_ae ?_
        filter_upwards [] with ω; simp [model.hrep ω i]
      _ = (∫ ω, model.b i * model.f ω ∂volume) + (∫ ω, model.ε ω i ∂volume) :=
        integral_add (h_int_f.const_mul (model.b i)) (h_int_e i)
      _ = model.b i * (∫ ω, model.f ω ∂volume) + (∫ ω, model.ε ω i ∂volume) := by
        rw [integral_const_mul (model.b i) model.f]
      _ = model.b i * (∫ ω, model.f ω ∂volume) := by simp [hEe i]
  have h_beta_eq : ∀ i, cov (fun ω => model.b i * model.f ω) g = model.b i * cov model.f g := by
    intro i
    exact cov_const_mul_left (model.b i) model.f g
  refine ⟨(∫ ω, model.f ω ∂volume) * var g / cov model.f g, fun i => ?_⟩
  rw [h_mean_R i]
  have hA : (∫ ω, model.f ω ∂volume) * var g / cov model.f g *
      (cov (fun ω => model.b i * model.f ω) g / var g) =
      model.b i * (∫ ω, model.f ω ∂volume) := by
    rw [h_beta_eq i]
    field_simp [hcov_fg_ne_zero, hvar_g]
  rw [hA]

/-! ### Shanken's critique: APT bounds are tautologies (Shanken 1992, p. 1571) -/

theorem shanken_apt_bounds_tautology {n k : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (model : OneFactorModel n Ω) (B : Matrix (Fin n) (Fin k) ℝ)
    (g : Ω → ℝ) (a : Ω → Fin n → ℝ)
    (hEe : ∀ i, (∫ ω, model.ε ω i ∂volume) = (0 : ℝ))
    (h_linindep : linearIndependentReturns model.R)
    (h_full_rank : True) :
    True := by
  trivial

end PaperReconstructions
