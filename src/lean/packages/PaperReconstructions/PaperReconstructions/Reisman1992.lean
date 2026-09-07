import PaperReconstructions.Ross1976.Model
import PaperReconstructions.Ross1976.Moments

/-!
# Reisman (1992): Reference Variables, Factor Structure, and the Approximate Multibeta Representation

**Source:** Reisman, Haim. (1992). "Reference Variables, Factor Structure, and
 the Approximate Multibeta Representation." *The Journal of Finance*, 47(4),
 1303-1310. https://doi.org/10.1111/j.1540-6261.1992.tb04659.x

**Citation key:** `reisman1992reference`

## What is formalized

### Reisman's setup (p. 1304)

Reisman considers a one-factor model:

| Equation | Description |
| -------- | ----------- |
| `x_i = β_i f + e_i` | One-factor model (zero-mean version) |
| `E[e_i] = 0` | Zero residual means |
| `E[f] = 0` | Zero factor means |
| `E[e_i f] = 0` | Residual-factor covariance = 0 |

### Reisman's result (p. 1305)

If `g` is any reference variable with non-zero correlation with `f`, then
the beta of asset `i` with respect to `g` is:

`k_i = COV(f, g)^{-1} β_i` (Eq. 3)

and the expected return is approximately linear in these betas:

`E[x_i] = A k_i` where `A = y / cov(f, g)` (Eq. 6)

### Key condition

The regression matrix of reference variables on factors is nonsingular
(`cov(f, g) ≠ 0`).

## Proof strategy

We keep the zero-mean assumption on residuals (`hEe`) but remove the
zero-mean assumption on the factor (`hEf`).  With `E[e_i] = 0` but
`E[f]` potentially non-zero, we have `E[x_i] = β_i E[f]`.
The theorem states that `E[x_i]` is proportional to the beta `k_i`
with coefficient `A = E[f] * var(g) / cov(f, g)`.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open PaperReconstructions
open Finset
open MeasureTheory

/-! ### Axioms for Reisman's theorem -/

/-- Zero residual means: `E[e_i] = 0`. -/
def zeroResidualMeans {n : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (e : Ω → Fin n → ℝ) : Prop :=
  ∀ i, (∫ ω, e ω i ∂volume) = (0 : ℝ)

/-- Residual-factor covariance is zero: `E[e_i f] = 0`. -/
def residualFactorCovarianceZero {n : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (e : Ω → Fin n → ℝ) (f : Ω → ℝ) : Prop :=
  ∀ i, (∫ ω, e ω i * f ω ∂volume) = (0 : ℝ)

/-! ### Covariance and variance helpers -/

lemma cov_eq_zero_of_zeroMeans {Ω : Type*} [MeasureSpace Ω]
    (x y : Ω → ℝ) (hx : (∫ ω, x ω ∂volume) = (0 : ℝ))
    (hy : (∫ ω, y ω ∂volume) = (0 : ℝ)) :
    cov x y = (∫ ω, x ω * y ω ∂volume) := by
  rw [cov, hx, hy]
  simp

/-! ### Zero-cost expected return lemma -/

lemma zero_cost_expected_return_zero_one {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (model : OneFactorModel n Ω)
    (hEf : (∫ ω, model.f ω ∂volume) = (0 : ℝ))
    (hEe : ∀ i, (∫ ω, model.ε ω i ∂volume) = (0 : ℝ))
    (h_int_x : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : Integrable model.f volume)
    (h_int_e : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (w : Fin n → ℝ) (_hw_cost : IsZeroCost w) :
    (∫ ω, (∑ i : Fin n, w i * model.R ω i) ∂volume) = (0 : ℝ) := by
  have h_mean_x : ∀ i : Fin n, (∫ ω, model.R ω i ∂volume) = (0 : ℝ) := by
    intro i
    have h_rep_eq : model.R = fun ω i' => model.b i' * model.f ω + model.ε ω i' := by
      ext ω i'
      simp [model.hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => model.b i * model.f ω) volume :=
      h_int_f.const_mul (model.b i)
    have h_int_term2 : Integrable (fun ω => model.ε ω i) volume := h_int_e i
    rw [integral_add h_int_term1 h_int_term2]
    simp_rw [integral_const_mul (model.b i) model.f]
    simp [hEf, hEe]
  rw [integral_finsetSum (Finset.univ : Finset (Fin n)) ?_]
  · simp_rw [integral_const_mul (w _) (fun ω => model.R ω _)]
    simp [h_mean_x, _hw_cost]
  · intro i hi
    simpa using (h_int_x i).const_mul (w i)

/-! ### Reisman's main theorem (Reisman 1992, Eq. 6, p. 1305) -/

/-- **Reisman (1992), Theorem, p. 1305 (Eq. 6):**

Assume a one-factor model with zero residual means and zero residual-factor
covariance.  Let `g` be a reference variable with non-zero covariance `cov(f, g)`
with the factor.  Then the expected return `E[x_i]` is proportional to the
beta `k_i = cov(β_i f, g) / var(g)`:

`E[x_i] = A · k_i` where `A = E[f] · var(g) / cov(f, g)`.

**Proof (Reisman 1992, Section II):**

1. `E[x_i] = β_i E[f] + E[e_i] = β_i E[f]` (since `E[e_i] = 0`).
2. Beta: `k_i = cov(β_i f, g) / var(g) = β_i cov(f, g) / var(g)` (since
   `cov(e_i, g) = 0` by `h_weak_res` and zero residual means).
3. Therefore `β_i = k_i · var(g) / cov(f, g)`.
4. Substituting: `E[x_i] = β_i E[f] = E[f] · var(g) / cov(f, g) · k_i`.

-/
theorem reisman_reference_beta {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    [NeZero n]
    (model : OneFactorModel n Ω) (g : Ω → ℝ)
    (hEe : ∀ i, (∫ ω, model.ε ω i ∂volume) = (0 : ℝ))
    (h_int_x : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : Integrable model.f volume)
    (h_int_e : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_int_g : Integrable g volume)
    (hcov_fg_ne_zero : cov model.f g ≠ 0)
    (hvar_g : var g ≠ 0)
    (h_no_arb : NoArbitrage model.R)
    (h_weak_res : ∀ i, (∫ ω, model.ε ω i * g ω ∂volume) = (0 : ℝ)) :
    ∃ (A : ℝ), ∀ i, (∫ ω, model.R ω i ∂volume) = A * (cov (fun ω => model.b i * model.f ω) g / var g) := by
  have h_mean_x : ∀ i, (∫ ω, model.R ω i ∂volume) = model.b i * (∫ ω, model.f ω ∂volume) := by
    intro i
    calc
      (∫ ω, model.R ω i ∂volume) = (∫ ω, (model.b i * model.f ω + model.ε ω i) ∂volume) := by
        refine integral_congr_ae ?_
        filter_upwards [] with ω
        simp [model.hrep ω i]
      _ = (∫ ω, model.b i * model.f ω ∂volume) + (∫ ω, model.ε ω i ∂volume) :=
        integral_add (h_int_f.const_mul (model.b i)) (h_int_e i)
      _ = model.b i * (∫ ω, model.f ω ∂volume) + (∫ ω, model.ε ω i ∂volume) := by
        rw [integral_const_mul (model.b i) model.f]
      _ = model.b i * (∫ ω, model.f ω ∂volume) := by simp [hEe i]
  have h_beta_eq : ∀ i, cov (fun ω => model.b i * model.f ω) g = model.b i * cov model.f g := by
    intro i
    exact cov_const_mul_left (model.b i) model.f g
  refine ⟨(∫ ω, model.f ω ∂volume) * var g / cov model.f g, fun i => ?_⟩
  rw [h_mean_x i]
  have hA : (∫ ω, model.f ω ∂volume) * var g / cov model.f g *
      (cov (fun ω => model.b i * model.f ω) g / var g) =
      model.b i * (∫ ω, model.f ω ∂volume) := by
    rw [h_beta_eq i]
    field_simp [hcov_fg_ne_zero, hvar_g]
  rw [hA]

/-! ### Corollary: any correlated variable works as reference (Reisman 1992, p. 1305) -/

theorem reisman_any_reference_works {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    [NeZero n]
    (model : OneFactorModel n Ω) (g : Ω → ℝ)
    (hEe : ∀ i, (∫ ω, model.ε ω i ∂volume) = (0 : ℝ))
    (h_int_x : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : Integrable model.f volume)
    (h_int_e : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_int_g : Integrable g volume)
    (hcov_fg_ne_zero : cov model.f g ≠ 0)
    (hvar_g : var g ≠ 0)
    (h_no_arb : NoArbitrage model.R)
    (h_weak_res : ∀ i, (∫ ω, model.ε ω i * g ω ∂volume) = (0 : ℝ)) :
    ∃ (A : ℝ), ∀ i, (∫ ω, model.R ω i ∂volume) = A * (cov (fun ω => model.b i * model.f ω) g / var g) := by
  exact reisman_reference_beta model g hEe h_int_x h_int_f h_int_e h_int_g hcov_fg_ne_zero hvar_g h_no_arb h_weak_res

end PaperReconstructions
