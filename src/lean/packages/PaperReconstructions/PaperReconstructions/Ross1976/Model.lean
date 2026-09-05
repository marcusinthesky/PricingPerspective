import PaperReconstructions.Ross1976.Types
import PaperReconstructions.FTAP

/-!
# Factor Models

Finite factor-model structures for asset returns. This module defines the
canonical discrete factor model and arbitrage predicates.

The `ExactFactorModel` is kept for backward compatibility with paper modules
that reference it; the canonical definition going forward is
`GeneralizedFactorModel`.
-/

namespace PaperReconstructions

open MeasureTheory
open Finset
open Matrix

/-! ### Exact factor model (backward compatibility) -/

structure ExactFactorModel (n k : ℕ) (Ω : Type*) [MeasureSpace Ω]
    (R : Ω → ReturnVec n) where
  B : Matrix (Fin n) (Fin k) ℝ
  f : Ω → FactorVec k
  ε : Ω → ReturnVec n
  hEf : ∀ (k : Fin k), (∫ ω, f ω k ∂volume) = (0 : ℝ)
  hEε : ∀ (i : Fin n), (∫ ω, ε ω i ∂volume) = (0 : ℝ)
  hCovε : ∀ (i j : Fin n), i ≠ j →
    (∫ ω, (ε ω i) * (ε ω j) ∂volume) = (0 : ℝ)
  hrep : ∀ ω, R ω = fun i => ((Finset.sum Finset.univ fun k' => B i k' * f ω k') + ε ω i)

/-! ### Generalized factor model (canonical definition) -/

/-- **Generalized factor model** (canonical discrete form).

A linear factor model where each asset return `R_i` is expressed as a linear
combination of `k` common factors `f_j` plus an idiosyncratic residual `ε_i`.
No assumptions are baked in: zero-mean errors, orthogonality, and other economic
assumptions are stated as separate theorem hypotheses when needed.

The representation equation is:
`R_i(ω) = (∑_j B_{ij} · f_j(ω)) + ε_i(ω)`
-/
structure GeneralizedFactorModel (n k : ℕ) (Ω : Type*) [MeasureSpace Ω] where
  R : Ω → ReturnVec n
  B : Matrix (Fin n) (Fin k) ℝ
  f : Ω → FactorVec k
  ε : Ω → ReturnVec n
  hrep : ∀ ω i, R ω i = (∑ j, B i j * f ω j) + ε ω i

/-! ### One-factor model -/

/-- **One-factor model**: a `GeneralizedFactorModel` with a single factor
(`k = 1`).  The factor is a scalar function `f : Ω → ℝ`, and the loading
matrix `B` reduces to a column vector `b : Fin n → ℝ`.

The representation equation simplifies to:
`R_i(ω) = b_i · f(ω) + ε_i(ω)`
-/
structure OneFactorModel (n : ℕ) (Ω : Type*) [MeasureSpace Ω] where
  R : Ω → ReturnVec n
  b : ReturnVec n
  f : Ω → ℝ
  ε : Ω → ReturnVec n
  hrep : ∀ ω i, R ω i = b i * f ω + ε ω i

/-! ### Arbitrage predicates -/

/-- **Zero-cost portfolio**: a portfolio whose weights sum to zero. -/
def IsZeroCost {n : ℕ} (w : ReturnVec n) : Prop :=
  (∑ i : Fin n, w i) = 0

/-- **No-arbitrage**: there is no nonzero zero-cost portfolio that generates
a nonnegative expected payoff under the return distribution.

Formally: for all weight vectors `w` with zero sum, if `w ≠ 0`, then the
integral of `∑_i w_i · R_i` is non-positive (i.e., no riskless profit). -/
def NoArbitrage {n : ℕ} {Ω : Type*} [MeasureSpace Ω] (R : Ω → ReturnVec n) : Prop :=
  ∀ (w : ReturnVec n), IsZeroCost w → w ≠ 0 → (∫ ω, (∑ i, w i * R ω i) ∂volume) ≤ 0

/-! ### Helper: sum over univ equals sum over a two-element subset when function is zero elsewhere -/

lemma sum_over_pair {n : ℕ} {a b : Fin n} (h_ne : a ≠ b) (f : Fin n → ℝ)
    (h_zero : ∀ x, x ≠ a → x ≠ b → f x = 0) :
    ∑ x : Fin n, f x = f a + f b := by
  have h_sub : ({a, b} : Finset (Fin n)) ⊆ Finset.univ := Finset.subset_univ _
  have h_zero' : ∀ x, x ∈ Finset.univ → x ∉ ({a, b} : Finset (Fin n)) → f x = 0 := by
    intro x hx_univ hx_not
    apply h_zero x
    · intro hx_eq_a; apply hx_not; simp [hx_eq_a]
    · intro hx_eq_b; apply hx_not; simp [hx_eq_b]
  rw [← Finset.sum_subset h_sub h_zero']
  simp [h_ne]

/-! ### Zero-cost portfolio expected return when factor means are zero -/

lemma zero_cost_expected_return_of_factor_means_zero {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    (R : Ω → ReturnVec n) (B : Matrix (Fin n) (Fin k) ℝ)
    (f : Ω → FactorVec k) (ε : Ω → ReturnVec n)
    (hEf : ∀ (j : Fin k), (∫ ω, f ω j ∂volume) = (0 : ℝ))
    (hrep : ∀ ω, R ω = fun i => ((Finset.sum Finset.univ fun j => B i j * f ω j) + ε ω i))
    (h_int_R : ∀ i, Integrable (fun ω => R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => ε ω i) volume)
    (w : Fin n → ℝ) (_hw_cost : IsZeroCost w) :
    (∫ ω, (∑ i : Fin n, w i * R ω i) ∂volume) = (∑ i : Fin n, w i * (∫ ω, ε ω i ∂volume)) := by
  have h_mean_R : ∀ i : Fin n, (∫ ω, R ω i ∂volume) = (∫ ω, ε ω i ∂volume) := by
    intro i
    have h_rep_eq : R = fun ω i' => (∑ j : Fin k, B i' j * f ω j) + ε ω i' := by
      ext ω i'; simp [hrep ω]
    rw [h_rep_eq]
    have h_int_term1 : Integrable (fun ω => ∑ j : Fin k, B i j * f ω j) volume := by
      refine integrable_finsetSum (Finset.univ : Finset (Fin k)) ?_
      intro j hj; simpa using (h_int_f j).const_mul (B i j)
    have h_int_term2 : Integrable (fun ω => ε ω i) volume := h_int_ε i
    rw [integral_add h_int_term1 h_int_term2]
    rw [integral_finsetSum (Finset.univ : Finset (Fin k)) ?_]
    · simp_rw [integral_const_mul (B i _) (fun ω => f ω _)]
      simp [hEf]
    · intro j hj; simpa using (h_int_f j).const_mul (B i j)
  rw [integral_finsetSum (Finset.univ : Finset (Fin n)) ?_]
  · simp_rw [integral_const_mul (w _) (fun ω => R ω _)]
    simp [h_mean_R]
  · intro i hi; simpa using (h_int_R i).const_mul (w i)

/-! ### No-arbitrage forces all residual means equal -/

/-- Under no-arbitrage and zero factor means, all residual means are equal.
This is a non-trivial consequence of the no-arbitrage condition. -/
lemma residual_means_equal_from_no_arbitrage {n k : ℕ} {Ω : Type*} [MeasureTheory.MeasureSpace Ω]
    [NeZero n]
    (model : GeneralizedFactorModel n k Ω)
    (hEf : ∀ (j : Fin k), (∫ ω, model.f ω j ∂volume) = (0 : ℝ))
    (h_int_R : ∀ i, Integrable (fun ω => model.R ω i) volume)
    (h_int_f : ∀ j, Integrable (fun ω => model.f ω j) volume)
    (h_int_ε : ∀ i, Integrable (fun ω => model.ε ω i) volume)
    (h_no_arb : NoArbitrage model.R)
    (i j : Fin n) :
    (∫ ω, model.ε ω i ∂volume) = (∫ ω, model.ε ω j ∂volume) := by
  by_cases h_eq : i = j
  · subst h_eq; rfl
  · let w : Fin n → ℝ := fun x =>
      if x = i then (1 : ℝ)
      else if x = j then (-1 : ℝ)
      else (0 : ℝ)
    have hw_cost : ∑ x : Fin n, w x = 0 := by
      rw [sum_over_pair h_eq w (by
        intro x hx_ne_i hx_ne_j
        simp [w, hx_ne_i, hx_ne_j])]
      simp [w]
      by_cases hji : j = i
      · exfalso; exact h_eq hji.symm
      · simp [hji]
    have hw_ne_zero : w ≠ 0 := by
      intro h; have : w i = 0 := by simpa using congrFun h i
      simp [w] at this
    have h_exp_return : (∫ ω, (∑ x : Fin n, w x * model.R ω x) ∂volume) ≤ 0 :=
      h_no_arb w hw_cost hw_ne_zero
    have h_portfolio_mean : (∫ ω, (∑ x : Fin n, w x * model.R ω x) ∂volume) =
        (∑ x : Fin n, w x * (∫ ω, model.ε ω x ∂volume)) :=
      zero_cost_expected_return_of_factor_means_zero model.R model.B model.f model.ε hEf
        (fun ω => funext (model.hrep ω))
        h_int_R h_int_f h_int_ε w hw_cost
    rw [h_portfolio_mean] at h_exp_return
    have h_sum_expr : (∑ x : Fin n, w x * (∫ ω, model.ε ω x ∂volume)) =
        (∫ ω, model.ε ω i ∂volume) - (∫ ω, model.ε ω j ∂volume) := by
      rw [sum_over_pair h_eq (fun x => w x * (∫ ω, model.ε ω x ∂volume)) (by
        intro x hx_ne_i hx_ne_j
        simp [w, hx_ne_i, hx_ne_j])]
      simp [w]
      by_cases hji : j = i
      · exfalso; exact h_eq hji.symm
      · simp [hji]; ring
    rw [h_sum_expr] at h_exp_return
    have h_le1 : (∫ ω, model.ε ω i ∂volume) ≤ (∫ ω, model.ε ω j ∂volume) := by linarith
    let w' : Fin n → ℝ := fun x =>
      if x = j then (1 : ℝ)
      else if x = i then (-1 : ℝ)
      else (0 : ℝ)
    have hw'_cost : ∑ x : Fin n, w' x = 0 := by
      rw [sum_over_pair (Ne.symm h_eq) w' (by
        intro x hx_ne_j hx_ne_i
        simp [w', hx_ne_j, hx_ne_i])]
      simp [w']
      by_cases hij : i = j
      · exfalso; exact h_eq hij
      · simp [hij]
    have hw'_ne_zero : w' ≠ 0 := by
      intro h; have : w' j = 0 := by simpa using congrFun h j
      simp [w'] at this
    have h_exp_return' : (∫ ω, (∑ x : Fin n, w' x * model.R ω x) ∂volume) ≤ 0 :=
      h_no_arb w' hw'_cost hw'_ne_zero
    have h_portfolio_mean' : (∫ ω, (∑ x : Fin n, w' x * model.R ω x) ∂volume) =
        (∑ x : Fin n, w' x * (∫ ω, model.ε ω x ∂volume)) :=
      zero_cost_expected_return_of_factor_means_zero model.R model.B model.f model.ε hEf
        (fun ω => funext (model.hrep ω))
        h_int_R h_int_f h_int_ε w' hw'_cost
    rw [h_portfolio_mean'] at h_exp_return'
    have h_sum_expr' : (∑ x : Fin n, w' x * (∫ ω, model.ε ω x ∂volume)) =
        (∫ ω, model.ε ω j ∂volume) - (∫ ω, model.ε ω i ∂volume) := by
      rw [sum_over_pair (Ne.symm h_eq) (fun x => w' x * (∫ ω, model.ε ω x ∂volume)) (by
        intro x hx_ne_j hx_ne_i
        simp [w', hx_ne_j, hx_ne_i])]
      simp [w']
      by_cases hij : i = j
      · exfalso; exact h_eq hij
      · simp [hij]; ring
    rw [h_sum_expr'] at h_exp_return'
    have h_le2 : (∫ ω, model.ε ω j ∂volume) ≤ (∫ ω, model.ε ω i ∂volume) := by linarith
    linarith

/-! ### FTAP

The one-period FTAP declarations that used to live here — `NoArbitrageFTAP`,
`IsEMMFTAP`, and `ftap_one_period_vector_iff` — now live in `PaperReconstructions.FTAP`,
imported above, so they are still available from this module's import path. They moved
because the FTAP is its own concept rather than part of the factor-model layer, and
because its backward direction is no longer a stub but a full construction.

`ftap_one_period_vector_iff` gained a `Measurable Y` hypothesis in the move. The
unconditional statement carried here previously was **false**, not merely unproven; see
`PaperReconstructions.FTAP.ftap_one_period_vector_iff_not_unconditional`.
-/

end PaperReconstructions
