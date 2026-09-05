import Mathlib.Probability.Moments.Variance
import Mathlib.Data.Finset.Defs
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Function.L1Space.Integrable

import PaperReconstructions.Ross1976.Types

/-!
# Moments

Variance, covariance, and derived quantities for asset returns.  All
definitions are `noncomputable` since they use Lebesgue integration.

## Axiom justification

All assumptions are explicit hypotheses.  No unchecked `axiom`s.
-/

namespace PaperReconstructions

open MeasureTheory
open Finset

/-! ### Variance and covariance -/

/-- **Variance** of a real-valued random variable `x` over a measure space `Ω`.

Defined as `E[x²] - E[x]²`. -/
noncomputable def var {Ω : Type*} [MeasureSpace Ω] (x : Ω → ℝ) : ℝ :=
  (∫ ω, x ω * x ω) - (∫ ω, x ω) * (∫ ω, x ω)

/-- **Covariance** between two real-valued random variables `x` and `y` over `Ω`.

Defined as `E[xy] - E[x]·E[y]`. -/
noncomputable def cov {Ω : Type*} [MeasureSpace Ω] (x y : Ω → ℝ) : ℝ :=
  (∫ ω, x ω * y ω) - (∫ ω, x ω) * (∫ ω, y ω)

@[simp]
lemma cov_apply {Ω : Type*} [MeasureSpace Ω] (x y : Ω → ℝ) : cov x y = (∫ ω, x ω * y ω) - (∫ ω, x ω) * (∫ ω, y ω) := rfl

/-! ### Covariance algebra -/

/-- `cov(c * x, y) = c * cov(x, y)`. -/
lemma cov_const_mul_left {Ω : Type*} [MeasureSpace Ω] (c : ℝ) (x y : Ω → ℝ) :
    cov (fun ω => c * x ω) y = c * cov x y := by
  rw [cov, cov]
  simp_rw [mul_assoc]
  rw [integral_const_mul c (fun ω => x ω * y ω), integral_const_mul c (fun ω => x ω)]
  ring

/-- `cov(x + y, z) = cov(x, z) + cov(y, z)`. -/
lemma cov_add_left {Ω : Type*} [MeasureSpace Ω] (x y z : Ω → ℝ)
    (hx : Integrable (fun ω => x ω * z ω) volume)
    (hy : Integrable (fun ω => y ω * z ω) volume)
    (hx' : Integrable x volume) (hy' : Integrable y volume)
    (hz : Integrable z volume) :
    cov (fun ω => x ω + y ω) z = cov x z + cov y z := by
  rw [cov, cov, cov]
  simp_rw [add_mul]
  have h_int1 : (∫ ω, x ω * z ω + y ω * z ω) = (∫ ω, x ω * z ω) + (∫ ω, y ω * z ω) :=
    integral_add hx hy
  have h_int2 : (∫ ω, x ω + y ω) = (∫ ω, x ω) + (∫ ω, y ω) :=
    integral_add hx' hy'
  rw [h_int1, h_int2]
  ring

/-- `cov(x, c * y) = c * cov(x, y)`. -/
lemma cov_const_mul_right {Ω : Type*} [MeasureSpace Ω] (c : ℝ) (x y : Ω → ℝ) :
    cov x (fun ω => c * y ω) = c * cov x y := by
  rw [cov, cov]
  simp_rw [mul_comm (c : ℝ), ← mul_assoc]
  rw [integral_mul_const c (fun ω => x ω * y ω), integral_mul_const c (fun ω => y ω)]
  ring

/-- `cov(x, y + z) = cov(x, y) + cov(x, z)`. -/
lemma cov_add_right {Ω : Type*} [MeasureSpace Ω] (x y z : Ω → ℝ)
    (hx : Integrable (fun ω => x ω * y ω) volume)
    (hx' : Integrable (fun ω => x ω * z ω) volume)
    (hy : Integrable y volume) (hz : Integrable z volume) :
    cov x (fun ω => y ω + z ω) = cov x y + cov x z := by
  rw [cov, cov, cov]
  simp_rw [mul_add]
  have h_int1 : (∫ ω, x ω * y ω + x ω * z ω) = (∫ ω, x ω * y ω) + (∫ ω, x ω * z ω) :=
    integral_add hx hx'
  have h_int2 : (∫ ω, y ω + z ω) = (∫ ω, y ω) + (∫ ω, z ω) :=
    integral_add hy hz
  rw [h_int1, h_int2]
  ring

/-! ### Variance of a linear combination -/

/-- `var(c * x) = c² * var(x)`. -/
lemma var_const_mul {Ω : Type*} [MeasureSpace Ω] (c : ℝ) (x : Ω → ℝ) :
    var (fun ω => c * x ω) = c^2 * var x := by
  rw [var, var]
  have h_sq : (fun ω => (c * x ω) * (c * x ω)) = (fun ω => c^2 * (x ω * x ω)) := by
    ext ω; ring
  rw [h_sq]
  rw [integral_const_mul (c ^ 2) (fun ω => x ω * x ω), integral_const_mul c (fun ω => x ω)]
  ring

/-- `var(x + y) = var(x) + var(y) + 2·cov(x, y)`. -/
lemma var_add {Ω : Type*} [MeasureSpace Ω] (x y : Ω → ℝ)
    (hxx : Integrable (fun ω => x ω * x ω) volume)
    (hyy : Integrable (fun ω => y ω * y ω) volume)
    (hxy : Integrable (fun ω => x ω * y ω) volume)
    (hx : Integrable x volume) (hy : Integrable y volume) :
    var (fun ω => x ω + y ω) = var x + var y + 2 * cov x y := by
  rw [var, var, cov, var]
  simp_rw [add_mul, mul_add]
  have h_int_all : (∫ ω : Ω, x ω * x ω + x ω * y ω + (y ω * x ω + y ω * y ω)) =
      (∫ ω, x ω * x ω) + (∫ ω, x ω * y ω) + (∫ ω, y ω * x ω) + (∫ ω, y ω * y ω) := by
    calc
      (∫ ω, x ω * x ω + x ω * y ω + (y ω * x ω + y ω * y ω))
          = (∫ ω, (x ω * x ω + x ω * y ω) + (y ω * x ω + y ω * y ω)) := by ring
      _ = (∫ ω, x ω * x ω + x ω * y ω) + (∫ ω, y ω * x ω + y ω * y ω) := by
        have hyx : Integrable (fun ω => y ω * x ω) volume := by simpa [mul_comm] using hxy
        exact integral_add (hxx.add hxy) (hyx.add hyy)
      _ = ((∫ ω, x ω * x ω) + (∫ ω, x ω * y ω)) + ((∫ ω, y ω * x ω) + (∫ ω, y ω * y ω)) := by
        rw [integral_add hxx hxy, integral_add (by simpa [mul_comm] using hxy) hyy]
      _ = (∫ ω, x ω * x ω) + (∫ ω, x ω * y ω) + (∫ ω, y ω * x ω) + (∫ ω, y ω * y ω) := by ring
  have h_int_xy : (∫ ω : Ω, x ω + y ω) = (∫ ω, x ω) + (∫ ω, y ω) :=
    integral_add hx hy
  rw [h_int_all, h_int_xy]
  have h_int_yx_eq_xy : (∫ ω : Ω, y ω * x ω) = (∫ ω, x ω * y ω) := by
    simp_rw [mul_comm (y _) (x _)]
  rw [h_int_yx_eq_xy]
  ring_nf

/-! ### Return-level wrappers -/

/-- **Expected return** of asset `i`: the integral of `R_i` over the volume measure.

Defined as `E[R_i] = ∫ R_i d(volume)`. -/
noncomputable def expectedReturn {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (R : Ω → ReturnVec n) (i : Fin n) : ℝ :=
  ∫ ω, R ω i ∂volume

/-- **Return variance** of asset `i`: the variance of `R_i` under the volume measure.

Defined as `Var(R_i) = var (λ ω, R ω i)`. -/
noncomputable def returnVariance {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (R : Ω → ReturnVec n) (i : Fin n) : ℝ :=
  var (fun ω => R ω i)

/-- **Return covariance** between assets `i` and `j`: the covariance of `R_i` and `R_j`
under the volume measure.

Defined as `Cov(R_i, R_j) = cov (λ ω, R ω i) (λ ω, R ω j)`. -/
noncomputable def returnCovariance {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (R : Ω → ReturnVec n) (i j : Fin n) : ℝ :=
  cov (fun ω => R ω i) (fun ω => R ω j)

/-- **Return correlation** between assets `i` and `j`.

Defined as `Corr(R_i, R_j) = Cov(R_i, R_j) / √(Var(R_i) · Var(R_j))`. -/
noncomputable def returnCorrelation {n : ℕ} {Ω : Type*} [MeasureSpace Ω]
    (R : Ω → ReturnVec n) (i j : Fin n) : ℝ :=
  returnCovariance R i j / Real.sqrt (returnVariance R i * returnVariance R j)

/-- **Beta** of asset `i` with respect to the market portfolio `R_m`.

Defined as `β_i = Cov(R_i, R_m) / Var(R_m)`. -/
noncomputable def beta (covRRm varRm : ℝ) : ℝ :=
  covRRm / varRm

/-- If `g` is L² on a probability space and `var g = 0`, then `g` is a.e. constant, so its
covariance with any `f` vanishes. The L² and probability-measure hypotheses are essential:
without them `var g = 0` does not force `g` a.e. constant under Lean's integral conventions. -/
lemma cov_eq_zero_of_var_eq_zero {Ω : Type*} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (f g : Ω → ℝ) (hg : MemLp g 2 volume)
    (hvar : var g = 0) : cov f g = 0 := by
  have hvar_mathlib : ProbabilityTheory.variance g volume = 0 := by
    have h_eq : var g = ProbabilityTheory.variance g volume := by
      rw [var, ProbabilityTheory.variance_eq_sub hg]
      simp [sq]
    rw [← h_eq, hvar]
  have hae : g =ᵐ[volume] fun _ => (∫ ω, g ω) := by
    have h := ProbabilityTheory.ae_eq_integral_of_variance_eq_zero hg hvar_mathlib
    filter_upwards [h] with ω hω
    simpa using hω
  rw [cov]
  have h_int_fg : (∫ ω, f ω * g ω) = (∫ ω, f ω) * (∫ ω, g ω) := by
    calc
      (∫ ω, f ω * g ω) = (∫ ω, f ω * (fun _ => (∫ ω', g ω')) ω) := by
        refine integral_congr_ae ?_
        filter_upwards [hae] with ω hω
        rw [hω]
      _ = (∫ ω, f ω) * (∫ ω, g ω) := by rw [integral_mul_const]
  rw [h_int_fg]
  ring

/-- Contrapositive of `cov_eq_zero_of_var_eq_zero`: if `cov f g ≠ 0` then `var g ≠ 0`. -/
lemma cov_ne_zero_imp_var_ne_zero {Ω : Type*} [MeasureSpace Ω]
    [IsProbabilityMeasure (volume : Measure Ω)]
    (f g : Ω → ℝ) (hg : MemLp g 2 volume)
    (hcov : cov f g ≠ 0) : var g ≠ 0 :=
  fun hvar => hcov (cov_eq_zero_of_var_eq_zero f g hg hvar)

end PaperReconstructions
