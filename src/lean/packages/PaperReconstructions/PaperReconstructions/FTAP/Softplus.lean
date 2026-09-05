import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.Algebra.GroupWithZero
import Mathlib.Topology.EMetricSpace.Lipschitz
import Mathlib.Data.NNReal.Defs
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum

/-!
# The softplus penalty and its logistic derivative

`softplus u = log (1 + eᵘ)` is the smooth convex penalty used to build the equivalent
martingale measure in the one-period FTAP, and `logistic u = eᵘ / (1 + eᵘ) = σ(u)` is its
derivative.

Three properties carry the FTAP argument:

* `posPart_le_softplus` — `u⁺ ≤ softplus u`, the lower bound powering **coercivity** of
  the potential;
* `softplus_le` — `softplus u ≤ |u| + log 2`, the linear-growth upper bound giving
  `L¹`-**integrability**;
* `logistic_pos` / `logistic_lt_one` — `σ ∈ (0, 1)`, giving both the uniform `L¹`
  domination for differentiating under the integral and the strict positivity that makes
  the resulting density define an *equivalent* measure.

## Provenance

Ported from [formal-mathfin](https://github.com/raphaelrrcoelho/formal-mathfin)
(`MathFin/Foundations/FTAPOnePeriodVector.lean`, Apache-2.0, © Raphael Coelho).
-/

namespace PaperReconstructions.FTAP

/-- Softplus penalty `log (1 + eᵘ)`. -/
noncomputable def softplus (u : ℝ) : ℝ := Real.log (1 + Real.exp u)

/-- Logistic function `eᵘ / (1 + eᵘ) = σ(u)`, the derivative of `softplus`. -/
noncomputable def logistic (u : ℝ) : ℝ := Real.exp u / (1 + Real.exp u)

lemma logistic_pos (u : ℝ) : 0 < logistic u := by rw [logistic]; positivity

lemma logistic_lt_one (u : ℝ) : logistic u < 1 := by
  rw [logistic, div_lt_one (by positivity)]; linarith [Real.exp_pos u]

lemma softplus_nonneg (u : ℝ) : 0 ≤ softplus u := by
  rw [softplus]; exact Real.log_nonneg (by linarith [Real.exp_pos u])

lemma self_le_softplus (u : ℝ) : u ≤ softplus u := by
  rw [softplus]
  calc u = Real.log (Real.exp u) := (Real.log_exp u).symm
    _ ≤ Real.log (1 + Real.exp u) :=
        Real.log_le_log (Real.exp_pos u) (by linarith [Real.exp_pos u])

/-- `u⁺ = max u 0 ≤ softplus u`: the lower bound powering coercivity of the potential. -/
lemma posPart_le_softplus (u : ℝ) : max u 0 ≤ softplus u :=
  max_le (self_le_softplus u) (softplus_nonneg u)

/-- `softplus u ≤ |u| + log 2`: the linear-growth upper bound giving `L¹`-integrability. -/
lemma softplus_le (u : ℝ) : softplus u ≤ |u| + Real.log 2 := by
  rw [softplus]
  have hb : 1 + Real.exp u ≤ 2 * Real.exp |u| := by
    have h1 : Real.exp u ≤ Real.exp |u| := Real.exp_le_exp.mpr (le_abs_self u)
    have h2 : (1 : ℝ) ≤ Real.exp |u| := Real.one_le_exp (abs_nonneg u)
    linarith
  calc Real.log (1 + Real.exp u) ≤ Real.log (2 * Real.exp |u|) :=
        Real.log_le_log (by positivity) hb
    _ = |u| + Real.log 2 := by
        rw [Real.log_mul (by norm_num) (Real.exp_ne_zero _), Real.log_exp]; ring

/-- `softplus` is differentiable with derivative the logistic `σ`. -/
lemma hasDerivAt_softplus (u : ℝ) : HasDerivAt softplus (logistic u) u := by
  have h2 : (1 : ℝ) + Real.exp u ≠ 0 := by positivity
  have h1 : HasDerivAt (fun v => 1 + Real.exp v) (Real.exp u) u := by
    simpa using (Real.hasDerivAt_exp u).const_add 1
  have h3 := (Real.hasDerivAt_log h2).comp u h1
  rw [show logistic u = (1 + Real.exp u)⁻¹ * Real.exp u from by
    rw [logistic, div_eq_inv_mul]]
  exact h3

lemma continuous_softplus : Continuous softplus := by
  have hpos : ∀ u : ℝ, (0 : ℝ) < 1 + Real.exp u := fun u => by positivity
  exact (continuous_const.add Real.continuous_exp).log (fun u => (hpos u).ne')

lemma continuous_logistic : Continuous logistic := by
  have hpos : ∀ u : ℝ, (0 : ℝ) < 1 + Real.exp u := fun u => by positivity
  exact Real.continuous_exp.div (continuous_const.add Real.continuous_exp)
    (fun u => (hpos u).ne')

/-- `softplus` is `1`-Lipschitz (its derivative `σ` lies in `(0, 1)`). -/
lemma lipschitzWith_softplus : LipschitzWith 1 softplus := by
  refine lipschitzWith_of_nnnorm_deriv_le
    (fun x => (hasDerivAt_softplus x).differentiableAt) fun x => ?_
  rw [(hasDerivAt_softplus x).deriv, ← NNReal.coe_le_coe, coe_nnnorm, NNReal.coe_one,
    Real.norm_eq_abs, abs_of_pos (logistic_pos x)]
  exact (logistic_lt_one x).le

/-- `s ↦ max s 0` (positive part) is `1`-Lipschitz. -/
lemma lipschitzWith_posPart : LipschitzWith 1 (fun s : ℝ => max s 0) :=
  LipschitzWith.id.max_const 0

end PaperReconstructions.FTAP
