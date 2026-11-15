import Architect
import PricingPerspective.Operating.RationalShare.EquilibriumResponse.OrthantComparativeStatics
import Mathlib.Probability.Moments.Covariance

/-!
# How the rational-share equilibrium responds to opportunity values

The existing comparative statics compare two economies through a generic operator perturbation.
This module specializes that comparison to the object the paper moves: the opportunity values
`r`, with demand geometry, outside appeal, and costs held fixed. It also records the
cash-flow and composition facts that the response operator is reported through.

* **Opportunity continuity.** Only the revenue weights change, so the pseudo-gradient moves by an
  opportunity-loading matrix applied to `r' - r`. Its coefficients are bounded by
  `kernel / outside`, so orthant equilibria move by at most
  `√(card ι * Σ_l Σ_a kernel² / outside²) * ‖r' - r‖ / mu`. No differentiability, interiority, or
  active-set condition is used.
* **Cash-flow exposure.** At fixed capacities, payoff is affine in `r`, and the slope on node `l`
  is the firm's share `appeal / denom` of that node. A random opportunity innovation therefore
  has covariance `S Ω Sᵀ`. A rival's capacity expansion weakly lowers the firm's payoff; this is
  the finite-difference form of the nonpositive cross-derivative.
* **Composition.** Normalizing a capacity block to its composition has directional derivative
  `(h - p Σh) / S`, which annihilates proportional scale changes exactly.
-/

namespace PricingPerspective.Operating.RationalShare.Nodes

open PricingPerspective.Operating.VariationalInequality
  PricingPerspective.Operating.RationalShare.EquilibriumResponse

open Finset
open scoped RealInnerProductSpace ProbabilityTheory

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-- The same demand geometry and outside appeal with new opportunity values. -/
def withRevenue (N : Nodes κ α) (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l) : Nodes κ α :=
  { N with revenue := r, revenue_nonneg := hr }

variable (N : Nodes κ α) (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l)

@[simp] theorem withRevenue_revenue : (N.withRevenue r hr).revenue = r := rfl

@[simp] theorem withRevenue_kernel : (N.withRevenue r hr).kernel = N.kernel := rfl

@[simp] theorem withRevenue_outside : (N.withRevenue r hr).outside = N.outside := rfl

@[simp] theorem withRevenue_appeal (x : Profile ι α) (i : ι) (l : κ) :
    (N.withRevenue r hr).appeal x i l = N.appeal x i l := rfl

@[simp] theorem withRevenue_denom (x : Profile ι α) (l : κ) :
    (N.withRevenue r hr).denom x l = N.denom x l := rfl

@[simp] theorem withRevenue_rival (x : Profile ι α) (i : ι) (l : κ) :
    (N.withRevenue r hr).rival x i l = N.rival x i l := rfl

/-- The loading of opportunity `l` on coordinate `(i, a)` of the negative marginal-profit
field: raising `r_l` by one unit lowers that coordinate by exactly this amount. -/
noncomputable def opportunityLoading (x : Profile ι α) (i : ι) (a : α) (l : κ) : ℝ :=
  N.rival x i l / N.denom x l ^ 2 * N.kernel l a

/-- Firm `i`'s share of node `l`: its operating-cash-flow exposure to that opportunity. -/
noncomputable def share (x : Profile ι α) (i : ι) (l : κ) : ℝ :=
  N.appeal x i l / N.denom x l

theorem outside_le_rival {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (l : κ) :
    N.outside l ≤ N.rival x i l := by
  have hle : N.appeal x i l ≤ ∑ j, N.appeal x j l :=
    Finset.single_le_sum (fun j _ ↦ N.appeal_nonneg hx j l) (Finset.mem_univ i)
  rw [rival, denom]
  linarith

theorem opportunityLoading_nonneg {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (a : α)
    (l : κ) : 0 ≤ N.opportunityLoading x i a l :=
  mul_nonneg (div_nonneg (N.rival_nonneg hx i l) (sq_nonneg _)) (N.kernel_nonneg l a)

/-- Each loading is at most `kernel / outside`: the rival field is at most the denominator,
and the denominator is at least the outside option. -/
theorem opportunityLoading_le {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (a : α) (l : κ) :
    N.opportunityLoading x i a l ≤ N.kernel l a / N.outside l := by
  have hb := N.outside_pos l
  have hD := N.denom_pos hx l
  have hrivD : N.rival x i l ≤ N.denom x l := by
    rw [rival]; linarith [N.appeal_nonneg hx i l]
  have hbD : N.outside l ≤ N.denom x l := (N.outside_le_rival hx i l).trans hrivD
  have hcoef : N.rival x i l / N.denom x l ^ 2 ≤ 1 / N.outside l := by
    rw [div_le_div_iff₀ (by positivity) hb]
    nlinarith [N.rival_nonneg hx i l]
  calc N.opportunityLoading x i a l
      = N.rival x i l / N.denom x l ^ 2 * N.kernel l a := rfl
    _ ≤ 1 / N.outside l * N.kernel l a :=
        mul_le_mul_of_nonneg_right hcoef (N.kernel_nonneg l a)
    _ = N.kernel l a / N.outside l := by ring

theorem share_nonneg {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) (i : ι) (l : κ) :
    0 ≤ N.share x i l :=
  div_nonneg (N.appeal_nonneg hx i l) (N.denom_pos hx l).le

end PricingPerspective.Operating.RationalShare.Nodes

namespace PricingPerspective.Operating.RationalShare.EquilibriumResponse

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace ProbabilityTheory

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α]

/-! ### Opportunity continuity -/

/-- Changing only opportunity values moves the negative marginal-profit field by the
opportunity-loading matrix applied to the change. -/
theorem pseudoGradient_withRevenue_sub (N : Nodes κ α) (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l)
    (c : ι → α → ℝ) (H : ι → α → α → ℝ) (x : Profile ι α) (i : ι) (a : α) :
    (pseudoGradient (N.withRevenue r hr) c H x - pseudoGradient N c H x) (i, a) =
      -∑ l, (r l - N.revenue l) * N.opportunityLoading x i a l := by
  simp only [PiLp.sub_apply, pseudoGradient_apply, Nodes.withRevenue_revenue,
    Nodes.withRevenue_rival, Nodes.withRevenue_denom, Nodes.withRevenue_kernel,
    Nodes.opportunityLoading]
  rw [← Finset.sum_neg_distrib, sub_sub_sub_cancel_left, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  ring

variable (ι) in
/-- The squared opportunity modulus `card ι * Σ_l Σ_a kernel² / outside²`. -/
noncomputable def opportunityModulusSq (N : Nodes κ α) : ℝ :=
  Fintype.card ι * ∑ l, ∑ a, N.kernel l a ^ 2 / N.outside l ^ 2

theorem opportunityModulusSq_nonneg (N : Nodes κ α) : 0 ≤ opportunityModulusSq ι N :=
  mul_nonneg (Nat.cast_nonneg _)
    (Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ by positivity)

/-- **The field perturbation is Lipschitz in opportunity values**, uniformly over the capacity
orthant. -/
theorem norm_sq_pseudoGradient_withRevenue_sub_le (N : Nodes κ α) (r : κ → ℝ)
    (hr : ∀ l, 0 ≤ r l) (c : ι → α → ℝ) (H : ι → α → α → ℝ) {x : Profile ι α}
    (hx : ∀ p, 0 ≤ x p) :
    ‖pseudoGradient (N.withRevenue r hr) c H x - pseudoGradient N c H x‖ ^ 2 ≤
      opportunityModulusSq ι N * ∑ l, (r l - N.revenue l) ^ 2 := by
  set δ : κ → ℝ := fun l ↦ r l - N.revenue l with hδ
  have hterm : ∀ i a,
      (pseudoGradient (N.withRevenue r hr) c H x - pseudoGradient N c H x) (i, a) ^ 2 ≤
        (∑ l, δ l ^ 2) * ∑ l, N.kernel l a ^ 2 / N.outside l ^ 2 := by
    intro i a
    rw [pseudoGradient_withRevenue_sub, neg_sq]
    refine (Finset.sum_mul_sq_le_sq_mul_sq _ _ _).trans
      (mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun l _ ↦ ?_)
        (Finset.sum_nonneg fun _ _ ↦ sq_nonneg _))
    have h0 := N.opportunityLoading_nonneg hx i a l
    have h1 := N.opportunityLoading_le hx i a l
    rw [← div_pow]
    exact pow_le_pow_left₀ h0 h1 2
  rw [norm_sq_eq_sum, Fintype.sum_prod_type]
  calc ∑ i, ∑ a, (pseudoGradient (N.withRevenue r hr) c H x - pseudoGradient N c H x) (i, a) ^ 2
      ≤ ∑ _i : ι, ∑ a, (∑ l, δ l ^ 2) * ∑ l, N.kernel l a ^ 2 / N.outside l ^ 2 :=
        Finset.sum_le_sum fun i _ ↦ Finset.sum_le_sum fun a _ ↦ hterm i a
    _ = opportunityModulusSq ι N * ∑ l, δ l ^ 2 := by
        rw [opportunityModulusSq, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
          ← Finset.mul_sum, Finset.sum_comm (f := fun a l ↦ N.kernel l a ^ 2 / N.outside l ^ 2)]
        ring

/-- **Opportunity continuity of the equilibrium.** Orthant equilibria under opportunity values
`N.revenue` and `r` differ by at most `√(opportunityModulusSq) * ‖r - N.revenue‖ / mu`.

This applies established strongly monotone VI sensitivity to the rational-share field. It needs
no differentiability, interiority, or fixed active set. -/
@[blueprint "eq:response-continuity"
  (statement := /--
    Equilibrium capacities are Lipschitz in opportunity values with constant $L_r/\nu$.
    Assumptions: Nash equilibria of two rational-share economies that differ only in nonnegative
    opportunity values, with symmetric uniformly positive cost curvature and nonnegative linear
    costs.
  -/)]
theorem orthantEquilibrium_dist_le_opportunityShift [DecidableEq ι]
    (N : Nodes κ α) (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {mu : ℝ} {x y : Profile ι α}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a)
    (hx : IsNashOrthant N c H x) (hy : IsNashOrthant (N.withRevenue r hr) c H y) :
    ‖y - x‖ ≤
      Real.sqrt (opportunityModulusSq ι N) * Real.sqrt (∑ l, (r l - N.revenue l) ^ 2) / mu := by
  have hbase := orthantEquilibrium_dist_le_operatorPerturbation N (N.withRevenue r hr) c c
    hsymm hsymm hmu hmu hH hH hc hc hx hy
  refine hbase.trans (div_le_div_of_nonneg_right ?_ hmu.le)
  rw [← Real.sqrt_mul (opportunityModulusSq_nonneg N),
    ← Real.sqrt_sq (norm_nonneg (pseudoGradient (N.withRevenue r hr) c H x -
      pseudoGradient N c H x))]
  exact Real.sqrt_le_sqrt (norm_sq_pseudoGradient_withRevenue_sub_le N r hr c H hx.1)

/-! ### Cash-flow exposure -/

/-- **Payoff is affine in opportunity values at fixed capacities**, with slope equal to the
firm's node shares. The exposure of operating cash flow to opportunity `l` is the share of `l`
that the firm captures. -/
@[blueprint "cor:exposure"
  (statement := /--
    At fixed capacities, payoff is affine in opportunity values with slopes equal to node
    shares. Assumptions: Fixed capacities and two opportunity-value vectors.
  -/)]
theorem payoff_withRevenue_sub (N : Nodes κ α) (r : κ → ℝ) (hr : ∀ l, 0 ≤ r l)
    (c : ι → α → ℝ) (H : ι → α → α → ℝ) (i : ι) (x : Profile ι α) :
    payoff (N.withRevenue r hr) c H i x - payoff N c H i x =
      ∑ l, (r l - N.revenue l) * N.share x i l := by
  simp only [payoff, Nodes.withRevenue_revenue, Nodes.withRevenue_appeal,
    Nodes.withRevenue_denom, Nodes.share]
  rw [sub_sub_sub_cancel_right, sub_sub_sub_cancel_right, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  ring

omit [Fintype ι] in
/-- **Exposure covariance.** Opportunity innovations `ε` with square-integrable coordinates,
passed through a predetermined share matrix, produce cash-flow changes with covariance
`Σ_l Σ_m s_il s_jm cov[ε_l, ε_m]`, that is `S Ω Sᵀ`. -/
@[blueprint "eq:exposure-covariance"
  (statement := /--
    Share-weighted opportunity innovations have covariance $S\Omega S^\top$. Assumptions:
    Square-integrable innovations and a predetermined share matrix.
  -/)]
theorem covariance_exposure {Ω : Type*} {mΩ : MeasurableSpace Ω}
    {μ : MeasureTheory.Measure Ω} [MeasureTheory.IsFiniteMeasure μ]
    (s : ι → κ → ℝ) (ε : κ → Ω → ℝ) (hε : ∀ l, MeasureTheory.MemLp (ε l) 2 μ) (i j : ι) :
    cov[fun ω ↦ ∑ l, s i l * ε l ω, fun ω ↦ ∑ m, s j m * ε m ω; μ] =
      ∑ l, ∑ m, s i l * s j m * cov[ε l, ε m; μ] := by
  rw [ProbabilityTheory.covariance_fun_sum_fun_sum (fun l ↦ (hε l).const_mul (s i l))
    (fun m ↦ (hε m).const_mul (s j m))]
  refine Finset.sum_congr rfl fun l _ ↦ Finset.sum_congr rfl fun m _ ↦ ?_
  rw [ProbabilityTheory.covariance_const_mul_left, ProbabilityTheory.covariance_const_mul_right]
  ring

/-- **Rival expansion weakly lowers a firm's payoff.** If a rival `j ≠ i` weakly raises every
capacity, firm `i`'s own appeal and costs are unchanged while every denominator weakly rises.
This is the finite-difference form of `∂π_i/∂q_j ≤ 0`. -/
@[blueprint "eq:rival-cross"
  (statement := /--
    A rival's weak capacity expansion weakly lowers a firm's payoff. Assumptions: A nonnegative
    profile and a rival whose capacities weakly rise.
  -/)]
theorem payoff_replaceFirm_rival_le [DecidableEq ι] (N : Nodes κ α) (c : ι → α → ℝ)
    (H : ι → α → α → ℝ) {x : Profile ι α} (hx : ∀ p, 0 ≤ x p) {i j : ι} (hji : j ≠ i)
    {q : α → ℝ} (hq : ∀ a, x (j, a) ≤ q a) :
    payoff N c H i (replaceFirm x j q) ≤ payoff N c H i x := by
  have hown : ∀ a, replaceFirm x j q (i, a) = x (i, a) := fun a ↦
    replaceFirm_of_ne x (Ne.symm hji) q a
  have happ : ∀ l, N.appeal (replaceFirm x j q) i l = N.appeal x i l := fun l ↦
    N.appeal_replaceFirm_of_ne x (Ne.symm hji) q l
  have hshift : ∀ l, 0 ≤ N.blockAppeal q l - N.appeal x j l := by
    intro l
    rw [Nodes.blockAppeal, Nodes.appeal, ← Finset.sum_sub_distrib]
    exact Finset.sum_nonneg fun a _ ↦ by
      rw [← mul_sub]; exact mul_nonneg (N.kernel_nonneg l a) (sub_nonneg.mpr (hq a))
  have hden : ∀ l, N.denom x l ≤ N.denom (replaceFirm x j q) l := fun l ↦ by
    rw [N.denom_replaceFirm x j q l]; linarith [hshift l]
  simp only [payoff, hown, happ]
  have hrev : ∑ l, N.revenue l * N.appeal x i l / N.denom (replaceFirm x j q) l ≤
      ∑ l, N.revenue l * N.appeal x i l / N.denom x l :=
    Finset.sum_le_sum fun l _ ↦
      div_le_div_of_nonneg_left
        (mul_nonneg (N.revenue_nonneg l) (N.appeal_nonneg hx i l)) (N.denom_pos hx l) (hden l)
  linarith

/-! ### Composition -/

/-- **The normalization derivative.** Along a direction `h`, the composition
`(q + t h) / Σ(q + t h)` has derivative `(h a - p a Σh) / S` at `t = 0`, where `S = Σq > 0` and
`p = q / S`. -/
@[blueprint "eq:composition-derivative"
  (statement := /--
    The derivative of composition with respect to capacity removes the proportional scale
    direction. Assumptions: A capacity block with positive total mass.
  -/)]
theorem hasDerivAt_composition {q h : α → ℝ} (hS : 0 < ∑ b, q b) (a : α) :
    HasDerivAt (fun t : ℝ ↦ (q a + t * h a) / ∑ b, (q b + t * h b))
      ((h a - q a / (∑ b, q b) * ∑ b, h b) / ∑ b, q b) 0 := by
  have hnum : HasDerivAt (fun t : ℝ ↦ q a + t * h a) (h a) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (h a)).const_add (q a)
  have hden : HasDerivAt (fun t : ℝ ↦ ∑ b, (q b + t * h b)) (∑ b, h b) 0 := by
    have := HasDerivAt.fun_sum (u := Finset.univ) (A := fun b t ↦ q b + t * h b)
      (A' := fun b ↦ h b) (x := (0 : ℝ))
      (fun b _ ↦ by simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (h b)).const_add (q b))
    simpa using this
  have hne : (∑ b, (q b + (0 : ℝ) * h b)) ≠ 0 := by simpa using hS.ne'
  refine (hnum.div hden hne).congr_deriv ?_
  simp only [zero_mul, add_zero]
  field_simp

/-- **Scale is annihilated exactly.** The normalization derivative in the direction of the
capacity block itself is zero: proportional expansion leaves composition unchanged. -/
theorem composition_derivative_self {q : α → ℝ} (hS : 0 < ∑ b, q b) (a : α) :
    (q a - q a / (∑ b, q b) * ∑ b, q b) / ∑ b, q b = 0 := by
  rw [div_mul_cancel₀ _ hS.ne', sub_self, zero_div]

end PricingPerspective.Operating.RationalShare.EquilibriumResponse
