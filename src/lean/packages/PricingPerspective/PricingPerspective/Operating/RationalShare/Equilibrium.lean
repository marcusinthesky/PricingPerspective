import Architect
import PricingPerspective.Operating.RationalShare.Lipschitz

set_option linter.style.longLine false

namespace PricingPerspective.Operating.RationalShare

open PricingPerspective.Operating.VariationalInequality

open Finset
open scoped RealInnerProductSpace

variable {κ ι α : Type*} [Fintype κ] [Fintype ι] [Fintype α] [DecidableEq ι]

/-! ### The capstone, and what it is conditional on

The equivalence above is unconditional, and so now is strong monotonicity: it is discharged from
the primitives by `stronglyMonotoneOn_pseudoGradient`, with the modulus supplied by own-cost
curvature. What the general statement below still carries as hypotheses is the Lipschitz bound and
the step-size conditions, in the same way `Transmission/CertificateRate.separationCeiling_rate`
carries its geometric-convergence premise. A reader can see exactly which half is done.

Note also that the raw Jacobian of a ratio payoff is **not** symmetric, and nothing above claims it
is. Monotonicity is a statement about its symmetric part, which is where `node_block_nonneg` acts;
asymmetry is not an obstruction to it. -/

/-- **Existence and uniqueness of a rational-share Nash equilibrium, conditional on the two
moduli.** The equivalence is the contribution; the moduli are premises.

What is *not* claimed: that the registered economy satisfies `hmono` and `hLip`. That is the open
connection the operating-spine manifest records, and it is a curvature question about the share
map, not an equilibrium question. Nor is the projected iteration behind `existsUnique_solvesVI`
the simultaneous proximal map the simulation runs -- it is a proof device, and nothing here is a
convergence theorem for those dynamics. -/
theorem existsUnique_isNash_of_moduli (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {lo hi : ι × α → ℝ} {mu L gamma : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a)
    (hpsd : ∀ i, ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hlo : ∀ p, 0 ≤ lo p) (hbox : ∀ p, lo p ≤ hi p)
    (hmu : 0 < mu)
    (hmono : StronglyMonotoneOn (pseudoGradient N c H) (box lo hi) mu)
    (hLip : ∀ a ∈ box lo hi, ∀ b ∈ box lo hi,
      ‖pseudoGradient N c H a - pseudoGradient N c H b‖ ≤ L * ‖a - b‖)
    (hgamma : 0 < gamma)
    (hq : 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 < 1)
    (hq0 : 0 ≤ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2) :
    ∃! v, IsNash N c H lo hi v := by
  have hne : (box lo hi).Nonempty := ⟨WithLp.toLp 2 lo, fun p => ⟨le_refl _, hbox p⟩⟩
  obtain ⟨v, hv, huniq⟩ := existsUnique_solvesVI (F := pseudoGradient N c H) hne
    (isComplete_box lo hi) (convex_box lo hi) hmu hmono hLip hgamma hq hq0
  refine ⟨v, isNash_of_solvesVI N c hsymm hpsd hlo hv, fun w hw => ?_⟩
  exact huniq w (solvesVI_of_isNash N c hsymm hpsd hlo hw)

/-- **The Lipschitz bound, discharged.** Cost and demand are bounded separately and added, each
by the Frobenius norm of its own coefficient array. Only finiteness is used downstream. -/
theorem lipschitzOn_pseudoGradient (N : Nodes κ α) (c : ι → α → ℝ) (H : ι → α → α → ℝ)
    {lo hi : ι × α → ℝ} (hlo : ∀ p, 0 ≤ lo p) :
    ∀ a ∈ box lo hi, ∀ b ∈ box lo hi,
      ‖pseudoGradient N c H a - pseudoGradient N c H b‖ ≤ lipConst N H * ‖a - b‖ := by
  intro x hxm y hym
  have hx := nonneg_of_mem_box hlo hxm
  have hy := nonneg_of_mem_box hlo hym
  have hsplit : pseudoGradient N c H x - pseudoGradient N c H y
      = (costField (ι := ι) c H x - costField (ι := ι) c H y)
        + (revenueField N x - revenueField N y) := by
    rw [pseudoGradient_eq_costField_add_revenueField N c H x,
      pseudoGradient_eq_costField_add_revenueField N c H y]
    abel
  have hcost := norm_le_frobenius
    (M := fun p q : ι × α => |if q.1 = p.1 then H p.1 p.2 q.2 else 0|)
    (v := costField (ι := ι) c H x - costField (ι := ι) c H y) (u := x - y)
    fun p => abs_costField_sub_le c H x y p
  have hrev := norm_le_frobenius (M := demandCoeff N)
    (v := revenueField N x - revenueField N y) (u := x - y)
    fun p => abs_revenueField_sub_le N hx hy p
  have hcf : Real.sqrt (∑ p : ι × α, ∑ q : ι × α,
      |if q.1 = p.1 then H p.1 p.2 q.2 else 0| ^ 2) = costFrobenius H :=
    congrArg Real.sqrt (Finset.sum_congr rfl fun p _ =>
      Finset.sum_congr rfl fun q _ => sq_abs _)
  rw [hcf] at hcost
  rw [hsplit, lipConst, add_mul]
  exact (norm_add_le _ _).trans (add_le_add hcost hrev)

/-! ### The capstone, from the primitives

Everything `existsUnique_solvesVI` needs is now discharged rather than assumed: monotonicity from
the demand-node inequality plus own-cost curvature, continuity from the Frobenius bound, and a
step size chosen so that **both** Zarantonello conditions hold. The step size is deliberately not
optimized -- it is a proof device, not the simulator's adjustment parameter -- and it is chosen to
need no relation between the two moduli.

`existsUnique_isNash_of_moduli` is kept as the general conditional statement. -/

/-- A curvature bound with a positive modulus is in particular a positive-semidefinite one. -/
theorem psd_of_curvature {H : ι → α → α → ℝ} {mu : ℝ} (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b) :
    ∀ i, ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, H i a b * u a * u b := fun i u =>
  (mul_nonneg hmu.le (Finset.sum_nonneg fun _ _ => sq_nonneg _)).trans (hH i u)

/-- **Existence and uniqueness of a rational-share Nash equilibrium, from the primitives, on the
orthant the simulation optimizes over.**

The hypotheses are all properties of the demand and cost data: nonnegative revenue, kernel and
linear cost, a strictly positive outside option, and a symmetric own-cost quadratic with a
strictly positive curvature modulus. No modulus of monotonicity or continuity is assumed, and no
step size has to be supplied.

`hcap` is not an economic restriction: it says the rectangle used inside the proof is wide enough
to contain every best response, and `isNash_iff_isNashOrthant` is what makes that harmless. What
is **not** claimed is anything about the simultaneous proximal map the simulation iterates, or
about the floating-point solution it computes. -/
theorem existsUnique_isNashOrthant (N : Nodes κ α) (c : ι → α → ℝ) {H : ι → α → α → ℝ}
    {hi : ι × α → ℝ} {mu : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) (hhi : ∀ p, 0 ≤ hi p)
    (hcap : ∀ p, 2 * totalRevenue N ≤ mu * hi p ^ 2) :
    ∃! v, IsNashOrthant N c H v := by
  have hLnn : 0 ≤ lipConst N H :=
    add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  set L := lipConst N H with hLdef
  set gamma := mu / (2 * (1 + mu ^ 2 + L ^ 2)) with hgdef
  have hden : (0:ℝ) < 2 * (1 + mu ^ 2 + L ^ 2) := by positivity
  have hgamma : 0 < gamma := div_pos hmu hden
  have h2gm : 2 * gamma * mu < 1 := by
    have hrw : 2 * gamma * mu = mu ^ 2 / (1 + mu ^ 2 + L ^ 2) := by
      rw [hgdef]
      field_simp
    rw [hrw, div_lt_one (by positivity)]
    nlinarith [sq_nonneg L]
  have hstep : gamma * L ^ 2 < 2 * mu := by
    rw [hgdef, div_mul_eq_mul_div, div_lt_iff₀ hden]
    nlinarith [sq_nonneg L, hmu, sq_nonneg mu]
  have hq : 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 < 1 := by
    have hlt : gamma ^ 2 * L ^ 2 < 2 * gamma * mu := by
      have hfac : gamma ^ 2 * L ^ 2 = gamma * (gamma * L ^ 2) := by ring
      rw [hfac]
      calc gamma * (gamma * L ^ 2) < gamma * (2 * mu) :=
            mul_lt_mul_of_pos_left hstep hgamma
        _ = 2 * gamma * mu := by ring
    linarith
  have hq0 : 0 ≤ 1 - 2 * gamma * mu + gamma ^ 2 * L ^ 2 := by
    nlinarith [h2gm, sq_nonneg (gamma * L)]
  have hpsd := psd_of_curvature hmu hH
  have hmono := stronglyMonotoneOn_pseudoGradient N c (lo := fun _ => (0:ℝ)) (hi := hi)
    (fun _ => le_refl 0) hH
  have hlip := lipschitzOn_pseudoGradient N c H (lo := fun _ => (0:ℝ)) (hi := hi)
    fun _ => le_refl 0
  obtain ⟨v, hv, huniq⟩ := existsUnique_isNash_of_moduli N c hsymm hpsd
    (fun _ => le_refl 0) hhi hmu hmono hlip hgamma hq hq0
  refine ⟨v, (isNash_iff_isNashOrthant N c hmu hc hH hhi hcap).1 hv, fun w hw => ?_⟩
  exact huniq w ((isNash_iff_isNashOrthant N c hmu hc hH hhi hcap).2 hw)

/-- A cap wide enough to hold every best response always exists, because revenue is finite and
curvature is strictly positive. This is what lets `hcap` disappear from the statement below. -/
theorem exists_wide_cap (N : Nodes κ α) {mu : ℝ} (hmu : 0 < mu) :
    ∃ hi : ι × α → ℝ, (∀ p, 0 ≤ hi p) ∧ ∀ p, 2 * totalRevenue N ≤ mu * hi p ^ 2 := by
  have hmu0 : mu ≠ 0 := ne_of_gt hmu
  have hnn : 0 ≤ 2 * totalRevenue N / mu :=
    div_nonneg (by linarith [totalRevenue_nonneg N]) hmu.le
  have hcancel : mu * (2 * totalRevenue N / mu) = 2 * totalRevenue N := by field_simp
  refine ⟨fun _ => Real.sqrt (2 * totalRevenue N / mu), fun _ => Real.sqrt_nonneg _,
    fun _ => ?_⟩
  rw [Real.sq_sqrt hnn, hcancel]

/-- **The general theorem, stated only in the primitives.**

Every rational-share game whose demand side has nonnegative revenue and kernel and a strictly
positive outside option, and whose own costs are nonnegative-linear plus a symmetric quadratic
with a strictly positive curvature modulus, has a **unique** Nash equilibrium on the nonnegative
capacity orthant.

No modulus of monotonicity, no Lipschitz constant, no step size and no capacity cap appear. This
is the statement downstream work should cite. It says nothing about the simultaneous proximal map
the simulation iterates, nor about the floating-point profile that map converges to. -/
@[blueprint "prop:equilibrium"
  (statement := /--
    Unique Nash equilibrium of the rational-share capacity economy under primitive curvature
    conditions. Assumptions: Finite fixed sites, nonnegative opportunities and costs, positive
    outside appeal and own curvature.
  -/)]
theorem existsUnique_isNashOrthant_of_primitives (N : Nodes κ α) (c : ι → α → ℝ)
    {H : ι → α → α → ℝ} {mu : ℝ}
    (hsymm : ∀ i a b, H i a b = H i b a) (hmu : 0 < mu)
    (hH : ∀ i, ∀ u : α → ℝ, mu * ∑ a, u a ^ 2 ≤ ∑ a, ∑ b, H i a b * u a * u b)
    (hc : ∀ i a, 0 ≤ c i a) :
    ∃! v, IsNashOrthant N c H v := by
  obtain ⟨hi, hhi, hcap⟩ := exists_wide_cap (ι := ι) N hmu
  exact existsUnique_isNashOrthant N c hsymm hmu hH hc hhi hcap

/-! ### The registered cost shape

The Python economy does not hand the theorem an arbitrary matrix. It builds
`H i = congestion • K + own • I` with `K` the congestion kernel over sites and `own` the
configured `own_curvature`, which the configuration requires to be strictly positive. The two
lemmas below discharge the capstone's cost hypotheses from that construction, leaving exactly one
premise: that `K` is positive semidefinite.

That premise is **not** proved here. `K` is a Gaussian kernel matrix, and its positive
semidefiniteness is a real theorem -- the Schur-product route via `Matrix.PosSemidef.hadamard`
applied to the exponential series is available in Mathlib but is a separate development. Until it
lands, the registered-model connection is open, and this is the single named thing that is
missing. -/

/-- The registered cost matrix: a congestion kernel plus own curvature. -/
def registeredCost [DecidableEq α] (K : α → α → ℝ) (congestion own : ℝ) (_i : ι) (a b : α) : ℝ :=
  congestion * K a b + own * (if a = b then 1 else 0)

omit [Fintype ι] [DecidableEq ι] in
theorem registeredCost_symm [DecidableEq α] {K : α → α → ℝ} (hK : ∀ a b, K a b = K b a)
    (congestion own : ℝ) (i : ι) (a b : α) :
    registeredCost K congestion own i a b = registeredCost K congestion own i b a := by
  simp only [registeredCost, hK a b]
  by_cases h : a = b
  · rw [h]
  · rw [if_neg h, if_neg (Ne.symm h)]

omit [Fintype ι] [DecidableEq ι] in
/-- **The registered cost supplies the modulus.** A positive-semidefinite congestion kernel
contributes nothing negative, and own curvature contributes exactly its own coefficient. -/
theorem registeredCost_curvature [DecidableEq α] {K : α → α → ℝ} {congestion own : ℝ}
    (hcong : 0 ≤ congestion) (hK : ∀ u : α → ℝ, 0 ≤ ∑ a, ∑ b, K a b * u a * u b) (i : ι)
    (u : α → ℝ) :
    own * ∑ a, u a ^ 2
      ≤ ∑ a, ∑ b, registeredCost K congestion own i a b * u a * u b := by
  have inner : ∀ a : α, (∑ b, registeredCost K congestion own i a b * u a * u b)
      = congestion * (∑ b, K a b * u a * u b) + own * u a ^ 2 := by
    intro a
    have expand : ∀ b : α, registeredCost K congestion own i a b * u a * u b
        = congestion * (K a b * u a * u b) + own * (if a = b then u a * u b else 0) := by
      intro b
      simp only [registeredCost]
      by_cases h : a = b
      · rw [if_pos h, if_pos h]; ring
      · rw [if_neg h, if_neg h]; ring
    rw [Finset.sum_congr rfl fun b _ => expand b, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, Finset.sum_ite_eq (Finset.univ : Finset α) a fun b => u a * u b]
    rw [if_pos (Finset.mem_univ a)]
    ring
  have hsplit : (∑ a, ∑ b, registeredCost K congestion own i a b * u a * u b)
      = congestion * (∑ a, ∑ b, K a b * u a * u b) + own * ∑ a, u a ^ 2 := by
    rw [Finset.sum_congr rfl fun a _ => inner a, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum]
  rw [hsplit]
  have := mul_nonneg hcong (hK u)
  linarith

end PricingPerspective.Operating.RationalShare
