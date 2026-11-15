import PricingPerspective.Operating.VariationalInequality.Monotone

set_option linter.style.longLine false

/-!
# The proximal decision rule: fixed points, and when it contracts

The operating dynamics are the *simultaneous* proximal map: every firm responds to the same
incumbent profile, penalised by a Euclidean adjustment cost `c / 2 * ‖z - x‖ ^ 2`. Two
questions about it are independent, and this module keeps them apart.

**Where it can rest.** Its fixed points are exactly the equilibria of the unpenalised game,
because the penalty gradient vanishes when the response is evaluated at the incumbent
profile itself. The penalty therefore changes the *path*, never the destination.

**Whether it converges.** That needs a contraction estimate, and the estimate needs a bound
on cross-sensitivity that uniqueness does not require. `proximal_response_dist_le` derives
the familiar `(c + L) / (c + mu)` factor from the two variational inequalities alone.

The consequence worth stating plainly is `contractionFactor_le_of_le`: raising the
adjustment weight `c` moves the factor *towards one*, never away from it, whenever the
game already contracts. So inertia cannot rescue *this sufficient bound* when `mu <= L`.

That is a statement about the bound, not about the dynamics. Inertia can and does stabilise
games that diverge without it: for three firms with
`pi_i = r q_i - q_i^2/2 - (3/4) q_i sum_(j != i) q_j`, the proximal Jacobian at the interior
symmetric equilibrium has spectral radius `3/2` at `c = 0` and `5/6` at `c = 1/2`, and it is
symmetric, so the projected map is a genuine contraction at the larger weight while the
row-sum condition fails at both. Nothing below rules that out.

## Main results

* `proxSolvesVI_self_iff` — proximal fixed points are exactly the variational-inequality
  solutions of the underlying game.
* `proximal_response_dist_le` — the one-step estimate `(c + mu) ‖d‖ ≤ (c + L) * d`.
* `contractionFactor_lt_one_iff` — the factor is below one exactly when `L < mu`.
* `contractionFactor_le_of_le` — the factor is nondecreasing in the adjustment weight.
-/

namespace PricingPerspective.Operating.VariationalInequality

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {F : E → E} {s : Set E} {c mu L d : ℝ} {x : E}

/-- `y` is the proximal response to incumbent `x` for the game map `F` on `s`. -/
def ProxSolvesVI (F : E → E) (s : Set E) (c : ℝ) (x y : E) : Prop :=
  y ∈ s ∧ ∀ z ∈ s, 0 ≤ ⟪F y + c • (y - x), z - y⟫

/-- **Fixed points are equilibria, and equilibria are fixed points.**

The adjustment penalty is evaluated at the incumbent profile, so its gradient vanishes
exactly where the response coincides with that profile. The penalty can slow the path to
an equilibrium; it cannot create one, remove one, or move one. -/
theorem proxSolvesVI_self_iff : ProxSolvesVI F s c x x ↔ SolvesVI F s x := by
  constructor <;> intro h <;> exact ⟨h.1, by simpa using h.2⟩

/-- **The one-step contraction estimate.**

Firm-level reading: `g` is the firm's own negative gradient at its own rival profile and
`gt` the same at the perturbed rival profile, so `hcross` is the cross-sensitivity budget
and `hx` the incumbent displacement. Testing each response's variational inequality with
the other and adding leaves exactly the own-monotonicity, penalty and cross terms.

Nothing here assumes the responses came from an optimisation: only that each satisfies its
proximal variational inequality. -/
theorem proximal_response_dist_le {g gt : E → E} {xi xti yi yti : E}
    (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hy : ProxSolvesVI g s c xi yi) (hyt : ProxSolvesVI gt s c xti yti)
    (hmono : mu * ‖yi - yti‖ ^ 2 ≤ ⟪g yi - g yti, yi - yti⟫)
    (hcross : ‖g yti - gt yti‖ ≤ L * d) (hx : ‖xi - xti‖ ≤ d) :
    (c + mu) * ‖yi - yti‖ ≤ (c + L) * d := by
  set w : E := yi - yti with hw
  have h1 := hy.2 yti hyt.1
  have h2 := hyt.2 yi hy.1
  have e1 : ⟪g yi + c • (yi - xi), yti - yi⟫ = -⟪g yi + c • (yi - xi), w⟫ := by
    rw [hw, show yti - yi = -(yi - yti) by abel, inner_neg_right]
  rw [e1] at h1
  rw [show yi - yti = w from hw.symm] at h2
  have expand : ⟪g yi + c • (yi - xi), w⟫ - ⟪gt yti + c • (yti - xti), w⟫
      = ⟪g yi - g yti, w⟫ + ⟪g yti - gt yti, w⟫ + c * ‖w‖ ^ 2 - c * ⟪xi - xti, w⟫ := by
    have hsplit : (g yi + c • (yi - xi)) - (gt yti + c • (yti - xti))
        = (g yi - g yti) + ((g yti - gt yti) + (c • w - c • (xi - xti))) := by
      rw [hw]; module
    rw [← inner_sub_left, hsplit]
    simp only [inner_add_left, inner_sub_left, real_inner_smul_left,
      real_inner_self_eq_norm_sq]
    ring
  have hsum : ⟪g yi - g yti, w⟫ + ⟪g yti - gt yti, w⟫ + c * ‖w‖ ^ 2
      - c * ⟪xi - xti, w⟫ ≤ 0 := by linarith [expand]
  have hb1 : -⟪g yti - gt yti, w⟫ ≤ L * d * ‖w‖ := by
    have habs := abs_le.mp (abs_real_inner_le_norm (g yti - gt yti) w)
    have hmul : ‖g yti - gt yti‖ * ‖w‖ ≤ L * d * ‖w‖ :=
      mul_le_mul_of_nonneg_right hcross (norm_nonneg w)
    linarith [habs.1]
  have hb2 : c * ⟪xi - xti, w⟫ ≤ c * (d * ‖w‖) := by
    have habs := abs_le.mp (abs_real_inner_le_norm (xi - xti) w)
    have : ⟪xi - xti, w⟫ ≤ d * ‖w‖ :=
      le_trans habs.2 (mul_le_mul_of_nonneg_right hx (norm_nonneg w))
    exact mul_le_mul_of_nonneg_left this hc
  have hLd : 0 ≤ L * d := le_trans (norm_nonneg _) hcross
  have hquad : (c + mu) * ‖w‖ ^ 2 ≤ (c + L) * d * ‖w‖ := by nlinarith
  rcases eq_or_lt_of_le (norm_nonneg w) with hzero | hpos
  · rw [← hzero]; nlinarith
  · refine le_of_mul_le_mul_right ?_ hpos
    nlinarith

/-- The proximal factor is below one exactly when own curvature beats cross-sensitivity. -/
theorem contractionFactor_lt_one_iff (hc : 0 ≤ c) (hmu : 0 < mu) :
    (c + L) / (c + mu) < 1 ↔ L < mu := by
  have hpos : 0 < c + mu := by linarith
  rw [div_lt_one hpos]
  constructor <;> intro h <;> linarith

/-- **The contraction bound is monotone in the adjustment weight.** Where the game already
contracts, raising the weight moves the factor towards one, so no weight pushes this bound
below one when `mu <= L`.

This constrains the *certificate*, not the dynamics: a game whose simultaneous best response
diverges can still be stabilised by inertia, with this row-sum-style bound failing
throughout. -/
theorem contractionFactor_le_of_le {c' : ℝ} (hmu : 0 < mu) (hL : L < mu) (hc : 0 ≤ c)
    (hcc : c ≤ c') : (c + L) / (c + mu) ≤ (c' + L) / (c' + mu) := by
  have h1 : 0 < c + mu := by linarith
  have h2 : 0 < c' + mu := by linarith
  rw [← sub_nonneg]
  have key : (c' + L) / (c' + mu) - (c + L) / (c + mu)
      = ((mu - L) * (c' - c)) / ((c' + mu) * (c + mu)) := by
    field_simp; ring
  rw [key]
  apply div_nonneg <;> nlinarith

end PricingPerspective.Operating.VariationalInequality
