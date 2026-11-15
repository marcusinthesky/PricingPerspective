import PricingPerspective.Operating.RationalShare.Equilibrium

set_option linter.style.longLine false

/-!
# The rational-share game, and why its equilibrium is the abstract one

`Operating/RationalShare/Equilibrium.lean` proves existence and uniqueness for the simulated
capacity game. Appeal depends on capacity and passes through a share denominator, so its
own-action gradient is a ratio rather than an affine map.

This module treats the ratio directly. The payoff below is the one the simulation runs -- revenue
is a share `r * a / (outside + total appeal)`, each firm controls a whole block of activities, and
appeal is a nonnegative linear image of that block -- and the equivalence between Nash equilibrium
and the variational inequality is proved for *it*, with no approximation and no derivative.

**The method, stated once.** An exact unilateral payoff difference consists of a linear term
minus a nonnegative remainder. For a single node, with the rival appeal `B` held fixed and
`D` the current denominator, the exact unilateral difference is

    r * (a + s) / (D + s) - r * a / D  =  r * B * s / D ^ 2  -  r * B * s ^ 2 / (D ^ 2 * (D + s))

with no remainder term hidden. The first summand is linear in the deviation and supplies the
pseudo-gradient; the second is nonnegative and of quadratic order, and plays exactly the role the
curvature term plays in the quadratic game. The denominators stay positive because a feasible
deviation keeps every appeal nonnegative and the outside option is strictly positive, so nothing
here needs a limit, a derivative, or a bound on second derivatives.

**Monotonicity, which is not a restriction here.** An earlier draft of this module said strong
monotonicity of the pseudo-gradient was a genuine restriction on the demand primitives and left it
unproved. That was wrong, and the reason is worth stating: `Appeal.nodeBlock` is not a fixed-appeal
object at all. It is the *unscaled symmetric numerator* of this ratio game's revenue Jacobian in
appeal coordinates -- the symmetric part of that Jacobian is `(revenue l / denom l ^ 3)` times it --
so `node_block_nonneg`, already proved in `Operating/VariationalInequality/Monotone.lean`, is exactly the inequality this
payoff needs. `stronglyMonotoneOn_pseudoGradient` below connects them, and the demand side turns
out to be monotone on the whole nonnegative orthant with no hypothesis beyond nonnegative appeal.

What the modulus cannot come from is that same demand side. The bound's slack is the outside option
times the squared *appeal* direction, and the activity-to-appeal kernel need not be injective, so a
nonzero capacity direction can have zero appeal image. The strictly positive modulus therefore comes
from own-cost curvature, which the registered economy supplies as a strictly positive
`own_curvature`. A merely positive-semidefinite `H`, which this definition also admits, gives
`mu = 0` and no uniqueness: that generality is worth keeping distinct from the registered instance.

**Two dynamics, not one.** As in `Game.lean`, `Existence.lean` produces its solution by Banach
iteration on a projected residual step, which is a proof device and not the simultaneous proximal
map the simulation runs. Nothing here is a convergence theorem for those dynamics.
-/
