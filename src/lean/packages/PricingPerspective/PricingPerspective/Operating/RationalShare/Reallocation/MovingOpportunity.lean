import PricingPerspective.Operating.RationalShare.Reallocation.RevealedPreference

/-!
# Revealed preference with moving opportunities and rival fields

This module specializes the generic two-date revealed-preference restriction to an environment
with two components: common operating opportunities and an outside-plus-rival field.  It also
records both exact attribution orders.  The full gain is invariant, while the quantities called
"crowding" and "opportunity" depend on which component is changed first.
-/

namespace PricingPerspective.Operating.RationalShare.Reallocation.MovingOpportunity

open PricingPerspective.Operating.VariationalInequality

variable {Position Opportunity Rival : Type*}

/-- Curry operating revenue over an opportunity state and a rival field into the product
environment required by the generic revealed-preference theorem. -/
def jointEnvironmentRevenue (revenue : Position → Opportunity → Rival → ℝ) :
    Position → Opportunity × Rival → ℝ :=
  fun position environment ↦ revenue position environment.1 environment.2

/-- The new position's operating-revenue advantage over the old position in one joint
environment. -/
def relativeAdvantage (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (opportunity : Opportunity) (rival : Rival) : ℝ :=
  revenue newPosition opportunity rival - revenue oldPosition opportunity rival

/-- The full two-date revealed-reallocation gain when both opportunities and rivals may move. -/
def fullGain (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) : ℝ :=
  relativeAdvantage revenue oldPosition newPosition newOpportunity newRival -
    relativeAdvantage revenue oldPosition newPosition oldOpportunity oldRival

/-- Rival-first crowding attribution: change the rival field while holding old opportunities. -/
def crowdingFirst (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity : Opportunity)
    (oldRival newRival : Rival) : ℝ :=
  relativeAdvantage revenue oldPosition newPosition oldOpportunity newRival -
    relativeAdvantage revenue oldPosition newPosition oldOpportunity oldRival

/-- Rival-first opportunity attribution: change opportunities after moving to the new rival
field. -/
def opportunitySecond (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (newRival : Rival) : ℝ :=
  relativeAdvantage revenue oldPosition newPosition newOpportunity newRival -
    relativeAdvantage revenue oldPosition newPosition oldOpportunity newRival

/-- Opportunity-first attribution: change opportunities while holding the old rival field. -/
def opportunityFirst (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival : Rival) : ℝ :=
  relativeAdvantage revenue oldPosition newPosition newOpportunity oldRival -
    relativeAdvantage revenue oldPosition newPosition oldOpportunity oldRival

/-- Opportunity-first crowding attribution: change the rival field after moving to new
opportunities. -/
def crowdingSecond (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (newOpportunity : Opportunity)
    (oldRival newRival : Rival) : ℝ :=
  relativeAdvantage revenue oldPosition newPosition newOpportunity newRival -
    relativeAdvantage revenue oldPosition newPosition newOpportunity oldRival

/-- The moving-opportunity full gain is the generic reallocation gain on the product
environment `(opportunity, rival)`. -/
theorem fullGain_eq_reallocationGain (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) :
    fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival =
      RevealedPreference.reallocationGain (jointEnvironmentRevenue revenue)
        oldPosition newPosition (oldOpportunity, oldRival) (newOpportunity, newRival) := by
  rfl

/-- Stable-cost pairwise optimality in the two joint environments implies a nonnegative full
gain. -/
theorem fullGain_nonneg_of_pairwise_choice
    (revenue : Position → Opportunity → Rival → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival)
    (oldChoice :
      RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost newPosition
          (oldOpportunity, oldRival) ≤
        RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost oldPosition
          (oldOpportunity, oldRival))
    (newChoice :
      RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost oldPosition
          (newOpportunity, newRival) ≤
        RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost newPosition
          (newOpportunity, newRival)) :
    0 ≤ fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival :=
  fullGain_eq_reallocationGain revenue oldPosition newPosition oldOpportunity newOpportunity
      oldRival newRival ▸
    RevealedPreference.reallocationGain_nonneg_of_pairwise_choice
      (jointEnvironmentRevenue revenue) cost oldPosition newPosition
      (oldOpportunity, oldRival) (newOpportunity, newRival) oldChoice newChoice

/-- Approximate stable-cost choices imply the same payoff-regret allowance for the full
moving-opportunity gain. -/
theorem fullGain_ge_neg_allowances
    (revenue : Position → Opportunity → Rival → ℝ) (cost : Position → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) (oldAllowance newAllowance : NNReal)
    (oldChoice :
      RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost newPosition
          (oldOpportunity, oldRival) ≤
        RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost oldPosition
          (oldOpportunity, oldRival) + (oldAllowance : ℝ))
    (newChoice :
      RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost oldPosition
          (newOpportunity, newRival) ≤
        RevealedPreference.netPayoff (jointEnvironmentRevenue revenue) cost newPosition
          (newOpportunity, newRival) + (newAllowance : ℝ)) :
    -((oldAllowance : ℝ) + (newAllowance : ℝ)) ≤
      fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival :=
  fullGain_eq_reallocationGain revenue oldPosition newPosition oldOpportunity newOpportunity
      oldRival newRival ▸
    RevealedPreference.reallocationGain_ge_neg_allowances
      (jointEnvironmentRevenue revenue) cost oldPosition newPosition
      (oldOpportunity, oldRival) (newOpportunity, newRival) oldAllowance newAllowance
      oldChoice newChoice

/-- In the rival-first attribution order, crowding followed by opportunity movement sums to the
full gain. -/
theorem crowdingFirst_add_opportunitySecond
    (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) :
    crowdingFirst revenue oldPosition newPosition oldOpportunity oldRival newRival +
        opportunitySecond revenue oldPosition newPosition oldOpportunity newOpportunity
          newRival =
      fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival := by
  simp only [fullGain, crowdingFirst, opportunitySecond, relativeAdvantage]
  ring

/-- In the opportunity-first attribution order, opportunity movement followed by crowding sums
to the same full gain. -/
theorem opportunityFirst_add_crowdingSecond
    (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) :
    opportunityFirst revenue oldPosition newPosition oldOpportunity newOpportunity oldRival +
        crowdingSecond revenue oldPosition newPosition newOpportunity oldRival newRival =
      fullGain revenue oldPosition newPosition oldOpportunity newOpportunity oldRival newRival := by
  simp only [fullGain, opportunityFirst, crowdingSecond, relativeAdvantage]
  ring

/-- The change in the crowding attribution across orders equals the corresponding change in the
opportunity attribution.  Hence neither component is uniquely identified by the invariant full
gain. -/
theorem crowdingSecond_sub_crowdingFirst_eq_opportunitySecond_sub_opportunityFirst
    (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (oldRival newRival : Rival) :
    crowdingSecond revenue oldPosition newPosition newOpportunity oldRival newRival -
        crowdingFirst revenue oldPosition newPosition oldOpportunity oldRival newRival =
      opportunitySecond revenue oldPosition newPosition oldOpportunity newOpportunity newRival -
        opportunityFirst revenue oldPosition newPosition oldOpportunity newOpportunity
          oldRival := by
  simp only [crowdingFirst, opportunitySecond, opportunityFirst, crowdingSecond,
    relativeAdvantage]
  ring

/-- With an unchanged rival field, either crowding attribution is exactly zero. -/
@[simp] theorem crowding_attributions_same_rival
    (revenue : Position → Opportunity → Rival → ℝ)
    (oldPosition newPosition : Position) (oldOpportunity newOpportunity : Opportunity)
    (rival : Rival) :
    crowdingFirst revenue oldPosition newPosition oldOpportunity rival rival = 0 ∧
      crowdingSecond revenue oldPosition newPosition newOpportunity rival rival = 0 := by
  simp [crowdingFirst, crowdingSecond]

end PricingPerspective.Operating.RationalShare.Reallocation.MovingOpportunity
