import PricingPerspective.Transmission.Defs
import PricingPerspective.Transmission.RiskCoordinates
import PricingPerspective.Transmission.Kernel
import PricingPerspective.Transmission.Frechet
import PricingPerspective.Transmission.Slack
import PricingPerspective.Transmission.Empirical
import PricingPerspective.Transmission.EmpiricalExact
import PricingPerspective.Transmission.FactorReturn
import PricingPerspective.Transmission.PortfolioLaw
import PricingPerspective.Transmission.Envelope
import PricingPerspective.Transmission.Transfer
import PricingPerspective.Transmission.Factor
import PricingPerspective.Transmission.Portfolio
import PricingPerspective.Transmission.Examples
import PricingPerspective.Transmission.Dispersion

/-!
# Random functional factor exposures

Stable import surface for the finite random-exposure theory spike. The definitions,
covariance envelope, and characteristic-to-exposure transfer are split by concept so
individual proofs remain independently auditable.
-/
