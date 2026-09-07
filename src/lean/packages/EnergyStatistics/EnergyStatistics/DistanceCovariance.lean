import EnergyStatistics.DistanceCovariance.Defs
import EnergyStatistics.DistanceCovariance.Basic
import EnergyStatistics.DistanceCovariance.Sample

/-!
# Distance Covariance

This module formalizes the population distance covariance (dCov²),
following Székely & Rizzo (2023), "The Energy of Data and Distance Correlation".

## Main definitions

* `distanceCovarianceSq`: Squared distance covariance dCov²(X, Y)
* `distanceVarianceSq`: Squared distance variance dVar²(X)
* `distanceVarianceSqSnd`: Squared distance variance dVar²(Y)

## Main theorems

* `dcov_nonneg`: dCov²(X, Y) ≥ 0

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapters 12, 15.
-/
