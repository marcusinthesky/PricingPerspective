import EnergyStatistics.Defs
import EnergyStatistics.DistanceCorrelation

set_option linter.style.longLine false

/-!
# Permutation Tests and Asymptotic Theory

This module contains placeholder scaffolds for the permutation test
framework and asymptotic distribution theory for distance correlation,
following Székely & Rizzo (2023), Chapter 13.

## Main theorems

* `permutation_test_distribution`: Under H₀, permutation distribution is distribution-free
* `dcor_asymptotic_null`: Asymptotic null distribution of n·dCor²_n

## References

* Székely, G. J., & Rizzo, M. L. (2023). The Energy of Data and Distance Correlation.
  CRC Press. Chapter 13.
-/

namespace EnergyStatistics

/-- The permutation test for independence based on distance correlation.

    Under H₀: X ⊥ Y, the permutation distribution of dCor is the same
    as the distribution of dCor computed on permuted pairs. This is a
    consistent test against all alternatives.

    Reference: Székely & Rizzo (2023), Chapter 13, Section 13.2.1. -/
theorem permutation_test_distribution
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β] :
    True := by
  sorry

/-- Asymptotic distribution of √n · dCor under the null.

    Under H₀: X ⊥ Y with finite moments:

    n · dCor²_n →d Q

    where Q is a weighted sum of independent χ²(1) random variables.
    The weights depend on the underlying distributions through their
    characteristic functions.

    Reference: Székely & Rizzo (2023), Chapter 13. -/
theorem dcor_asymptotic_null
    {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [PseudoMetricSpace α] [PseudoMetricSpace β] :
    True := by
  sorry

end EnergyStatistics
