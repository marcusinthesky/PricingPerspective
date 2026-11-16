"""Dependency rules exercise real ownership boundaries, not directory inventories."""

from __future__ import annotations

import unittest

from check import violations


class BoundaryTests(unittest.TestCase):
    def test_foundations_cannot_depend_on_applications(self) -> None:
        self.assertTrue(
            violations([("EnergyStatistics.Defs", "PricingPerspective.Exposure.Defs")])
        )

    def test_cores_cannot_depend_on_compositions(self) -> None:
        self.assertTrue(
            violations(
                [
                    (
                        "PricingPerspective.Spatial.Model",
                        "PricingPerspective.Connections.MathEcon",
                    )
                ]
            )
        )
        self.assertTrue(
            violations(
                [
                    (
                        "PricingPerspective.Operating.RationalShare.Defs",
                        "PricingPerspective.Spatial.Model",
                    )
                ]
            )
        )

    def test_leaf_dependencies_in_both_directions_remain_legal(self) -> None:
        self.assertEqual(
            violations(
                [
                    (
                        "PricingPerspective.Exposure.Energy.Core",
                        "PricingPerspective.Transmission.Floor",
                    ),
                    (
                        "PricingPerspective.Transmission.Dispersion.PortfolioVariance",
                        "PricingPerspective.Exposure.Energy.Portfolio",
                    ),
                    (
                        "PricingPerspective.Connections.OperatingFinance.Exposure",
                        "PricingPerspective.Operating.Valuation.StateExposure",
                    ),
                    ("EnergyStatistics.Defs", "StatadicFormal.Geometry.NegativeType.Defs"),
                ]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
