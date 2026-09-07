"""Public distance-metric API for simulation sample clouds."""

from jcor.discrepancy.transport import sliced_wasserstein as sliced_wasserstein
from jcor.discrepancy.transport import w1_1d as w1_1d
from jcor.discrepancy.transport import w2_1d as w2_1d

from .dispatch import pairwise_distance_matrix as pairwise_distance_matrix
from .energy import pairwise_energy_distance as pairwise_energy_distance

__all__ = [
    "pairwise_distance_matrix",
    "pairwise_energy_distance",
    "sliced_wasserstein",
    "w1_1d",
    "w2_1d",
]
