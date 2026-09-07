"""Shared data-generating processes for simulation Monte-Carlo validators."""

from simulation.dgp.spatial import SarDGP, make_synthetic_w, simulate_sar
from simulation.generators.assets import ClusterSample, cluster_embeddings

__all__ = [
    "ClusterSample",
    "SarDGP",
    "cluster_embeddings",
    "make_synthetic_w",
    "simulate_sar",
]
