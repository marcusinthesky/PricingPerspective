"""Simulation — GPU-capable JAX Monte-Carlo package.

Re-exports the backend configuration entry-point so callers can do::

    import simulation

    info = simulation.configure(precision="float64", platform="cpu")
"""

from simulation.backend import BackendInfo, configure

__version__ = "0.1.0"

__all__ = ["BackendInfo", "__version__", "configure"]
