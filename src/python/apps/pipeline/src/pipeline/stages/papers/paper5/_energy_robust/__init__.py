"""Private step modules for Paper 5's energy-robustness prerequisite.

``pipeline.stages.papers.paper5.energy_robust`` is the only supported import path;
it owns the ``run_paper5_energy_robustness`` entry point and the addressed
contract and optimizer surface. The modules here are an internal organisation
detail of that stage; do not re-export them here or import them outside
``energy_robust.py``.
"""
