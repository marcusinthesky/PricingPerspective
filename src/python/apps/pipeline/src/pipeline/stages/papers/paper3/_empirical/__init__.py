"""Private step modules for the Paper 3 empirical stage.

``pipeline.stages.papers.paper3.empirical`` is the only supported import path --
it owns the ``run_paper3_empirical`` entry point and the addressed contract /
helper surface. The modules here are an internal organisation detail of that
stage; do NOT re-export them here, and do not import them from outside
``empirical.py``.
"""
