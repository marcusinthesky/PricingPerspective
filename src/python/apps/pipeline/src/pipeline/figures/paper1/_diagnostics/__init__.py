"""Private figure modules for the Paper 1 referee-response diagnostics.

``pipeline.figures.paper1.diagnostics`` is the only supported import path -- it
owns the ``Paper1DiagnosticPaths`` contract, the ``render_paper1_diagnostics``
entry point, and the addressed helper surface. The modules here are an internal
organisation detail of that role; do NOT re-export them here, and do not import
them from outside ``diagnostics.py``.
"""
