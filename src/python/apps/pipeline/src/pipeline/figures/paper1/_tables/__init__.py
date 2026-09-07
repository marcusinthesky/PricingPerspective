"""Private table-renderer modules for the Paper 1 empirical exhibits.

``pipeline.figures.paper1.tables`` is the only supported import path -- it owns
the ``Paper1EmpiricalTablePaths`` contract, the render sequence, and the
addressed renderer surface. The modules here are an internal organisation
detail of that role; do NOT re-export them here, and do not import them from
outside ``tables.py``.
"""
