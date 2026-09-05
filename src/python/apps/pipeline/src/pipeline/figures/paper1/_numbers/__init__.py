"""Private record-builder modules for the Paper 1 numbers.tex record set.

``pipeline.figures.paper1.numbers`` is the only supported import path -- it owns
the contracts (``Paper1NumberPaths``, ``Paper1NumberInputs``), the loader, and
the record order. The modules here are an internal organisation detail of that
role and carry no stability guarantee; do NOT re-export them here, and do not
import them from outside ``numbers.py``.
"""
