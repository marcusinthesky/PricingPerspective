"""The sole disk / DuckDB / params side-effect layer.

Everything that touches the filesystem or an external format on behalf of a
stage lives here: ``params.yaml`` accessors (:mod:`pipeline.io.params`),
publication-value / ``numbers.tex`` emission (:mod:`pipeline.io.values`), and
DuckDB/parquet readers and writers. ``io`` may be imported by
``pipeline.stages``, but never imports ``pipeline.stages`` itself.
"""

from __future__ import annotations
