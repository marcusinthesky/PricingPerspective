"""Pure, I/O-free numerical kernels.

This layer holds unit-testable numerical functions only: no disk reads/writes,
no DuckDB, no ``params.yaml`` parsing, no LaTeX emission. It imports neither
``pipeline.io`` nor ``pipeline.stages`` — dependencies in this package point
strictly downward (``stages`` -> ``io``/``_kernels``; ``io`` -> nothing in this
package; ``_kernels`` -> nothing in this package). This keeps a shared-kernel
edit from invalidating DVC stages that do not actually use it.

Empty for now; t04.3 migrates finely-split pure-compute helpers in here out of
the existing flat modules.
"""

from __future__ import annotations
