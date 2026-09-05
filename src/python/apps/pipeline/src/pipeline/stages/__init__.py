"""Thin orchestrators, one per DVC stage-cluster.

Each module here wires ``pipeline.io`` and ``pipeline._kernels`` calls together
to satisfy one or more ``dvc.yaml`` stages; it holds no numerical logic of its
own and no side-effect primitives beyond what ``pipeline.io`` exposes.
``stages`` may import ``pipeline.io`` and ``pipeline._kernels``, never the
reverse.

Tiered into shared ``substrate/`` (corpus/embeddings/market/energy/ablation/
mantel/dimensionality — reused across papers) and per-paper ``papers/paperN/``
subpackages (paper1/dyadic_confound, paper3/empirical,
paper5/).
"""

from __future__ import annotations
