# Analysis architecture

`pipeline` owns experiment specifications, sample alignment, DVC artifacts and
publication bindings. It consumes the local `hotelling` and `dotell` packages.
Neither dependency imports `pipeline`. Public Statadic owns reusable statistical
methods and their numerical implementations.

Within Pipeline, command adapters call stage orchestrators and figure renderers.
Shared I/O validates data identities and artifact contracts; `_kernels` contains
pure numerical routines. DVC declares the retained dataflow and Semflow records
the reachable repository code for each command.

Public reproduction begins at sanitized embeddings and permitted numeric or
cohort metadata. Acquiring articles and invoking embedding providers are outside
that boundary. Article IDs, dates, sample membership and published scientific
settings remain fixed across the handoff.

Precision belongs to each numerical stage. Commands and spawned workers that
require float64 select it before constructing JAX arrays. Imports and the shared
compilation-cache configuration do not change global precision. Tests run in
separate package processes so fixture and precision policies cannot leak.

The workspace lock preserves the versions used by the reported analyses.
Dependency changes require explicit replay and comparison with the published
values. A documentation or provenance change must not silently change a sample,
estimand, solver, resampling design or inference method.
