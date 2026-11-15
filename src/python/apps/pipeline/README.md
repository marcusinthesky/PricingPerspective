# Paper reproduction pipeline

`pipeline` computes the shared geometry, statistical analyses and publication
assets for Papers 1, 3, 5 and 6. Its CLI is composed from substrate adapters and
one lane per paper. DVC owns input/output paths, parameter selection and stage
execution; Semflow tracks the reachable scientific source dependencies.

The public inputs are computed vectors, adjusted returns, classifications and
numeric controls. No news scraping or article-text processing is required.
Embedding checks enforce a text-free schema and the fixed cohort/checksum
manifest. Separate numeric TF–IDF fits preserve the lexical control estimands.

The statistical implementation is the pinned public Statadic distribution.
`hotelling` owns Paper 6's dated operating model; `dotell` owns shared runtime
telemetry and JAX cache policy. Computations explicitly own float64 initialization
and retain the original sample rules, solver tolerances and resampling schedules.

From the repository root:

```sh
just python::pipeline::test-fresh
just python::pipeline::test-all
dvc repro
just reproduction-check
```

Tests use independent references where available, including exhaustive transport
assignments, spatial-estimation oracles and analytic fixtures. Final validation
runs complete suites in separate package processes without Testmon selection.
See [REPLICATION.md](../../../../REPLICATION.md) for the fixed inputs, historical
benchmarks and published-precision acceptance check.
