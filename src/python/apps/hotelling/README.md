# Dated operating adjustment

`hotelling` provides the dated capacity game and measurement kernels used by
Paper 6. The operating model separates its finite-history estimand from
conditional article sensitivity and population sampling claims.

The pipeline composes these kernels with frozen numeric designs, article-support
identifiers/dates and adjusted-return panels. Historical recovery checkpoints are
immutable numeric benchmarks; current analyses keep independent stage identities
and resumption checks. Source, dependency and Lean proof identities are recorded
separately from numerical equality.

Statadic supplies the public numerical primitives. The Lean publication checks
certify the retained formal declarations separately from numerical execution.
Package tests check the operating model, reporting geometry and numerical
assumptions.

```sh
just python::hotelling::test-fresh
just python::hotelling::test-all
```

Use the root [README](../../../../README.md) for complete input
and execution instructions.
