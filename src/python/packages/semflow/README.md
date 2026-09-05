---
description: "Semantic Python fingerprints and review-first DVC dependency patches that suppress non-executable pipeline churn without taking ownership of the stage graph."
---

# semflow

`semflow` resolves a DVC stage's installed Typer command, computes the conservative
first-party symbol closure reachable from its callback, and replaces the stage's raw
Python `deps:` entries with one deterministic entrypoint JSON fingerprint. DVC
continues to own commands, data dependencies, parameters, outputs, matrices,
`foreach`, caching, and frozen stages.

Version 0.1 deliberately has three verbs:

```bash
semflow check          # read-only; exit 1 when YAML or hashes are stale
semflow patch          # print the exact unified diff; write nothing
semflow patch --apply  # materialize the reviewed diff; atomic per file, YAML last
semflow refresh --apply # materialize JSON changes only; refuse stale DVC YAML
```

The same commands are available from the pipeline application as
`pipeline semflow …`.

`refresh` is the pure control-plane operation used by the optional DVC
`semflow_refresh` stage. It recomputes the fingerprints but never adds, removes,
or rewrites a dependency alias in `dvc.yaml`. Run `patch --apply` first when
instrumentation itself is stale.

## Configuration

`semflow.toml` is discovered from the current directory upward. A
`[tool.semflow]` table in `pyproject.toml` is also supported.

```toml
project-root = "."
pipeline-file = "dvc.yaml"
hash-dir = ".semflow/hashes"
source-roots = [
  "src/python/apps/pipeline/src",
  "src/python/apps/simulation/src",
  "src/python/packages/jcor/src",
]

# Optional when a console script is not installed in the active environment.
[applications]
example = "example.cli:app"

[hashing]
ignore-docstrings = true
ignore-annotations = true
ignore-type-checking-blocks = true

[tracking]
auto-typer = true
control-stages = ["semflow_refresh"]
```

With `auto-typer = true`, installed Typer callbacks referenced by DVC commands are
tracked automatically. The stage's Python dependencies collapse to a path such as
`.semflow/hashes/entrypoints/simulation/audit-kappa-identity.json`. Its `members`
array records only the reachable functions, classes, globals, and import bindings,
plus an explained whole-module fallback if a first-party reference cannot be
resolved safely.

Manual tracking uses an identity-preserving decorator, so Typer receives the same
function object and signature:

```python
from semflow import track


@app.command("audit")
@track(include=("example.audit:EXTRA_POLICY",))
def audit() -> None: ...
```

Set `auto-typer = false` to instrument only callbacks carrying `@track`. Explicit
`include` entries use `module:symbol` syntax and seed the same transitive static
closure. Paths outside `source-roots` remain ordinary DVC dependencies. Stages whose
commands are outside the discovered applications retain file/directory fingerprint
behavior for their declared Python deps.

## Safety contract

- `patch` is preview-only unless `--apply` is explicit. Apply aborts if any planned
  file changed after the plan was built; fingerprint files are replaced atomically
  before the authored YAML is replaced last.
- YAML editing is source-dependency-local: command folding, matrices, stage order,
  and unrelated comments remain byte-for-byte unchanged. A comment directly attached
  to a removed Python dependency is removed with that dependency rather than left
  misleadingly in place.
- Generated JSON records the entrypoint, callback, normalization policy, aggregate
  digest, reachable member digests, and any conservative fallbacks; it contains no
  timestamps or host-specific values.
- `check` must gate commits once the DVC patch is activated. Without that gate, a
  semantic source edit could leave a stale JSON proxy and suppress invalidation.
- Configured `control-stages` are excluded from instrumentation discovery so a
  DVC refresh stage cannot instrument itself.
- Annotation removal is opt-in and limited to undecorated function signatures and
  function-local annotations. Typer callbacks and decorated functions always retain
  annotations; class/module annotations remain semantic because dataclasses, Pydantic,
  and `NamedTuple` can consume them as runtime schema.

The closure is static and first-party: direct names, imports, attributes, globals,
decorators, defaults, and transitively reached symbols are followed. Runtime tracing
is deliberately not authoritative, and string-driven dynamic imports are outside the
initial contract. An unresolved static first-party edge hashes the owning module as a
conservative fallback and records the reason in JSON.

## Development

```bash
just python::semflow::check
just python::semflow::test
just python::semflow::lint
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the boundary decisions.
