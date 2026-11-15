# Python analysis workspace

The uv workspace contains three local packages:

| Member | Responsibility |
|---|---|
| `apps/pipeline` | Four-paper DVC stages and publication rendering. |
| `apps/hotelling` | Operating games, recovery simulations and solver contracts. |
| `packages/dotell` | Logging, telemetry and persistent JAX compilation caching. |

Reusable statistical methods come from the public `statadic==0.5.0a2` wheel.
The native `insitu-cli==0.2.0rc2` wheel materializes the documentation trees.
`apps/insitu-repository/lifecycle.justfile` is its thin command wrapper.
`just nix::inventory` exposes the shared Nix tool inventory. Semflow is pinned to
an immutable Git revision, and `uv.lock` fixes the complete environment.

From the repository root:

```bash
just python::setup
just python::lint
just python::typecheck
just python::test-fresh
```

`test-fresh` runs each package in its own pytest process without Testmon selection.
Fast suites exclude tests marked `slow` or `gpu`; `just python::test-all` includes
all tiers. Paper-stage execution and data retrieval are described in the root
README and `REPLICATION.md`.

After reproduction, `just reproduction-check` compares the actual manuscript
number formats, exact counts and cohorts, and the scientific geometry or pixels
of every retained figure. Its fixed references are retrieved with the other DVC
inputs. Add `--raw-diagnostic` when invoking
`scripts/replication_release/script.py` directly to inspect changes in unprinted
digits; they do not fail the published-precision comparison.

<!-- insitu:begin tree
id = "tree"
depth = 1
-->

```text
python/
├── apps/
├── packages/
├── scripts/
├── ARCHITECTURE.md
├── justfile
├── pyproject.toml
├── README.md
├── tach.toml
└── uv.lock
```

<!-- insitu:end -->
