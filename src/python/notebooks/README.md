---
description: Standalone PEP 723 marimo notebooks, one directory each.
---

# notebooks

Each notebook is a self-contained PEP 723 unit — `notebook.py` plus the
`notebook.py.lock` that pins it — not a uv workspace member. Nothing here is
imported by `apps/` or `packages/`; the dependency arrow only ever points inward.

```text
notebooks/<slug>/
├── notebook.py        # marimo notebook; `# /// script` declares its dependencies
├── notebook.py.lock   # `just python::notebooks::lock <slug>`
├── README.md          # what the notebook is for
├── __marimo__/        # marimo's own outputs, including the .ipynb rendering
└── figures/           # generated plots, where a notebook produces them
```

`playground/` additionally carries what its Hugging Face Space needs: `custom.css`
and `head.html` injected into the export, a `public/` folder copied verbatim into
it, and an untracked `dist/` holding the built bundle.

Because these are not workspace members, marimo runs from its own pinned,
ephemeral environment (`uvx`) and every recipe takes the slug as an argument:

```bash
just python::notebooks::edit <slug>    # live editor, --sandbox
just python::notebooks::wasm <slug>    # self-contained WebAssembly bundle
just python::notebooks::html <slug>    # static HTML snapshot
just python::notebooks::lock <slug>    # refresh notebook.py.lock
just python::notebooks::check <slug>   # format-check, lint, typecheck
```

Ruff still applies its full rule set to notebook source; only the handful of rules
that describe marimo's *generated* file format are waived, in the workspace
`pyproject.toml`. Type checking runs per file against each notebook's own inline
environment, via `just python::typecheck-scripts`.
