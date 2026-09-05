---
description: Reactive, typed materializations for Markdown and YAML front matter.
---

# Insitu

Insitu keeps generated Markdown and YAML front matter synchronized with typed,
location-aware resources. Its reusable core provides a SQLite repository index,
Jinja and SQL transforms, native file watching, dependency-cycle checks, atomic
writes, and language-neutral JSON-RPC extensions.

## Managed regions

```markdown
<!-- insitu:begin tree
id = "layout"
depth = 2
-->
generated content
<!-- insitu:end -->
```

The opening comment contains a TOML header. `id` identifies the owned target;
the comment's optional kind selects a transform. Region bodies are output and
are excluded from the source projection, preventing generated content from
becoming its own input.

## Commands

```bash
uv run --project src/python --package insitu insitu check --root .
uv run --project src/python --package insitu insitu sync --root .
uv run --project src/python --package insitu insitu watch --root .
```

This repository normally uses `just insitu::check`, `just insitu::sync`, and
`just insitu::watch`; those commands add the repository-specific structural
pre-pass and projectors from `src/python/apps/insitu-repository`.

## Extension trust

Extensions are supervised processes with resource permissions enforced for
host-mediated reads. They are trusted code, not operating-system sandboxes:
their processes inherit the invoking user's environment and filesystem access.
