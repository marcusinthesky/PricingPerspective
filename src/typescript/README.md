---
description: "Bun-managed TypeScript workspace for deployable applications and shared packages."
---

# src/typescript/

The JavaScript/TypeScript workspace for the polyglot repository. Bun owns
dependency installation and script execution; Turbo owns the package task graph.

## Layout

```text
.
├── apps/
│   └── portfolio/       # Static Next.js research website
├── packages/            # Shared workspace packages (reserved for additions)
├── bun.lock             # Single workspace lockfile
├── package.json         # Workspace manifest and public commands
├── turbo.json           # Task graph and cache boundaries
└── tsconfig.base.json   # Shared TypeScript compiler defaults
```

Every application or package under `apps/*` or `packages/*` has its own
`package.json`. Dependencies stay with the package that imports them; the root
manifest contains only workspace metadata and Turbo.

## Commands

Run these from `src/typescript/`:

```bash
bun ci
bun run dev
bun run typecheck
bun run lint
bun run format
bun run format:check
bun run build
bun run ci
```

The equivalent repository-level recipes are `just typescript::setup`,
`just typescript::lint`, `just typescript::typecheck`, and
`just typescript::verify`. CI runs `bun run ci` from this workspace root.

## Boundaries

Turbo tasks are declared once in `turbo.json`. Build-like tasks declare their
artifacts, development tasks are persistent and uncached, and the public-site
URL is part of the build/verification environment hash. Add a shared library
under `packages/` only when more than one application has a real dependency on
it; do not create a shared package as a dumping ground.
