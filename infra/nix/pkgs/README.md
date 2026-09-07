---
description: Packages this flake vendors because nixpkgs has no usable attribute for them — one file per derivation.
---

# infra/nix/pkgs/ — vendored derivations

One file per package the dev shell cannot take straight from `pkgs`. Each file
owns its own rationale and its `BUMPING:` runbook, because that prose is read at
the moment its derivation is edited.

## What earns a file here

| Reason | Packages |
|---|---|
| Upstream is not in nixpkgs | `frontmatter-cli`, `dbt-language-server`, `merman-cli`, `semble-rs` |
| The nixpkgs version trails the workspace contract | `bun` |
| The nixpkgs attribute is the wrong shape | `chktex`, `pgf-metrics`, `latexdiff` (all need TeX Live scaffolding the standalone attribute lacks) |
| Built from a flake input, not from `pkgs` | `duckdb-cli`, `dbt-with-duckdb`, `lean-toolchain` |

A package that is simply `pkgs.<name>` does **not** belong here — declare it in
the dev shell instead.

## Call convention

`default.nix` uses `callPackage` when a file **constructs** a derivation: it
fills the nixpkgs arguments by name and returns a `.override`, so a pin can be
retargeted without editing the file. It uses plain `import` when a file
**selects** an existing package or returns a set instead — `callPackage`'s
`makeOverridable` wrapper would otherwise bolt an `override` attribute onto a
package that already has its own (`duckdb-cli`) or onto an attrset with nothing
to override (`lean-toolchain`).

## Building one

Every package here is a flake output, so a bad bump fails on the package that
broke rather than halfway through realizing the dev shell:

```bash
nix build ./infra/nix#semble-rs
nix flake show ./infra/nix          # the full list
```

`lean-toolchain` is an attrset rather than a derivation; the flake output is its
`.lean-all` member, which is what the shells put on PATH.

## Adding one

New `.nix` files must be `git add`-ed before Nix can see them — this flake
resolves as `git+file://…?dir=infra/nix`, so untracked files are not copied into
the store and evaluation fails on a file that plainly exists on disk.
