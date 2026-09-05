---
description: "Provenance gate asserting every CLI tool the repository declares is provisioned by the Nix dev shell rather than a machine-local profile."
---

# Toolchain provenance

The repository *declares* the CLI tools it expects on `PATH` in two
machine-readable places, and *provisions* them in a third:

| Role | Where |
|---|---|
| declares | `.agents/claude/settings.json` — `Bash(<tool> …)` permission entries |
| declares | every `.pre-commit-config.yaml` — `entry:` of each `language: system` hook |
| provisions | `infra/nix/flake.nix` — the devShell `packages` list |

`just nix::toolchain-provenance` connects the three.

## Why the obvious check is the wrong one

The intuitive gate — run the repository's documented commands inside a
`PATH`-scrubbed `nix develop` and see what breaks — is slow, noisy, and
non-deterministic: every recipe shells out to a dozen binaries, and a failure
could mean a missing tool or an unrelated bug. This contract asks the narrower
question that actually discriminates, without executing anything:

```text
store     resolved under /nix/store   -- provisioned by the flake, portable
ambient   resolved elsewhere          -- a machine-local profile answered
missing   did not resolve at all      -- stale declaration, or absent package
```

`ambient` is the interesting class. An interactive Nix shell inherits the
ambient `PATH`, so a home-manager profile silently satisfies a lookup the flake
never provisioned. The gap is invisible on the maintainer's machine and fatal in
CI and fresh worktrees.

## The two failures it was built from

Both were found on 2026-08-01, both had been latent for a long time, and they
are fixed in **opposite directions** — which is why the gate reports provenance
rather than mere presence:

- `rumdl` and `gh` resolved only from `/etc/profiles/per-user/…`. Both were
  genuinely in use — three prek configs pin `rumdl`, the `plan-node-audit` skill
  invokes it directly, and `settings.json` allowlists `gh pr create`. **Fix: add
  to the flake.**
- `typst` and `tinymist` were allowlisted in `settings.json` and advertised by
  `infra/nix/README.md`, but Typst had been retired as a renderer and the claim
  manifest that justified keeping the tooling now renders LaTeX. **Fix: delete
  the dead declaration.**

## Scope and known limits

The scanner reads `entry:` values, so it sees the *leading* binary only. A hook
written `entry: bash -c '… rumdl …'` contributes `bash` (exempt) and hides
`rumdl`. This is deliberate — resolving binaries inside arbitrary shell strings
needs a shell parser and yields false positives — but it means the hook `entry:`
source is a weaker signal than the permission allowlist. Prefer declaring a tool
in `settings.json`, or invoke it as the entry directly.

Exemptions live in `_EXEMPT` in `toolchain_provenance.py` and each requires a stated reason:
an unexplained exemption is indistinguishable from an unnoticed gap. `nix`
itself is exempt because it bootstraps the shell it would otherwise be checked
against.

The companion check is `just nix::rumdl-lockstep`, which asserts the flake's
`rumdl` version equals the version pinned by every prek `rev:`. Provenance and
version are separate failures: a store-provided binary at the wrong version
still makes ad-hoc linting disagree with the gate.

## Running

Must run **inside** the dev shell — it inspects the `PATH` the shell builds.

```bash
just nix::toolchain-provenance     # or: prek run toolchain-provenance --all-files
```

The `toolchain-provenance` prek hook triggers on any edit to a declaration site
(`**/.pre-commit-config.yaml`, `.agents/claude/settings.json`) or to
`infra/nix/**/*.nix` / `infra/nix/flake.lock`, with `pass_filenames: false` — the contract is over
the declaration set as a whole, not over the touched file.
