---
description: Conventional Lean quality checks and the multi-package path adapter for Mathlib text style.
---

# Lean quality

`just lean::lint` runs the complete quality gate from the repository's devenv shell.

1. Mathlib's `mk_all --check` verifies the three complete package import indexes.
2. Lake builds `Main` and `Publication` before linting compiled declarations.
   The compiled import-boundary check runs immediately after that build, so it sees
   current sources. The standalone `just lean::boundaries` command also builds first.
3. The pinned Batteries `runLinter` checks all three package roots
   and `Publication`, including slow environment linters. Each root runs in its own
   process so imported environments are released between packages. The local import-only
   `Main` is built and style-checked, but excluded from dynamic import-by-name linting:
   an upstream dependency also ships a `Main` module and occurs first on the search path.
4. Mathlib's text linters check first-party package sources, root publication files,
   Lake configurations and Lean tools. `just lean::fmt` applies available fixes and
   fails if any findings remain. This is not a general Lean formatter.

`just lean::quality-test` exercises both entry points with fake tool processes: an
audit requires a fresh-build marker, a failed build prevents auditing, and a failed
boundary check prevents later linting. It also checks complete root coverage and
propagation of declaration-lint failures. Fetch the mathlib cache before the first build.

Run `just lean::indexes` after adding or deleting a package module. Complete indexes
ensure package-root linting includes every source module, including modules absent
from the old headline-only roll-ups. These indexes intentionally export all package
modules; they should not be minimized as if they were ordinary implementation imports.

## Existing debt

[`scripts/nolints.json`](../../scripts/nolints.json) uses Batteries' conventional
declaration/linter exception format. The retained proof closure has **47 existing findings**:
32 unused arguments, 10 missing definition/field docstrings, 3 definition names
containing underscores and 2 simp-normal-form findings. It suppresses only those
named declaration/rule pairs; new findings fail. This is an adoption baseline, not a
claim that those issues are repaired. Unused assumptions can be part of reviewed
mathematical interfaces: do not change a theorem statement to pass lint.

Do not run Batteries `--update` as an automatic fix: it records current failures as
exceptions and replaces the entire file on each module run. Remove individual entries
when repairing debt; justify any additions in review. No gate updates this file.

`just lean::lint-builtin` runs native Lake linting on all four roots and currently
fails on pre-existing syntax warnings. The equivalent native
`just lean::quality-report` produces a JSON stream. The report does not fail on
findings; build/report execution errors still fail CI. Native environment linting
only runs registered native linters; Batteries' environment linters run separately.

## Import and trust tools

- `just lean::imports`: community `import-graph` unused-transitive-import analysis.
  Advisory; it returns success even when it prints candidates. For local proof work,
  import `ImportGraph.Tools` and use `#min_imports` or `#redundant_imports` temporarily.
  Rebuild after edits: tactics, syntax and deliberate public APIs need manual review.
  The recipe builds its upstream modules and runs the upstream entry point through
  `lean --run`; its native executable crashed in this pinned Nix toolchain.
- `just lean::shake`: upstream `lake shake --keep-public`. It currently rejects legacy
  files without a `module` header. Converting the entire proof library to the module
  system is separate work; this command is not a CI gate.
- `just lean::warnings`: upstream warning-as-error build. Existing warnings mean this is stricter than the maintained build policy.
- `just lean::kernel-check`: upstream sandboxed `lake check`. Requires bubblewrap and
  permitted user namespaces. This optional check replays proofs independently;
  no sandbox or axiom checks are disabled.

The sorry/True inventory, publication claims and dependency pin audit retain
their distinct contracts. Generic linters do not replace these or architecture policy.
Third-party formatters and whole-program dead-code tools are not required dependencies.

## Implementation boundary

`Style.lean` delegates all rules and fixes to Mathlib's public `lintModules` API.
The upstream `lint-style` CLI assumes module paths are relative to the root package;
our package sources live under `packages/<package>/`. The adapter supplies explicit
paths, checks nonempty input and verifies remaining findings after fixing. It defines
no lint rules. Tool directories containing hyphens are checked from their own working
directory because they cannot be represented as Lean module identifiers. The adapter
runs through `lean --run`; its native executable also crashed under this toolchain.
`scripts/quality.sh` only orchestrates the upstream commands and paths.
