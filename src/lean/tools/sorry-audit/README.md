---
description: Compiled declaration inventory for sorry dependencies, trivial proposition shapes, and custom axioms.
---

# Compiled proof-integrity gate

Run `bash src/lean/scripts/sorry_gate.sh` from the repository root. The wrapper runs
`audit.py` with the standard library through `uv run --no-project python`. Run its
regression suite with `just lean::sorry-test`; CI executes the same suite before the gate.

The driver discovers every package source module, including modules absent from roll-up
imports, plus the root `Main` and `Publication` modules, including the publication assembly subtree. Before any build it rejects `set_option debug.*`
commands with a lexical scan that handles nested comments, strings, quoted identifiers,
and multiline commands. It then runs `lake build` for every discovered module, so a
source change or a previously unbuilt module cannot be hidden by stale imported oleans.
Compilation errors and incomplete/classifier-error output fail closed.
Source hashes must also remain unchanged throughout the build and census.

`Audit.lean` inspects the compiled environment. Every first-party declaration, including
private declarations and compiler helpers, receives a transitive `Lean.collectAxioms`
check. Any declared custom axiom or dependency outside `propext`, `Classical.choice`,
`Quot.sound`, and the inventoried `sorryAx` fails unconditionally; custom axioms cannot
be authorized by adding an allowlist entry. Direct and macro-expanded admissions become
ordinary `sorryAx` dependencies and are therefore visible.

The allowlist names declarations instead of counting words or lines:

```text
sorry Namespace.unfinished
true-result Namespace.stub
true-iff Namespace.placeholder_equivalence
true-definition Namespace.placeholder_predicate
true-premise Namespace.conditional_theorem:h_stub
```

Blank lines and whole-line `#` comments are permitted. Duplicate, malformed, missing,
new, and stale entries fail. An intentional `True` predicate also needs a named reviewed
exception; there is no implicit exemption for modeling definitions.

The semantic classifier unfolds aliases and checks theorem/proof-definition conclusions
that reduce to `True`, a top-level equivalence with a side reducing to `True`, Prop-valued
definitions reducing to `True`, and statement binders whose premise reduces to
`True`. Ordinary prose and a local `have h : True` do not change the theorem's statement
and produce no exception. Every compiled declaration receives these checks, including
private names, compiler helpers, and declarations emitted by macros or elaborators.
Names and source-name selections never grant an implicit exemption. A generated
declaration with one of these shapes needs its own reviewed inventory row.

For a concrete review of current rows and optional fresh JSON evidence:

```bash
bash src/lean/scripts/sorry_gate.sh --inventory --output /tmp/lean-integrity-new.json
```

`--inventory` still enforces preflight, fresh compilation, and the trust policy. It only
omits comparison with the reviewed allowlist; it never edits that file automatically.

These shape checks identify review obligations, not all mathematical misrepresentation.
They cannot establish that an arbitrary theorem formalizes its English description,
detect every contradictory economic hypothesis, or reject legitimate uses of `True` by
intent. Publication statement hashes and human semantic review remain complementary.
The gate trusts the pinned Lean compiler and its compiled dependencies and does not replay
every proof through an independent kernel checker. Its lexical debug preflight is not a
sandbox for arbitrary malicious metaprograms or Lake configuration code.
