# Publication claim adapter

Native LeanArchitect annotations beside the original project declarations own the exact
manuscript label → declaration association and statement summary. The root
`Publication` library retains the external mathlib annotation and imports the
annotated project modules. Native `Publication`, `PricingPerspective`, and
`WassersteinGeometry` library facets export their own annotations; the companion
Blueprint consumes their native TeX nodes.

`mapping.toml` and `*-publication.toml` retain publication ownership, assumptions,
and identifier-free disclosure text. They do not repeat declarations, summaries,
or source locations. Every native label must occur exactly once across those
registries, and every registry label must exist in the fresh native export.
Duplicate, missing, and unclaimed labels fail extraction. Native nodes must
explicitly carry `notReady: false` and `hasLean: true`; absent or incorrectly typed
flags fail. Project annotations must originate in their declaration module. Only
imported `Mathlib.*` declarations may be annotated in `Publication.Claims`.

`extract.py` rebuilds all three native JSON library facets before reading their library/module
exports. It derives each declaration's original source module and package from
native metadata, then independently checks its elaborated statement and
transitive axioms through Lean. It reuses the compiled `sorry-audit/Audit.lean`
classifier to reject mapped `True` conclusions, aliases, definitions,
equivalences, and premises, including globally inventoried scaffolds. Native
`leanOk` markers never establish publication safety. Only `propext`,
`Classical.choice`, and `Quot.sound` are accepted; admissions and custom axioms
fail. The global integrity gate remains a separate required certification step.

The schema and generated JSON contract retain `statement_type`, `axioms`, and
`content_hash`. Hashes include the native declaration, normalized summary, and
independently elaborated statement, so a native binding, metadata, or theorem
statement edit changes the publication identity and requires review.

`render.py` emits only each paper's identifier-free
`src/latex/projects/<paper>/src/generated/lean_status.tex`. Each paper preserves
its owning registry's hash. A withdrawn paper receives an empty projection;
failed claims never receive checked status. No custom Blueprint declaration or
proof-marker projection is retained.

```bash
just lean::certify
just lean::claims-publication-check
uv run src/lean/tools/claim-manifest/tests/test_manifest.py
```

For a new claim, author its native annotation after reviewing the exact theorem
and add its label, paper, assumptions and disclosure to the owning TOML registry.
Run certification and review the resulting manifest and publication projection.
Direct `extract.py` invocation also rebuilds native JSON and validates mapped
contracts, but full certification additionally audits all first-party compiled
declarations. Manuscript prose changes use the cheaper publication check; Paper 6
certification requires both proof and publication checks to be current.
