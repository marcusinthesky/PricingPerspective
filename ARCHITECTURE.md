# Architecture

The repository co-locates four papers with their numerical computations, formal
proofs and static website so a scientific change can be reviewed across all four
layers.

## Reproduction boundary

The DVC graph begins at checksummed numeric observations: computed document
vectors, identifiers/dates, adjusted returns, classifications and derived
controls. A schema/checksum gate admits only the published text-free vector
representation. Removing article text changes storage and provenance; it does
not change vector values, sample membership or the numerical estimators.

Each expensive analysis writes reusable numeric results. Separate render stages
produce manuscript tables, scalar bindings and figures from those saved results.
Fixed historical numerical benchmarks are inputs with an explicit archival role;
they are not reported as newly computed campaigns.

## Computation and proofs

The uv workspace contains `pipeline`, `hotelling` and `dotell`. Statadic is an
external pinned distribution. The application owns scientific configuration,
cohort selection and publication contracts; the dependency owns its general
statistical and numerical implementations. Each process explicitly initializes
its required float64 policy, and tests execute per package.

Semflow fingerprints the reachable repository source for computational
entrypoints. The numeric co-mention adapter and input-validation gate instead
track their complete source and lockfile dependencies directly in DVC.
Dependency versions remain locked; changing those versions requires replaying
the affected computations rather than assuming source fingerprints establish
numerical equivalence.

Lean imports the public Statadic proof package at an immutable Git revision.
EnergyStatistics supplies the empirical energy and explicit Gaussian embedding
bridge. WassersteinGeometry supplies the general transport-cost results needed
by the paper-specific proofs. These packages retain the normalization and
hypotheses used by the manuscripts. Certification checks compiled statements and
axioms; manuscript correspondence separately checks claim labels and disclosures.
These checks complement numerical tests rather than proving floating-point code.

## Environment and publishing

A pinned Nix package inventory supplies devenv profiles for development, checks,
Lean and reproduction. The same inventory drives the public tool listing.
Insitu is externally installed and renders documentation trees. `just nix::tools`
lists the pinned tools. These documentation views have no role in the scientific
estimands.

LaTeX is the manuscript source, Tectonic is its renderer, and the required TeX
bundle is included for offline compilation. Citation-selected bibliographies
come from CSL frontmatter records. Source bundles stage only the required paper
files and assets.

The website registry owns the four paper identities, descriptions and links.
Pages, structured data and machine-readable summaries derive from that registry.
Lean Blueprint pages are regenerated from the retained, certified proof tree.
