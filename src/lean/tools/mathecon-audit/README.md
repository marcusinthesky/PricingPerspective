# Dependency pin audit

`just lean::dependency-audit` verifies the four tracked Lake manifests and their
matching Lean toolchains. It also checks direct public Statadic requirements,
including EnergyStatistics standalone wiring, its immutable revision and
`src/lean` subdirectory. Negative tests reject missing or changed requirements,
source URLs, revisions, manifests and compiler versions.
