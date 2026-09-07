# Changelog

All notable changes to this package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed

- **`FactorModelStructures` package** (entire directory deleted).
  All definitions migrated to `PaperReconstructions/Ross1976/`.
  The namespace changed from `FactorModelStructures` to `PaperReconstructions`.

### Added

- `Ross1976/Types.lean` — `ReturnVec`, `FactorVec` type aliases (canonical location).
- `Ross1976/Model.lean` — `GeneralizedFactorModel`, `OneFactorModel`, `NoArbitrage`,
  `IsZeroCost`, `NoArbitrageFTAP`, `IsEMMFTAP`, `ftap_one_period_vector_iff` (canonical).
- `Ross1976/Moments.lean` — `var`, `cov`, `cov_const_mul_left`, `expectedReturn`,
  `beta`, covariance algebra (canonical).
- `Ross1976/Ross1976.lean` — Ross (1976) APT theorems moved from `Ross1976.lean`.
- `CHANGELOG.md` — migration history and eliminated-package record.

### Changed

- All paper modules (`ChamberlainRothschild1983`, `Connor1984`, `Ingersoll1984`,
  `MiddletonSatchell2000`, `Reisman1992`) now import from
  `PaperReconstructions.Ross1976.Model` / `.Moments` instead of
  `FactorModelStructures.Model` / `.Moments`.
- `open FactorModelStructures` replaced with `open PaperReconstructions` in all
  paper modules.
- All `FactorModelStructures.*` qualified references replaced with unqualified
  names (resolvable via `open PaperReconstructions`).
- `PricingPerspective/Discrete/Model.lean`, `Moments.lean`, `Arbitrage.lean`
  are now thin re-export wrappers importing from `PaperReconstructions.Ross1976.*`.
- `PricingPerspective/Types.lean` now imports and re-exports
  `PaperReconstructions.Ross1976.Types` for `ReturnVec`/`FactorVec`.
- `PricingPerspective/lakefile.lean` now requires `PaperReconstructions`.
- `lake-manifest.json` — removed stale `FactorModelStructures` entries from
  root and `EnergyStatistics` manifests.
