# Changelog

## [2.4.68] - 2026-11-08

### Added
- Evidence-labeled code references for static rules, runtime-correlated candidates, parameter locations, and related pipeline call sites.
- A repeatable synthetic fault-campaign runner and its results artifact.
- More live-run findings for abstention, output-format validation, prompt-injection signals, arithmetic checks, and code-level localization.

### Changed
- The extension evaluation view distinguishes reference observation from reference-supported evidence.
- The campaign summary stores a repository-relative results path to avoid exposing the machine username.

### Notes
- The synthetic campaign checks deterministic rules against seeded templates. Its score is not a real-world accuracy claim.
- The extension still requires a local Python environment with the documented analyzer dependencies.
