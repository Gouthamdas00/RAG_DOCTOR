# Security policy

## Reporting a vulnerability

Please do not report security vulnerabilities in a public issue. Use GitHub's private vulnerability reporting for this repository when available. If private reporting is unavailable, contact the maintainer privately through the GitHub profile for `Gouthamdas00` and include steps to reproduce, affected versions, and a suggested severity.

## Supported versions

Security fixes are currently provided for the latest published version. Before a public release, the maintainer will add a supported-version policy.

## Local-data handling

RAG Doctor is designed to analyze workspace data locally. Content capture is opt-in, and included run content is retained in analyzer process memory for the active session. Secret and PII scrubbing is best-effort; it is not a guarantee that all sensitive values will be detected. Review any content before enabling content-level diagnosis.
