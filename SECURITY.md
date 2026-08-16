# Security policy

## Supported versions

Security fixes are applied to the latest tagged release and `main`. Pre-1.0 versions may receive source-compatible fixes where practical; a fix may require a minor-version API correction when safety demands it.

## Reporting

Report suspected vulnerabilities privately through GitHub Security Advisories for `ezefranca/ml5.swift`. Include affected versions, platform/toolchain, reproduction, impact, and any proposed mitigation. Do not open a public issue before coordinated disclosure. Never attach confidential data, credentials, or proprietary model weights.

## Response

Maintainers will acknowledge a report, reproduce and assess it, prepare tests and a fix, coordinate disclosure, publish a checksummed release with provenance, and run post-release client/documentation/SPI verification. Compromised releases are documented and yanked according to `Documentation/Releasing.md`.

## Boundaries

ML5 processes untrusted model packages, archives, datasets, images, and remote URLs. Remote sources require HTTPS, a caller-provided SHA-256 digest, ownership, license, provenance, and version metadata; compiled cache entries are rehashed. Validation, bounded caches, cancellation, typed failures, Core ML sandboxing, and explicit accelerator policy reduce risk but do not establish model safety or trust. Host applications remain responsible for server trust, data privacy, model evaluation, fairness, output policy, and platform entitlements. The package contains no telemetry, analytics SDK, advertising identifier use, credentials, or third-party production model.
