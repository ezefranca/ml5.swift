# Performance baseline methodology

Performance checks are regression tripwires, not universal throughput claims. `Configuration/PerformanceBudgets.json` gives ML5's canonical XOR/affine/CPU-Metal quality suite a broad 45-second wall-clock ceiling including test-process startup.

The 2026-08-16 Apple M4 reference completed the focused suite in 0.65 seconds; the final integrated run completed it in 0.85 seconds. Numerical tests separately assert convergence, held-out quality, deterministic histories, and CPU/MPSGraph tolerances. Release notes record hardware, OS, toolchain, thermal context, observed time, and budget result.

For tagged candidates, run `Scripts/run_instruments_audits.sh` with Time Profiler,
Allocations, Leaks, and Core ML. The script ad-hoc signs a temporary audit
executable with Apple's debug entitlement and deletes it afterward; it does not
change system security policy. Record findings and never commit trace archives.
Deterministic lifetime tests, sanitizer jobs, bounded model caches, and explicit
eviction remain CI gates.

## Current Instruments acceptance

On 2026-08-16, the release smoke workload completed readable 15.6-second Time
Profiler, Allocations, Leaks, and Core ML archives on an Apple M4 Mac running
macOS 26.6.1. `xctrace` reported no run issue for any archive; the Leaks table
contained no exported leak-row schema. Power Profiler explicitly requires a
physical iOS/iPadOS application target and remains a tagged-device release check
rather than a macOS package gate.
