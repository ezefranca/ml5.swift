# Performance baseline methodology

Performance checks are regression tripwires, not universal throughput claims. `Configuration/PerformanceBudgets.json` gives ML5's canonical XOR/affine/CPU-Metal quality suite a broad 45-second wall-clock ceiling including test-process startup.

The 2026-08-16 Apple M4 reference completed the focused suite in 0.65 seconds; the final integrated run completed it in 0.85 seconds. Numerical tests separately assert convergence, held-out quality, deterministic histories, and CPU/MPSGraph tolerances. Release notes record hardware, OS, toolchain, thermal context, observed time, and budget result.

For tagged candidates, run `Scripts/run_instruments_audits.sh` with Time Profiler, Allocations, Leaks, Core ML, and Power Profiler after granting the host's protected process-analysis permission. Record findings and never commit trace archives. Deterministic lifetime tests, sanitizer jobs, bounded model caches, and explicit eviction remain CI gates.
