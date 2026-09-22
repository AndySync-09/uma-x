# Native UMA-X

This directory is reserved for the performance-critical native subsystem.

The first native component will be a GGML integration that can observe tensor allocation/execution and feed a transformer-aware memory planner.

Do not add custom matrix kernels merely to duplicate llama.cpp. Native code belongs here only when it implements an UMA-X responsibility such as:

- tensor residency;
- asynchronous prefetch;
- eviction;
- access tracing;
- hardware cost measurement;
- execution placement;
- synchronization needed by those features.

The v0.1 Python package is the control plane and benchmark harness.
