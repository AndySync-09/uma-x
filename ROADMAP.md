# Roadmap

## v0.1 — measurement first

- CLI: doctor, run, bench.
- Machine and llama.cpp device discovery.
- Reproducible plain llama.cpp vs UMA-X planner comparison.
- Model/hardware-scoped benchmark profiles.
- Windows + Linux CI.

Exit criterion: the same benchmark can be reproduced from a saved artifact.

## v0.2 — access trace

- Add the first native GGML integration.
- Capture tensor allocation and execution order.
- Build static GGUF topology metadata.
- Persist next-use and residency observations.

Exit criterion: UMA-X can produce a real tensor access trace for a small model without changing output.

## v0.3 — transformer-aware memory

- Add a bounded UMA-X resident tensor cache.
- Add asynchronous predictive prefetch.
- Add next-use-aware eviction.
- Measure under controlled RAM pressure.

Exit criterion: demonstrate a reproducible reduction in storage-induced stalls versus the control path.

## v0.4 — Intel iGPU path

- Optimize shared-memory placement for Intel integrated graphics.
- Measure copy avoidance and synchronization overhead.
- Add device-specific policy only when backed by measurements.

Exit criterion: measurable end-to-end gain on a documented Intel Windows reference PC.

## v0.5 — NPU auxiliary execution

- Probe supported NPU workloads.
- Keep the target model on the best measured device.
- Evaluate a small draft model or other independent auxiliary workload on the NPU.

Exit criterion: NPU use improves end-to-end verified generation speed rather than only NPU utilization.

## v0.6 — adaptive speculative execution

- Add draft/verify scheduling.
- Learn useful speculative depth per model, machine, and workload.
- Optimize accepted target-verified tokens per expensive target pass.

## Large-model research gate

Only after the small-model path is reproducible:

1. 7B
2. 14B
3. 32B
4. 70B

The 70B / 20 generated-token/s objective remains a research target until independently reproduced on a fully disclosed configuration.
