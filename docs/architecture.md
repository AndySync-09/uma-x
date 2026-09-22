# UMA-X architecture

## Definition

UMA-X is a **transformer-aware virtual-memory and heterogeneous-compute subsystem** for local AI.

The long-term abstraction is one logical AI device backed by unequal physical tiers:

- NVMe: large, persistent backing capacity.
- System memory: hot tensor and KV working set.
- CPU: orchestration and suitable compute.
- Integrated GPU: primary high-throughput model compute where beneficial.
- NPU: independent or compatible workloads where end-to-end measurements justify placement.

UMA-X must never pretend these tiers have equal latency or bandwidth.

## Why llama.cpp first

v0.x uses llama.cpp as the reference inference engine. Reimplementing mature kernels would make it difficult to determine whether gains came from UMA-X's memory/compute policies or from unrelated kernel work.

The intended integration path is:

```
umax CLI
   |
UMA-X planner / profiler
   |
UMA-X GGML backend (future milestone)
   |
llama.cpp graph + kernels
   |
NVMe <-> RAM <-> CPU / iGPU / NPU
```

The v0.1 bootstrap is intentionally earlier than that diagram: it is an instrumented wrapper around the same llama.cpp executable. Its purpose is to establish a reproducible control experiment and artifact format.

## Model intelligence

UMA-X will learn at three levels.

### Static model map

From GGUF metadata and graph construction:

- tensor names, shapes and sizes;
- transformer block ordering;
- quantization format;
- model architecture;
- expected KV-cache growth.

### Dynamic access trace

Once the GGML backend exists, the runtime will capture:

- tensor use order;
- next-use distance;
- residency duration;
- prefetch hit/miss;
- bytes promoted and evicted;
- compute-device placement;
- stall time attributable to storage or migration.

### Machine-specific execution profile

UMA-X persists a model + hardware profile. The profile is optimization metadata only and never contains model weights.

The first-run profile should evolve toward:

```json
{
  "model_hash": "...",
  "hardware_hash": "...",
  "memory": {
    "resident_budget_bytes": 0,
    "prefetch_distance": 0
  },
  "compute": {
    "target": "igpu",
    "draft": "npu"
  },
  "generation": {
    "speculation_depth": 0
  }
}
```

## Memory fabric

The memory subsystem is software managed and transformer aware.

A future steady-state decode should look conceptually like:

```
execute block N
    |
    +-- block N+1 already resident
    +-- block N+2 being promoted from NVMe
    +-- expired tensors being demoted/evicted
```

The central metric is not cache hit rate alone. It is **useful generated tokens per byte moved through the slowest tier**.

## Compute fabric

UMA-X presents a logical execution device but measures each physical compute target separately. It should prefer coarse, independently useful placement over arbitrary fine-grained splitting that adds synchronization overhead.

Initial research hypothesis:

- iGPU: target-model matrix-heavy execution;
- NPU: draft model or independent compatible subgraphs;
- CPU: scheduling, tokenization, sampling, I/O and fallback operations.

This is a hypothesis to benchmark, not a fixed policy.

## Benchmark contract

Every performance claim must preserve:

- identical model file;
- identical llama.cpp revision/binary;
- identical prompt;
- identical generated-token budget;
- identical context size;
- identical sampling/seed where supported;
- same machine and power mode;
- repeated runs;
- raw logs and machine-readable artifacts.

v0.1 compares **plain llama.cpp** against **the same llama.cpp binary invoked through UMA-X observe mode**. A near-zero delta is expected. That result is valuable: it validates that future improvements are measured against a stable control.

## Roadmap

### M0 — control experiment

- CLI: `umax run model.gguf`
- `umax doctor`
- deterministic A/B benchmark harness
- persistent execution profile
- raw benchmark artifacts

### M1 — GGML instrumentation backend

- tensor-level event stream
- static GGUF tensor map
- model/hardware fingerprint
- residency and stall telemetry

### M2 — transformer-aware RAM residency

- bounded working-set manager
- predictive prefetch
- eviction based on next-use distance
- NVMe staging
- compare against llama.cpp mmap/offload baseline

### M3 — heterogeneous compute planner

- measured CPU/iGPU/NPU capability table
- placement cost model
- coarse workload scheduling

### M4 — generation intelligence

- speculative draft/verify path
- adaptive draft depth
- optimize verified output tokens per target-model pass

### M5 — large-model qualification

Scale only after smaller models demonstrate reproducible gains. 70B and 20 tok/s remain research targets until measured on a disclosed configuration.
