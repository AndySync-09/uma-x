# UMA-X architecture

UMA-X is a **transformer-aware virtual-memory and heterogeneous-compute subsystem** for local AI.

The project starts with llama.cpp as the reference inference engine. UMA-X does not reimplement mature matrix kernels before the memory-and-scheduling hypothesis is proven.

## Logical device

The intended abstraction is:

~~~text
                 llama.cpp
                    |
               UMA-X adapter
                    |
          +---------+---------+
          |  UMA-X logical    |
          |     AI device     |
          +---------+---------+
                    |
        +-----------+-----------+
        |                       |
  Memory fabric           Compute fabric
        |                       |
 NVMe <-> RAM             CPU / iGPU / NPU
        |                       |
        +-----------+-----------+
                    |
             Model intelligence
~~~

The logical device does not erase physical differences. UMA-X keeps explicit cost models for capacity, latency, bandwidth, transfer/conversion cost, and supported operations.

## Why transformer-aware

Generic virtual memory waits for demand. Transformer execution exposes substantial structure before the next access occurs.

UMA-X will combine two information sources.

### Static model knowledge

Before generation begins, UMA-X can inspect model metadata and build a topology map:

- tensor names and sizes;
- transformer blocks;
- attention and feed-forward structure;
- quantization;
- estimated working set;
- KV-cache growth.

### Dynamic machine knowledge

During execution, UMA-X will learn model- and machine-specific behavior:

- observed tensor next-use distance;
- residency duration;
- prefetch lead time;
- storage latency and throughput;
- effective RAM bandwidth;
- device execution time;
- transfer and synchronization cost;
- KV pressure;
- speculative acceptance and verification cost.

The learned profile is scoped to the model fingerprint and hardware fingerprint.

## Three feedback loops

### Memory intelligence

Decides what should be resident, prefetched, demoted, or evicted.

Long-term policy inputs include tensor size, next-use distance, cost to reload, reuse probability, current memory pressure, and KV growth.

### Compute intelligence

Selects CPU, iGPU, or NPU based on measured end-to-end cost rather than advertised TOPS.

The first implementation should prefer coarse independent work over arbitrary per-layer accelerator splitting. For example, the iGPU can run the target model while an NPU later runs a compatible draft model.

### Generation intelligence

Tunes generation-level choices such as batch size and speculative depth. The objective is verified output tokens per second, not raw candidate throughput.

## v0.1 boundary

v0.1 contains the control plane and benchmark harness only.

It deliberately does not claim:

- custom tensor paging;
- predictive NVMe prefetch;
- a native GGML backend;
- NPU execution;
- speculative decoding gains.

The v0.1 benchmark compares ordinary llama.cpp defaults with an explicit UMA-X planner path and records both commands.

## Native backend path

Once the baseline is reproducible, the next native milestone is a GGML backend/adapter, provisionally:

~~~text
ggml-umax
~~~

Its job will be to expose enough execution and allocation events to build a real access trace without changing model semantics.

The first native experiment should answer one question:

> Can a model-aware prefetch schedule reduce observable stalls compared with ordinary OS-backed mapping under controlled memory pressure?

Only after that result is measured should UMA-X add more complex compute placement.
