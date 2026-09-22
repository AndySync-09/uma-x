# UMA-X

**Transformer-aware virtual memory and heterogeneous compute for local AI.**

UMA-X is an open-source systems project exploring a simple idea: treat the whole PC as an AI execution hierarchy instead of treating local inference as a GPU-only problem.

The target user experience is intentionally small:

```bash
umax run model.gguf
```

Underneath that command, UMA-X is designed to learn a transformer's access pattern and coordinate **NVMe storage, system memory, CPU, integrated GPU, and NPU** as one logical AI device.

## Status

**Research prototype — v0.1 bootstrap.**

The first milestone is deliberately narrow:

1. Establish a reproducible llama.cpp baseline on a small GGUF model.
2. Run the same workload through a UMA-X-backed path.
3. Compare time-to-first-token, generation tokens/s, prompt throughput, memory use, and run-to-run variance.
4. Only claim improvements that are reproduced from captured benchmark artifacts.

UMA-X does **not** currently claim that SSD is equivalent to RAM, that all accelerators are interchangeable, or that large-model performance targets have been achieved.

## Design thesis

UMA-X is being built as a **transformer-aware virtual-memory and heterogeneous-compute subsystem**, not as another chat application.

The runtime will evolve around three feedback loops:

- **Memory intelligence** — tensor residency, prefetch distance, eviction priority, KV-cache pressure.
- **Compute intelligence** — CPU / iGPU / NPU placement based on measured capability and transfer cost.
- **Generation intelligence** — batching and speculative verification policies informed by observed execution.

The initial implementation uses llama.cpp as the reference inference engine. UMA-X will add an adapter/backend layer rather than reimplementing mature model kernels before the hypothesis is proven.

## Repository principles

- Benchmark first.
- Same model, same engine revision, same prompt, same token budget.
- No benchmark-specific shortcuts.
- No hidden cloud fallback.
- No inflated "unified memory" claims: UMA-X models each tier's real latency and bandwidth.
- Windows is the first-class target; portable core code is preferred where practical.
- Performance claims require reproducible evidence.

## License

Apache-2.0. See `LICENSE`.
