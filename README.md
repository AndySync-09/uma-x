# UMA-X

**Transformer-aware virtual memory and heterogeneous compute for local AI.**

UMA-X is an open-source systems project exploring a simple idea: treat the whole PC as an AI execution hierarchy instead of treating local inference as a GPU-only problem.

The target experience stays simple:

~~~powershell
umax run model.gguf
~~~

Underneath that command, UMA-X is designed to learn a transformer's access pattern and coordinate **NVMe storage, system memory, CPU, integrated GPU, and NPU** as one logical AI device.

## What UMA-X is

UMA-X is being built as a **transformer-aware virtual-memory and heterogeneous-compute subsystem**.

It is not another chat UI, and v0.1 is not a replacement inference engine. llama.cpp remains the reference engine while UMA-X develops the parts that are different:

- model-aware memory placement;
- predictive tensor prefetch;
- next-use-aware eviction;
- hardware cost measurement;
- CPU / iGPU / NPU placement;
- adaptive generation policy.

If llama.cpp later becomes the limiting layer, native UMA-X kernels can be added only where measurements justify them.

## Current milestone — v0.1

v0.1 establishes the control experiment before custom paging is introduced.

Available commands:

~~~powershell
umax doctor
umax run model.gguf
umax bench model.gguf
~~~

### Install the development build

~~~powershell
git clone https://github.com/AndySync-09/uma-x.git
cd uma-x
python -m pip install -e .
~~~

Install or build a recent llama.cpp separately and either put `llama-cli` and `llama-bench` on PATH or point UMA-X at its binary directory:

~~~powershell
umax doctor --llama-bin-dir C:\path\to\llama.cpp\bin
~~~

### First benchmark

Start with a small 1B-2B Q4 GGUF. The goal is to validate the measurement path before moving upward in model size.

~~~powershell
umax bench C:\models\small-model.gguf --llama-bin-dir C:\path\to\llama.cpp\bin
~~~

UMA-X runs:

1. plain `llama-bench` as the control;
2. the same llama.cpp build and same GGUF through the UMA-X planner;
3. multiple repetitions for prompt processing and text generation;
4. a saved JSON artifact containing the exact commands, raw benchmark rows, machine snapshot, and UMA-X plan.

v0.1 benchmark differences are **planner/control-path results**, not proof of transformer-aware virtual-memory acceleration. That subsystem begins in v0.2/v0.3.

## Model Intelligence

UMA-X learning is scoped to a specific **model + machine** pair.

The first profile records benchmark observations. Later native milestones will add real transformer access observations such as:

- tensor next-use distance;
- resident lifetime;
- prefetch lead time;
- storage stall time;
- effective RAM bandwidth;
- device execution cost;
- KV-cache pressure;
- speculative acceptance and verification cost.

This allows the execution plan to improve for the machine actually running the model rather than relying on one hard-coded configuration.

## Repository principles

- Benchmark first.
- Same model, engine revision, prompt, and token budget.
- No benchmark-specific shortcuts.
- No hidden cloud fallback.
- Do not describe SSD as RAM or combine capacities without exposing the hierarchy.
- Do not count rejected speculative tokens as output.
- Windows is the first-class target; portable core code is preferred.
- Performance claims require reproducible artifacts.

## Research target

The long-term research target is efficient local inference up to 70B-class models on Windows PCs using the entire local memory/compute hierarchy.

**No 70B / 20 generated-token/s claim has been made or verified.**

See [Architecture](docs/architecture.md), [Benchmarking](docs/benchmarking.md), and [Roadmap](ROADMAP.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
